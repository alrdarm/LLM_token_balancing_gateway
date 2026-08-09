"""Database-enforced invariants.

These constraints are the last line of defence: application code is expected to
uphold them, but concurrency and future call sites make "expected" insufficient
for anything that guards money or replay safety.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import (
    AttemptKind,
    BudgetWindow,
    Endpoint,
    IdempotencyState,
    RequestState,
    ReservationStatus,
)
from gateway.persistence.models import (
    Attempt,
    Budget,
    BudgetReservation,
    IdempotencyRecord,
    Request,
)
from gateway.persistence.unit_of_work import unit_of_work

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _request(request_id: str) -> Request:
    return Request(
        id=request_id,
        client_id="client_a",
        endpoint=Endpoint.CHAT_COMPLETIONS.value,
        requested_model="auto",
        normalized_controls_json={},
        state=RequestState.RECEIVED.value,
        deadline_at=NOW + timedelta(seconds=30),
        input_hash="a" * 64,
        max_cost=Decimal("1.000000000"),
    )


def _attempt(attempt_id: str, request_id: str, sequence: int) -> Attempt:
    return Attempt(
        id=attempt_id,
        request_id=request_id,
        sequence=sequence,
        model_id="fake/general",
        route_rank=1,
        attempt_kind=AttemptKind.GENERATION.value,
    )


def test_attempt_sequence_is_unique_per_request(session_factory: sessionmaker[Session]):
    """Two attempts claiming the same position would corrupt the audit order."""
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_seq"))
        uow.flush()
        uow.requests.add_attempt(_attempt("att_a", "req_seq", 1))

    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            uow.requests.add_attempt(_attempt("att_b", "req_seq", 1))


def test_same_sequence_in_different_requests_is_allowed(
    session_factory: sessionmaker[Session],
):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_one"))
        uow.requests.add(_request("req_two"))
        uow.flush()
        uow.requests.add_attempt(_attempt("att_one", "req_one", 1))
        uow.requests.add_attempt(_attempt("att_two", "req_two", 1))

    with unit_of_work(session_factory) as uow:
        assert len(uow.requests.attempts_for("req_one")) == 1
        assert len(uow.requests.attempts_for("req_two")) == 1


def test_reservation_is_unique_per_attempt_and_budget(
    session_factory: sessionmaker[Session],
):
    """Stops a retry from double-reserving against the same budget (§6)."""
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_res"))
        budget = uow.budgets.add(
            Budget(
                scope="global",
                window_kind=BudgetWindow.TOTAL.value,
                window_start=NOW,
                hard_limit=Decimal("5"),
            )
        )
        uow.flush()
        uow.budgets.add_reservation(
            BudgetReservation(
                request_id="req_res",
                attempt_id="att_dup",
                budget_id=budget.id,
                reserved_amount=Decimal("0.001"),
                status=ReservationStatus.ACTIVE.value,
                expires_at=NOW + timedelta(minutes=5),
            )
        )
        budget_id = budget.id

    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            uow.budgets.add_reservation(
                BudgetReservation(
                    request_id="req_res",
                    attempt_id="att_dup",
                    budget_id=budget_id,
                    reserved_amount=Decimal("0.001"),
                    status=ReservationStatus.ACTIVE.value,
                    expires_at=NOW + timedelta(minutes=5),
                )
            )


def test_idempotency_key_is_unique_per_client(session_factory: sessionmaker[Session]):
    """T07 in §12: the same key twice must not create two records."""
    with unit_of_work(session_factory) as uow:
        uow.idempotency.add(
            IdempotencyRecord(
                client_id="client_a",
                idempotency_key="shared",
                input_hash="a" * 64,
                state=IdempotencyState.IN_PROGRESS.value,
                expires_at=NOW + timedelta(hours=24),
            )
        )

    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            uow.idempotency.add(
                IdempotencyRecord(
                    client_id="client_a",
                    idempotency_key="shared",
                    input_hash="b" * 64,
                    state=IdempotencyState.IN_PROGRESS.value,
                    expires_at=NOW + timedelta(hours=24),
                )
            )


def test_same_key_for_different_clients_is_allowed(session_factory: sessionmaker[Session]):
    """Keys are client-scoped; one tenant must not block another's key."""
    with unit_of_work(session_factory) as uow:
        for client in ("client_a", "client_b"):
            uow.idempotency.add(
                IdempotencyRecord(
                    client_id=client,
                    idempotency_key="same-key",
                    input_hash="a" * 64,
                    state=IdempotencyState.IN_PROGRESS.value,
                    expires_at=NOW + timedelta(hours=24),
                )
            )

    with unit_of_work(session_factory) as uow:
        assert uow.idempotency.find("client_a", "same-key") is not None
        assert uow.idempotency.find("client_b", "same-key") is not None


def test_unknown_request_state_is_rejected(session_factory: sessionmaker[Session]):
    """A typo'd state would make the lifecycle unanalysable."""
    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            request = _request("req_bad_state")
            request.state = "NOT_A_STATE"
            uow.requests.add(request)


def test_negative_budget_limit_is_rejected(session_factory: sessionmaker[Session]):
    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            uow.budgets.add(
                Budget(
                    scope="bad",
                    window_kind=BudgetWindow.DAILY.value,
                    window_start=NOW,
                    hard_limit=Decimal("-1"),
                )
            )


def test_foreign_keys_are_enforced(session_factory: sessionmaker[Session]):
    """SQLite ignores foreign keys unless the pragma is set on every connection."""
    with pytest.raises(IntegrityError):
        with unit_of_work(session_factory) as uow:
            uow.requests.add_attempt(_attempt("att_orphan", "req_does_not_exist", 1))


def test_deleting_a_request_cascades_to_attempts(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_cascade"))
        uow.flush()
        uow.requests.add_attempt(_attempt("att_cascade", "req_cascade", 1))

    with unit_of_work(session_factory) as uow:
        request = uow.requests.get("req_cascade")
        assert request is not None
        uow.session.delete(request)

    with unit_of_work(session_factory) as uow:
        assert uow.requests.attempts_for("req_cascade") == []
