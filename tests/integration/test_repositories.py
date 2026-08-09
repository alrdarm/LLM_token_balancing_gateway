"""Repository queries and transaction behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import (
    AttemptKind,
    BudgetWindow,
    Endpoint,
    IdempotencyState,
    RequestState,
    ReservationStatus,
)
from gateway.persistence.engine import immediate_transaction
from gateway.persistence.models import (
    Attempt,
    Budget,
    BudgetReservation,
    IdempotencyRecord,
    QuotaSnapshot,
    Request,
)
from gateway.persistence.unit_of_work import unit_of_work

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _request(request_id: str, state: RequestState = RequestState.RECEIVED) -> Request:
    return Request(
        id=request_id,
        client_id="client_a",
        endpoint=Endpoint.CHAT_COMPLETIONS.value,
        requested_model="auto",
        normalized_controls_json={},
        state=state.value,
        deadline_at=NOW + timedelta(seconds=30),
        input_hash="a" * 64,
        max_cost=Decimal("1"),
    )


def test_next_attempt_sequence_starts_at_one(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_seq"))
        uow.flush()
        assert uow.requests.next_attempt_sequence("req_seq") == 1


def test_next_attempt_sequence_advances(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_seq"))
        uow.flush()
        for sequence in (1, 2):
            uow.requests.add_attempt(
                Attempt(
                    id=f"att_{sequence}",
                    request_id="req_seq",
                    sequence=sequence,
                    model_id="fake/general",
                    route_rank=1,
                    attempt_kind=AttemptKind.GENERATION.value,
                )
            )
        uow.flush()
        assert uow.requests.next_attempt_sequence("req_seq") == 3


def test_list_in_state_filters_and_orders(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_a", RequestState.INVOKING))
        uow.requests.add(_request("req_b", RequestState.SUCCEEDED))
        uow.requests.add(_request("req_c", RequestState.INVOKING))

    with unit_of_work(session_factory) as uow:
        invoking = uow.requests.list_in_state(RequestState.INVOKING)
        assert {request.id for request in invoking} == {"req_a", "req_c"}


def test_price_at_selects_the_effective_row(session_factory: sessionmaker[Session]):
    """A superseded price must still apply to a request that ran under it."""
    from gateway.persistence.models import Model, ModelPrice

    with unit_of_work(session_factory) as uow:
        model = Model(
            id="fake/priced",
            provider="fake",
            provider_model_id="priced",
            display_name="Priced",
            data_handling_tier="standard",
            quality_tier="standard",
            capabilities=[],
            supported_endpoints=[],
            context_window_tokens=1000,
            max_output_tokens=100,
        )
        model.prices.append(
            ModelPrice(
                input_per_1k_tokens=Decimal("0.001"),
                output_per_1k_tokens=Decimal("0.002"),
                effective_from=NOW - timedelta(days=10),
                effective_to=NOW,
            )
        )
        model.prices.append(
            ModelPrice(
                input_per_1k_tokens=Decimal("0.005"),
                output_per_1k_tokens=Decimal("0.010"),
                effective_from=NOW,
            )
        )
        uow.models.add(model)

    with unit_of_work(session_factory) as uow:
        old = uow.models.price_at("fake/priced", NOW - timedelta(days=1))
        new = uow.models.price_at("fake/priced", NOW + timedelta(days=1))
        missing = uow.models.price_at("fake/priced", NOW - timedelta(days=20))

        assert old is not None and old.input_per_1k_tokens == Decimal("0.001")
        assert new is not None and new.input_per_1k_tokens == Decimal("0.005")
        assert missing is None


def test_expired_reservations_are_reported_not_resolved(
    session_factory: sessionmaker[Session],
):
    """The reconciler must see them; expiring them blindly could double-bill."""
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_request("req_exp"))
        budget = uow.budgets.add(
            Budget(
                scope="global",
                window_kind=BudgetWindow.TOTAL.value,
                window_start=NOW,
                hard_limit=Decimal("5"),
            )
        )
        uow.flush()
        for suffix, expires in (
            ("old", NOW - timedelta(minutes=1)),
            ("new", NOW + timedelta(minutes=5)),
        ):
            uow.budgets.add_reservation(
                BudgetReservation(
                    request_id="req_exp",
                    attempt_id=f"att_{suffix}",
                    budget_id=budget.id,
                    reserved_amount=Decimal("0.01"),
                    status=ReservationStatus.ACTIVE.value,
                    expires_at=expires,
                )
            )

    with unit_of_work(session_factory) as uow:
        expired = uow.budgets.expired_reservations(NOW)
        assert [reservation.attempt_id for reservation in expired] == ["att_old"]
        assert expired[0].status == ReservationStatus.ACTIVE.value


def test_purge_expired_idempotency_records(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        for key, expires in (
            ("stale", NOW - timedelta(hours=1)),
            ("live", NOW + timedelta(hours=1)),
        ):
            uow.idempotency.add(
                IdempotencyRecord(
                    client_id="client_a",
                    idempotency_key=key,
                    input_hash="a" * 64,
                    state=IdempotencyState.COMPLETED.value,
                    expires_at=expires,
                )
            )

    with unit_of_work(session_factory) as uow:
        assert uow.idempotency.purge_expired(NOW) == 1

    with unit_of_work(session_factory) as uow:
        assert uow.idempotency.find("client_a", "stale") is None
        assert uow.idempotency.find("client_a", "live") is not None


def test_mark_completed_records_the_response_reference(
    session_factory: sessionmaker[Session],
):
    with unit_of_work(session_factory) as uow:
        uow.idempotency.add(
            IdempotencyRecord(
                client_id="client_a",
                idempotency_key="key",
                input_hash="a" * 64,
                state=IdempotencyState.IN_PROGRESS.value,
                expires_at=NOW + timedelta(hours=24),
            )
        )

    with unit_of_work(session_factory) as uow:
        record = uow.idempotency.find("client_a", "key")
        assert record is not None
        uow.idempotency.mark_completed(record, "resp_ref_123")

    with unit_of_work(session_factory) as uow:
        record = uow.idempotency.find("client_a", "key")
        assert record is not None
        assert record.state == IdempotencyState.COMPLETED.value
        assert record.response_ref == "resp_ref_123"


def test_latest_quota_snapshot_for_provider_wide_window(
    session_factory: sessionmaker[Session],
):
    """A provider-wide observation has no model, so NULL must match NULL."""
    with unit_of_work(session_factory) as uow:
        uow.quotas.record(
            QuotaSnapshot(
                provider="fake",
                model_id=None,
                window_kind=BudgetWindow.HOURLY.value,
                observed_at=NOW,
                remaining=50,
            )
        )
        uow.quotas.record(
            QuotaSnapshot(
                provider="fake",
                model_id="fake/general",
                window_kind=BudgetWindow.HOURLY.value,
                observed_at=NOW,
                remaining=10,
            )
        )

    with unit_of_work(session_factory) as uow:
        provider_wide = uow.quotas.latest("fake", None, BudgetWindow.HOURLY.value)
        assert provider_wide is not None
        assert provider_wide.remaining == 50


def test_active_for_falls_back_to_the_default_policy(
    session_factory: sessionmaker[Session],
):
    from gateway.persistence.models import RoutingPolicy

    with unit_of_work(session_factory) as uow:
        uow.policies.add(
            RoutingPolicy(
                policy_id="default-v1",
                version=1,
                task_class=None,
                active=True,
                quality_floor="standard",
            )
        )

    with unit_of_work(session_factory) as uow:
        fallback = uow.policies.active_for("code_generation")
        assert fallback is not None
        assert fallback.policy_id == "default-v1"
        assert len(uow.policies.list_active()) == 1


# ---------------------------------------------------------------------------
# Transaction behaviour
# ---------------------------------------------------------------------------


def test_immediate_transaction_commits_on_success(session_factory: sessionmaker[Session]):
    session = session_factory()
    try:
        with immediate_transaction(session) as tx:
            tx.add(
                Budget(
                    scope="immediate",
                    window_kind=BudgetWindow.TOTAL.value,
                    window_start=NOW,
                    hard_limit=Decimal("1"),
                )
            )
    finally:
        session.close()

    with unit_of_work(session_factory) as uow:
        assert len(uow.budgets.for_scopes(["immediate"], NOW)) == 1


def test_immediate_transaction_rolls_back_on_error(session_factory: sessionmaker[Session]):
    """A failed reservation must leave no partial state behind (§7)."""
    session = session_factory()
    try:
        with pytest.raises(RuntimeError):
            with immediate_transaction(session) as tx:
                tx.add(
                    Budget(
                        scope="rolled-back",
                        window_kind=BudgetWindow.TOTAL.value,
                        window_start=NOW,
                        hard_limit=Decimal("1"),
                    )
                )
                tx.flush()
                raise RuntimeError("provider blew up")
    finally:
        session.close()

    with unit_of_work(session_factory) as uow:
        assert uow.budgets.for_scopes(["rolled-back"], NOW) == []


def test_unit_of_work_rolls_back_on_error(session_factory: sessionmaker[Session]):
    with pytest.raises(RuntimeError):
        with unit_of_work(session_factory) as uow:
            uow.requests.add(_request("req_rollback"))
            uow.flush()
            raise RuntimeError("boom")

    with unit_of_work(session_factory) as uow:
        assert uow.requests.get("req_rollback") is None
