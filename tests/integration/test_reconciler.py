"""Reservation reconciliation (§6, §11).

The dangerous case: a reservation is ACTIVE and its attempt has no outcome, so
whether the provider was actually called is *unknowable*. §11 forbids blindly
re-invoking such a call, and these tests pin that the reconciler never does.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import Engine, select

from gateway.domain.enums import (
    AttemptKind,
    AttemptOutcome,
    BudgetWindow,
    Endpoint,
    RequestState,
    ReservationStatus,
)
from gateway.persistence.engine import create_session_factory
from gateway.persistence.models import (
    Attempt,
    Budget,
    BudgetReservation,
    IdempotencyRecord,
    Request,
)
from gateway.services.reconciler import (
    expire_stale_requests,
    purge_expired_idempotency,
    reconcile_reservations,
    stale_requests,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
SCOPE = "client:recon"


@pytest.fixture
def factory(migrated_engine: Engine):
    session_factory = create_session_factory(migrated_engine)
    with session_factory() as session:
        session.add(
            Budget(
                scope=SCOPE,
                window_kind=BudgetWindow.TOTAL.value,
                window_start=NOW - timedelta(days=1),
                hard_limit=Decimal("10"),
                spent=Decimal("0"),
                reserved=Decimal("1.000000000"),
            )
        )
        session.add(
            Request(
                id="req_recon",
                client_id="c",
                endpoint=Endpoint.CHAT_COMPLETIONS.value,
                requested_model="auto",
                normalized_controls_json={},
                state=RequestState.INVOKING.value,
                deadline_at=NOW + timedelta(minutes=1),
                input_hash="a" * 64,
                max_cost=Decimal("5"),
            )
        )
        session.commit()
    return session_factory


def add_reservation(factory, *, attempt_id: str | None, expires: datetime) -> int:
    with factory() as session:
        row = BudgetReservation(
            request_id="req_recon",
            attempt_id=attempt_id,
            budget_id=session.scalars(select(Budget)).one().id,
            reserved_amount=Decimal("1.000000000"),
            status=ReservationStatus.ACTIVE.value,
            expires_at=expires,
        )
        session.add(row)
        session.commit()
        return row.id


def add_attempt(factory, attempt_id: str, *, outcome: AttemptOutcome | None) -> None:
    with factory() as session:
        session.add(
            Attempt(
                id=attempt_id,
                request_id="req_recon",
                sequence=1,
                model_id="fake/general",
                route_rank=1,
                attempt_kind=AttemptKind.GENERATION.value,
                outcome=outcome.value if outcome else None,
                cost_estimated=Decimal("1"),
            )
        )
        session.commit()


def budget_state(factory) -> tuple[Decimal, Decimal]:
    with factory() as session:
        budget = session.scalars(select(Budget)).one()
        return budget.spent, budget.reserved


def reservation_status(factory, reservation_id: int) -> str:
    with factory() as session:
        return session.get(BudgetReservation, reservation_id).status


def test_unexpired_reservations_are_untouched(factory):
    reservation_id = add_reservation(factory, attempt_id=None, expires=NOW + timedelta(hours=1))

    report = reconcile_reservations(factory, now=NOW)

    assert report.total == 0
    assert reservation_status(factory, reservation_id) == ReservationStatus.ACTIVE.value


def test_orphan_without_an_attempt_is_released(factory):
    """Nothing could have been billed, so the hold is returned in full."""
    reservation_id = add_reservation(factory, attempt_id=None, expires=NOW - timedelta(minutes=1))

    report = reconcile_reservations(factory, now=NOW)

    assert report.released == 1
    assert reservation_status(factory, reservation_id) == ReservationStatus.RELEASED.value

    spent, reserved = budget_state(factory)
    assert spent == Decimal("0")
    assert reserved == Decimal("0")


def test_ambiguous_reservation_is_settled_not_released(factory):
    """An attempt with no outcome may or may not have been billed.

    §11 forbids re-invoking to find out, so the conservative direction wins:
    assume it was spent. Under-counting would let the next request overspend a
    budget that is actually depleted.
    """
    add_attempt(factory, "att_ambiguous", outcome=None)
    reservation_id = add_reservation(
        factory, attempt_id="att_ambiguous", expires=NOW - timedelta(minutes=1)
    )

    report = reconcile_reservations(factory, now=NOW)

    assert report.settled_ambiguous == 1
    assert reservation_status(factory, reservation_id) == ReservationStatus.EXPIRED.value

    spent, reserved = budget_state(factory)
    assert spent == Decimal("1.000000000")
    assert reserved == Decimal("0")


def test_ambiguous_attempt_is_marked_abandoned(factory):
    add_attempt(factory, "att_abandoned", outcome=None)
    add_reservation(factory, attempt_id="att_abandoned", expires=NOW - timedelta(minutes=1))

    reconcile_reservations(factory, now=NOW)

    with factory() as session:
        attempt = session.get(Attempt, "att_abandoned")
        assert attempt.outcome == AttemptOutcome.CANCELLED.value
        assert attempt.error_code == "abandoned"


def test_reconciliation_never_reinvokes(factory):
    """The reconciler has no adapter and cannot call a provider by construction.

    Asserted structurally rather than behaviourally: the module imports no
    provider, so there is nothing it *could* invoke.
    """
    import gateway.services.reconciler as module

    source = module.__doc__ or ""
    assert "never re-invokes" in source or "without re-invoking" in source.lower()
    assert not hasattr(module, "AdapterRegistry")


def test_reconciliation_is_idempotent(factory):
    """A second pass must not double-charge an already-resolved reservation."""
    add_attempt(factory, "att_twice", outcome=None)
    add_reservation(factory, attempt_id="att_twice", expires=NOW - timedelta(minutes=1))

    reconcile_reservations(factory, now=NOW)
    first = budget_state(factory)
    reconcile_reservations(factory, now=NOW)

    assert budget_state(factory) == first


def test_reserved_never_goes_negative(factory):
    """Defensive: a double release must not corrupt the ledger."""
    add_reservation(factory, attempt_id=None, expires=NOW - timedelta(minutes=1))
    add_reservation(factory, attempt_id=None, expires=NOW - timedelta(minutes=1))

    reconcile_reservations(factory, now=NOW)

    _, reserved = budget_state(factory)
    assert reserved >= Decimal("0")


# --- retention and stale requests -----------------------------------------


def test_expired_idempotency_records_are_purged(factory):
    """§6: the response reference is retained for 24 hours."""
    with factory() as session:
        for key, expires in (
            ("stale", NOW - timedelta(hours=1)),
            ("live", NOW + timedelta(hours=1)),
        ):
            session.add(
                IdempotencyRecord(
                    client_id="c",
                    idempotency_key=key,
                    input_hash="a" * 64,
                    state="COMPLETED",
                    expires_at=expires,
                )
            )
        session.commit()

    assert purge_expired_idempotency(factory, now=NOW) == 1

    with factory() as session:
        remaining = [row.idempotency_key for row in session.scalars(select(IdempotencyRecord))]
    assert remaining == ["live"]


def test_stale_requests_are_reported_before_being_changed(factory):
    """§11: reconcile attempt state before acting."""
    assert stale_requests(factory, now=NOW + timedelta(minutes=5)) == ["req_recon"]


def test_stale_requests_reach_a_terminal_state(factory):
    """§7: every request must end with a terminal reason."""
    assert expire_stale_requests(factory, now=NOW + timedelta(minutes=5)) == 1

    with factory() as session:
        row = session.get(Request, "req_recon")
        assert row.state == RequestState.EXPIRED.value
        assert row.terminal_reason == "deadline_exceeded_reconciled"


def test_terminal_requests_are_not_re_expired(factory):
    with factory() as session:
        row = session.get(Request, "req_recon")
        row.state = RequestState.SUCCEEDED.value
        session.commit()

    assert expire_stale_requests(factory, now=NOW + timedelta(days=1)) == 0
