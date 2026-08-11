"""Budget reservation under real concurrency (§6, §14).

"Concurrent tests prove hard budgets cannot be oversubscribed in SQLite or
PostgreSQL." These use **real threads against a real database** rather than
mocked sessions: the invariant lives in the interaction between transaction
isolation, lock ordering, and the read-then-write sequence, and a mock proves
none of it.

Timing sensitivity is inherent. Each test uses a barrier so the threads
genuinely contend rather than running in sequence by luck.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import BudgetWindow, Endpoint, RequestState, ReservationStatus
from gateway.persistence.engine import create_session_factory, immediate_transaction
from gateway.persistence.models import Budget, BudgetReservation, Request
from gateway.services.budget import BudgetError, headroom, release, reserve, settle

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
SCOPE = "client:concurrent"


def make_budget(session: Session, limit: str) -> int:
    budget = Budget(
        scope=SCOPE,
        window_kind=BudgetWindow.TOTAL.value,
        window_start=NOW - timedelta(days=1),
        hard_limit=Decimal(limit),
        spent=Decimal("0"),
        reserved=Decimal("0"),
    )
    session.add(budget)
    session.commit()
    return budget.id


def make_requests(session: Session, request_ids: list[str]) -> None:
    """Create the request rows reservations point at.

    ``budget_reservations.request_id`` is a real foreign key, so a reservation
    cannot exist without its request -- which is the point: an orphaned hold
    could never be reconciled against attempt state.
    """
    for request_id in request_ids:
        session.add(
            Request(
                id=request_id,
                client_id="client_concurrent",
                endpoint=Endpoint.CHAT_COMPLETIONS.value,
                requested_model="auto",
                normalized_controls_json={},
                state=RequestState.RESERVING.value,
                deadline_at=NOW + timedelta(minutes=5),
                input_hash="a" * 64,
                max_cost=Decimal("100"),
            )
        )
    session.commit()


def setup(factory, *, limit: str, request_ids: list[str]) -> None:
    """Create the budget and every request the test will reserve against."""
    with factory() as session:
        make_budget(session, limit)
        make_requests(session, request_ids)


def attempt_reservation(
    factory: sessionmaker[Session],
    *,
    attempt_id: str,
    amount: str,
    barrier: threading.Barrier | None = None,
) -> str:
    """Try one reservation. Returns 'ok', 'denied', or 'contended'.

    ``contended`` covers a database-level lock conflict, which is a legitimate
    outcome under contention -- the reservation did not happen, which is what
    the invariant cares about.
    """
    session = factory()
    try:
        if barrier is not None:
            barrier.wait(timeout=10)
        with immediate_transaction(session):
            reserve(
                session,
                request_id=f"req_{attempt_id}",
                attempt_id=attempt_id,
                scopes=[SCOPE],
                estimate=Decimal(amount),
                request_cost_so_far=Decimal("0"),
                effective_max_cost=None,
                at=NOW,
            )
        return "ok"
    except BudgetError:
        return "denied"
    except OperationalError:
        return "contended"
    finally:
        session.close()


def run_concurrently(factory, count: int, amount: str) -> list[str]:
    """Fire ``count`` reservations that all start at the same instant."""
    barrier = threading.Barrier(count)
    results: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        outcome = attempt_reservation(
            factory, attempt_id=f"att_{index}", amount=amount, barrier=barrier
        )
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    return results


def totals(factory) -> tuple[Decimal, Decimal, Decimal]:
    with factory() as session:
        budget = session.scalars(select(Budget).where(Budget.scope == SCOPE)).one()
        return budget.hard_limit, budget.spent, budget.reserved


def active_count(factory) -> int:
    with factory() as session:
        rows = session.scalars(
            select(BudgetReservation).where(
                BudgetReservation.status == ReservationStatus.ACTIVE.value
            )
        ).all()
        return len(rows)


# ---------------------------------------------------------------------------


def test_two_requests_compete_for_the_last_budget(migrated_engine: Engine):
    """T02 in §12: exactly one reservation succeeds."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=[f"req_att_{i}" for i in range(2)])

    # Each wants 0.6; only one can fit in a limit of 1.0.
    results = run_concurrently(factory, 2, "0.600000000")

    assert results.count("ok") == 1, f"expected exactly one winner, got {results}"

    limit, spent, reserved = totals(factory)
    assert reserved == Decimal("0.600000000")
    assert spent + reserved <= limit


def test_many_concurrent_reservations_never_oversubscribe(migrated_engine: Engine):
    """The core invariant, under heavier contention."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=[f"req_att_{i}" for i in range(10)])

    # Ten threads each wanting 0.15: at most six can fit.
    results = run_concurrently(factory, 10, "0.150000000")
    winners = results.count("ok")

    limit, spent, reserved = totals(factory)

    assert spent + reserved <= limit, (
        f"budget oversubscribed: spent={spent} reserved={reserved} limit={limit}"
    )
    assert winners == active_count(factory)
    assert reserved == Decimal("0.150000000") * winners
    assert winners <= 6


def test_exact_headroom_is_usable(migrated_engine: Engine):
    """A reservation that exactly fills the budget must be allowed."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="0.500000000", request_ids=["req_att_exact"])

    assert attempt_reservation(factory, attempt_id="att_exact", amount="0.500000000") == "ok"
    _, _, reserved = totals(factory)
    assert reserved == Decimal("0.500000000")


def test_one_unit_over_headroom_is_denied(migrated_engine: Engine):
    """Off-by-one at the smallest representable amount."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="0.500000000", request_ids=["req_att_over"])

    assert attempt_reservation(factory, attempt_id="att_over", amount="0.500000001") == "denied"
    _, _, reserved = totals(factory)
    assert reserved == Decimal("0")


def test_settle_moves_reserved_into_spent(migrated_engine: Engine):
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=["req_settle"])

    session = factory()
    with immediate_transaction(session):
        reservation = reserve(
            session,
            request_id="req_settle",
            attempt_id="att_settle",
            scopes=[SCOPE],
            estimate=Decimal("0.100000000"),
            request_cost_so_far=Decimal("0"),
            effective_max_cost=None,
            at=NOW,
        )
    with immediate_transaction(session):
        settle(session, reservation, actual_cost=Decimal("0.037000000"), at=NOW)
    session.close()

    limit, spent, reserved = totals(factory)
    assert reserved == Decimal("0")
    assert spent == Decimal("0.037000000")
    assert headroom_of(factory) == limit - spent


def headroom_of(factory) -> Decimal:
    with factory() as session:
        budget = session.scalars(select(Budget).where(Budget.scope == SCOPE)).one()
        return headroom(budget)


def test_release_returns_the_whole_hold(migrated_engine: Engine):
    """Every terminal path that did not spend must return its reservation."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=["req_rel"])

    session = factory()
    with immediate_transaction(session):
        reservation = reserve(
            session,
            request_id="req_rel",
            attempt_id="att_rel",
            scopes=[SCOPE],
            estimate=Decimal("0.250000000"),
            request_cost_so_far=Decimal("0"),
            effective_max_cost=None,
            at=NOW,
        )
    with immediate_transaction(session):
        release(session, reservation, at=NOW)
    session.close()

    limit, spent, reserved = totals(factory)
    assert reserved == Decimal("0")
    assert spent == Decimal("0")
    assert headroom_of(factory) == limit


def test_reserve_release_cycle_leaks_nothing(migrated_engine: Engine):
    """Repeated reserve/release must return the budget to its exact start."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=[f"req_{i}" for i in range(12)])

    start = headroom_of(factory)
    for index in range(12):
        session = factory()
        with immediate_transaction(session):
            reservation = reserve(
                session,
                request_id=f"req_{index}",
                attempt_id=f"att_cycle_{index}",
                scopes=[SCOPE],
                estimate=Decimal("0.900000000"),
                request_cost_so_far=Decimal("0"),
                effective_max_cost=None,
                at=NOW,
            )
        with immediate_transaction(session):
            release(session, reservation, at=NOW)
        session.close()

    assert headroom_of(factory) == start


def test_double_settle_does_not_double_charge(migrated_engine: Engine):
    """Retrying a settle must be a no-op, not a second charge."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=["req_double"])

    session = factory()
    with immediate_transaction(session):
        reservation = reserve(
            session,
            request_id="req_double",
            attempt_id="att_double",
            scopes=[SCOPE],
            estimate=Decimal("0.100000000"),
            request_cost_so_far=Decimal("0"),
            effective_max_cost=None,
            at=NOW,
        )
    with immediate_transaction(session):
        settle(session, reservation, actual_cost=Decimal("0.050000000"), at=NOW)
    with immediate_transaction(session):
        settle(session, reservation, actual_cost=Decimal("0.050000000"), at=NOW)
    session.close()

    _, spent, reserved = totals(factory)
    assert spent == Decimal("0.050000000")
    assert reserved == Decimal("0")


def test_settling_more_than_reserved_is_recorded_accurately(migrated_engine: Engine):
    """§6: an under-estimate is settled at the true figure, not capped.

    Capping would understate spend and let the next attempt run against a
    budget that is really exhausted.
    """
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="1.000000000", request_ids=["req_under"])

    session = factory()
    with immediate_transaction(session):
        reservation = reserve(
            session,
            request_id="req_under",
            attempt_id="att_under",
            scopes=[SCOPE],
            estimate=Decimal("0.010000000"),
            request_cost_so_far=Decimal("0"),
            effective_max_cost=None,
            at=NOW,
        )
    with immediate_transaction(session):
        settle(session, reservation, actual_cost=Decimal("0.030000000"), at=NOW)
    session.close()

    _, spent, reserved = totals(factory)
    assert spent == Decimal("0.030000000")
    assert reserved == Decimal("0")


def test_request_ceiling_is_checked_before_any_budget_is_touched(migrated_engine: Engine):
    """A request that cannot afford its next attempt never takes a lock."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="100.000000000", request_ids=["req_ceiling"])

    session = factory()
    try:
        with pytest.raises(BudgetError, match="request cost ceiling"):
            with immediate_transaction(session):
                reserve(
                    session,
                    request_id="req_ceiling",
                    attempt_id="att_ceiling",
                    scopes=[SCOPE],
                    estimate=Decimal("0.060000000"),
                    request_cost_so_far=Decimal("0.050000000"),
                    effective_max_cost=Decimal("0.100000000"),
                    at=NOW,
                )
    finally:
        session.close()

    _, _, reserved = totals(factory)
    assert reserved == Decimal("0")


def test_concurrent_settles_keep_the_ledger_exact(migrated_engine: Engine):
    """Parallel settlement of independent reservations must not lose spend."""
    factory = create_session_factory(migrated_engine)
    setup(factory, limit="10.000000000", request_ids=[f"req_s{i}" for i in range(8)])

    reservations = []
    for index in range(8):
        session = factory()
        with immediate_transaction(session):
            reservations.append(
                reserve(
                    session,
                    request_id=f"req_s{index}",
                    attempt_id=f"att_s{index}",
                    scopes=[SCOPE],
                    estimate=Decimal("0.100000000"),
                    request_cost_so_far=Decimal("0"),
                    effective_max_cost=None,
                    at=NOW,
                )
            )
        session.close()

    barrier = threading.Barrier(len(reservations))

    def settle_one(reservation) -> None:
        session = factory()
        try:
            barrier.wait(timeout=10)
            for _ in range(5):
                try:
                    with immediate_transaction(session):
                        settle(session, reservation, actual_cost=Decimal("0.020000000"), at=NOW)
                    return
                except OperationalError:
                    session.rollback()
        finally:
            session.close()

    threads = [threading.Thread(target=settle_one, args=(r,)) for r in reservations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    _, spent, reserved = totals(factory)
    assert reserved == Decimal("0"), "every hold should have been resolved"
    assert spent == Decimal("0.020000000") * len(reservations)
