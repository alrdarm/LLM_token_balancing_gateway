"""Atomic budget reservation (§6).

The load-bearing invariant of the whole gateway: **reserve estimated monetary
cost atomically before every billable invocation; settle or release on every
exit path, including failures and cancellation.**

The algorithm, exactly as §6 specifies it::

    estimate = price(input_estimate, expected_or_max_output) + request_fee
    assert request_cost_so_far + estimate <= effective_max_cost
    BEGIN ATOMIC
      lock applicable budgets in stable scope order
      assert each hard_limit - spent - reserved >= estimate
      increment reserved; insert active reservation rows
    COMMIT
    invoke provider (never hold DB lock)
    BEGIN ATOMIC
      active -> settled/released
      decrement reserved; increment spent by actual monetary cost
    COMMIT

Three details do the real work:

* **Stable lock order.** Budgets are locked sorted by scope. Two requests
  touching the same pair of budgets in opposite orders would deadlock.
* **The lock is never held across a provider call.** A provider can take
  thirty seconds; holding a write lock that long would serialise the whole
  gateway, and on SQLite would block every other reservation outright.
* **Settlement is exact, not estimated.** ``reserved`` is decremented by what
  was held and ``spent`` incremented by what was actually billed, so an
  under-estimate cannot silently vanish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.domain.enums import ReservationStatus
from gateway.domain.errors import BudgetExceededError
from gateway.persistence.models import Budget, BudgetReservation
from gateway.persistence.types import to_money

logger = logging.getLogger(__name__)

#: How long a reservation may stay ACTIVE before the reconciler may consider it
#: orphaned. Longer than any plausible provider call plus its retries, because
#: expiring a live reservation would let the same money be spent twice.
DEFAULT_RESERVATION_TTL = timedelta(minutes=15)


class BudgetError(BudgetExceededError):
    """Raised when a reservation cannot be made."""


@dataclass(frozen=True, slots=True)
class Reservation:
    """A successful hold across every applicable budget."""

    request_id: str
    attempt_id: str
    amount: Decimal
    reservation_ids: tuple[int, ...]
    budget_ids: tuple[int, ...]
    expires_at: datetime


def _lock_budgets(session: Session, scopes: list[str], at: datetime) -> list[Budget]:
    """Select applicable budgets in a stable order, locking them for update.

    PostgreSQL takes row locks via ``FOR UPDATE``. SQLite has no row locking,
    so the caller must already be inside a ``BEGIN IMMEDIATE`` transaction --
    the whole-database write lock is what makes the read-then-write sequence
    atomic there.
    """
    statement = (
        select(Budget)
        .where(Budget.scope.in_(scopes), Budget.window_start <= at)
        .where((Budget.window_end.is_(None)) | (Budget.window_end > at))
        # Stable order prevents deadlock between concurrent reservations.
        .order_by(Budget.scope, Budget.window_start, Budget.id)
    )

    if session.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update()

    return list(session.scalars(statement))


def headroom(budget: Budget) -> Decimal:
    """Spendable amount remaining on ``budget``."""
    return budget.hard_limit - budget.spent - budget.reserved


def reserve(
    session: Session,
    *,
    request_id: str,
    attempt_id: str,
    scopes: list[str],
    estimate: Decimal,
    request_cost_so_far: Decimal,
    effective_max_cost: Decimal | None,
    at: datetime | None = None,
    ttl: timedelta = DEFAULT_RESERVATION_TTL,
) -> Reservation:
    """Hold ``estimate`` against every applicable budget, or raise.

    Must be called inside a transaction the caller commits: the reservation and
    the ``reserved`` increments have to land together or not at all. On SQLite
    that transaction must have been opened with ``BEGIN IMMEDIATE``.
    """
    now = at or datetime.now(UTC)
    amount = to_money(estimate)

    # The caller's own ceiling is checked before any budget is touched, so a
    # request that cannot afford its next attempt never takes a lock.
    if effective_max_cost is not None:
        projected = to_money(request_cost_so_far) + amount
        if projected > effective_max_cost:
            raise BudgetError(
                "This attempt would exceed the request cost ceiling.",
                param="gateway.max_cost",
            )

    budgets = _lock_budgets(session, sorted(set(scopes)), now)

    for budget in budgets:
        if headroom(budget) < amount:
            logger.warning(
                "Budget reservation denied",
                extra={"event": "budget_denied"},
            )
            raise BudgetError("Insufficient budget headroom for this attempt.")

    expires_at = now + ttl
    reservation_ids: list[int] = []

    for budget in budgets:
        budget.reserved = budget.reserved + amount
        row = BudgetReservation(
            request_id=request_id,
            attempt_id=attempt_id,
            budget_id=budget.id,
            reserved_amount=amount,
            status=ReservationStatus.ACTIVE.value,
            expires_at=expires_at,
        )
        session.add(row)
        session.flush()
        reservation_ids.append(row.id)

    logger.info("Budget reserved", extra={"event": "budget_reserved"})

    return Reservation(
        request_id=request_id,
        attempt_id=attempt_id,
        amount=amount,
        reservation_ids=tuple(reservation_ids),
        budget_ids=tuple(budget.id for budget in budgets),
        expires_at=expires_at,
    )


def _resolve(
    session: Session,
    reservation: Reservation,
    *,
    status: ReservationStatus,
    actual_cost: Decimal,
    at: datetime | None = None,
) -> None:
    """Move a reservation to a terminal status and adjust its budgets.

    Settlement is a read-modify-write on ``reserved`` and ``spent``, so it must
    take the same locks reservation does. Without them PostgreSQL's READ
    COMMITTED lets two concurrent settlements read the same balance and both
    write back, losing one of the updates -- real money silently vanishing from
    the ledger. SQLite's whole-database write lock hides this, which is exactly
    why the concurrency suite runs on both backends.

    Lock order is fixed -- reservations by ID, then budgets by ID -- so
    concurrent settlements cannot deadlock against each other or against
    :func:`reserve`.
    """
    now = at or datetime.now(UTC)
    settled = to_money(actual_cost)
    is_postgres = session.get_bind().dialect.name == "postgresql"

    reservation_query = (
        select(BudgetReservation)
        .where(BudgetReservation.id.in_(reservation.reservation_ids))
        .order_by(BudgetReservation.id)
    )
    if is_postgres:
        reservation_query = reservation_query.with_for_update()
    rows = list(session.scalars(reservation_query))

    budget_ids = sorted({row.budget_id for row in rows})
    budget_query = select(Budget).where(Budget.id.in_(budget_ids)).order_by(Budget.id)
    if is_postgres:
        budget_query = budget_query.with_for_update()
    budgets = {budget.id: budget for budget in session.scalars(budget_query)}

    for row in rows:
        if row.status != ReservationStatus.ACTIVE.value:
            # Already settled, released, or expired. Re-applying the deltas
            # would corrupt the ledger, so this is deliberately a no-op.
            logger.warning(
                "Reservation already resolved",
                extra={"event": "reservation_double_resolve"},
            )
            continue

        budget = budgets.get(row.budget_id)
        if budget is None:  # pragma: no cover - defensive
            continue

        # Release exactly what was held, then charge exactly what was billed.
        # Doing both keeps the ledger correct even when the estimate was wrong.
        budget.reserved = budget.reserved - row.reserved_amount
        if budget.reserved < 0:
            budget.reserved = Decimal("0")
        budget.spent = budget.spent + settled

        row.status = status.value
        row.settled_amount = settled
        row.resolved_at = now


def settle(
    session: Session,
    reservation: Reservation,
    *,
    actual_cost: Decimal,
    at: datetime | None = None,
) -> None:
    """Convert a hold into actual spend (§6).

    ``actual_cost`` may exceed the estimate; §6 requires the true figure to be
    settled accurately and a high-severity metric emitted, because silently
    capping it would understate spend and let the next attempt proceed on a
    budget that is really exhausted.
    """
    settled = to_money(actual_cost)
    if settled > reservation.amount:
        logger.error(
            "Actual cost exceeded the reservation estimate",
            extra={"event": "cost_underestimated"},
        )

    _resolve(session, reservation, status=ReservationStatus.SETTLED, actual_cost=settled, at=at)
    logger.info("Budget settled", extra={"event": "budget_settled"})


def release(
    session: Session,
    reservation: Reservation,
    *,
    at: datetime | None = None,
) -> None:
    """Return an unused hold (§7).

    Every terminal path that did not spend must land here -- including
    cancellation and provider failure before any billable work.
    """
    _resolve(
        session, reservation, status=ReservationStatus.RELEASED, actual_cost=Decimal("0"), at=at
    )
    logger.info("Budget released", extra={"event": "budget_released"})


def settle_partial(
    session: Session,
    reservation: Reservation,
    *,
    actual_cost: Decimal,
    at: datetime | None = None,
) -> None:
    """Settle a partially billable attempt, e.g. a stream cut mid-flight (§8).

    Distinct from :func:`release` because usage did occur: the provider will
    bill for the tokens it emitted, so releasing the whole hold would lose real
    spend.
    """
    settle(session, reservation, actual_cost=actual_cost, at=at)


def expire_orphans(
    session: Session,
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> list[BudgetReservation]:
    """Find active reservations past their expiry (§11).

    Returns them **without** resolving anything. §11 requires reconciling
    attempt state first and never blindly re-invoking an ambiguous billable
    call, so deciding each one's fate belongs to the M7 reconciler, which can
    see whether the attempt actually completed.
    """
    at = now or datetime.now(UTC)
    return list(
        session.scalars(
            select(BudgetReservation)
            .where(
                BudgetReservation.status == ReservationStatus.ACTIVE.value,
                BudgetReservation.expires_at <= at,
            )
            .order_by(BudgetReservation.expires_at)
            .limit(limit)
        )
    )
