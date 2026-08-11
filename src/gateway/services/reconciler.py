"""Reservation reconciliation (§6, §11).

*"A reconciler expires orphan reservations and marks stale invocations
abandoned without blindly repeating ambiguous billable calls."*

The dangerous case this exists for: a reservation is ACTIVE and its attempt has
no recorded outcome. That means the gateway crashed, or was killed, somewhere
between reserving and settling -- and critically, **it is unknowable from here
whether the provider was actually called**. The money may or may not have been
spent.

So reconciliation never guesses. An orphan whose attempt never started is
released, because nothing could have been billed. An orphan whose attempt
*did* start is marked EXPIRED and its estimate **settled, not released**: the
conservative direction, because under-counting spend lets the next request
overspend a budget that is really depleted, while over-counting merely denies
a request that might have been affordable. §11 forbids re-invoking such a call
without provider idempotency, and this module never re-invokes anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import AttemptOutcome, RequestState, ReservationStatus
from gateway.persistence.engine import immediate_transaction
from gateway.persistence.models import Attempt, Budget, BudgetReservation, IdempotencyRecord
from gateway.telemetry.metrics import metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """What one reconciliation pass did."""

    released: int = 0
    settled_ambiguous: int = 0
    idempotency_purged: int = 0

    @property
    def total(self) -> int:
        return self.released + self.settled_ambiguous


def _attempt_started(session: Session, attempt_id: str | None) -> tuple[bool, bool]:
    """Return (attempt exists, provider may have been called)."""
    if attempt_id is None:
        return False, False

    attempt = session.get(Attempt, attempt_id)
    if attempt is None:
        return False, False

    # An attempt row with no outcome was in flight when the process stopped.
    # Whether the provider ran is genuinely unknown.
    ambiguous = attempt.outcome is None
    return True, ambiguous


def reconcile_reservations(
    session_factory: sessionmaker[Session],
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> ReconciliationReport:
    """Resolve expired reservations without re-invoking anything."""
    at = now or datetime.now(UTC)
    released = 0
    settled = 0

    session = session_factory()
    try:
        with immediate_transaction(session):
            expired = list(
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

            for reservation in expired:
                _, ambiguous = _attempt_started(session, reservation.attempt_id)
                budget = session.get(Budget, reservation.budget_id)
                if budget is None:  # pragma: no cover - defensive
                    continue

                budget.reserved = max(budget.reserved - reservation.reserved_amount, Decimal("0"))

                if ambiguous:
                    # Conservative: assume it was billed. Under-counting spend
                    # is the more expensive mistake.
                    budget.spent = budget.spent + reservation.reserved_amount
                    reservation.settled_amount = reservation.reserved_amount
                    reservation.status = ReservationStatus.EXPIRED.value
                    settled += 1
                    logger.warning(
                        "Settled an ambiguous reservation without re-invoking",
                        extra={"event": "reservation_ambiguous"},
                    )
                    metrics.reservation_failed(reason="ambiguous_expired")

                    attempt = session.get(Attempt, reservation.attempt_id or "")
                    if attempt is not None:
                        attempt.outcome = AttemptOutcome.CANCELLED.value
                        attempt.error_code = "abandoned"
                else:
                    # Nothing could have been billed, so the hold is returned.
                    reservation.settled_amount = Decimal("0")
                    reservation.status = ReservationStatus.RELEASED.value
                    released += 1
                    metrics.reservation_failed(reason="orphan_released")

                reservation.resolved_at = at
    finally:
        session.close()

    if released or settled:
        logger.info(
            "Reconciled expired reservations",
            extra={"event": "reconciliation_complete"},
        )

    return ReconciliationReport(released=released, settled_ambiguous=settled)


def purge_expired_idempotency(
    session_factory: sessionmaker[Session], *, now: datetime | None = None
) -> int:
    """Drop idempotency records past their retention window (§6: 24 hours)."""
    at = now or datetime.now(UTC)
    session = session_factory()
    try:
        with immediate_transaction(session):
            expired = list(
                session.scalars(select(IdempotencyRecord).where(IdempotencyRecord.expires_at <= at))
            )
            for record in expired:
                session.delete(record)
        return len(expired)
    finally:
        session.close()


def stale_requests(
    session_factory: sessionmaker[Session], *, now: datetime | None = None
) -> list[str]:
    """Requests past their deadline that never reached a terminal state.

    Reported rather than mutated: deciding a request's fate needs its attempt
    state, and §11 requires reconciling that before acting.
    """
    at = now or datetime.now(UTC)
    from gateway.domain.enums import TERMINAL_STATES
    from gateway.persistence.models import Request

    terminal = {state.value for state in TERMINAL_STATES}
    session = session_factory()
    try:
        rows = session.scalars(
            select(Request).where(
                Request.deadline_at <= at,
                Request.state.notin_(terminal),
            )
        )
        return [row.id for row in rows]
    finally:
        session.close()


def expire_stale_requests(
    session_factory: sessionmaker[Session], *, now: datetime | None = None
) -> int:
    """Mark deadline-exceeded requests EXPIRED with a terminal reason (§7)."""
    at = now or datetime.now(UTC)
    from gateway.persistence.models import Request

    ids = stale_requests(session_factory, now=at)
    if not ids:
        return 0

    session = session_factory()
    try:
        with immediate_transaction(session):
            for request_id in ids:
                row = session.get(Request, request_id)
                if row is None:  # pragma: no cover - defensive
                    continue
                row.state = RequestState.EXPIRED.value
                row.terminal_reason = "deadline_exceeded_reconciled"
        return len(ids)
    finally:
        session.close()
