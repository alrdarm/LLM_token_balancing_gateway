"""Repositories.

Each repository owns queries for one aggregate. They deliberately expose only
what M1 needs -- persistence and retrieval -- and no policy. Budget reservation
in particular is *not* here: its atomicity rules (§6) belong to the
``BudgetManager`` in M4, and a convenience "reserve" helper on this layer would
invite bypassing them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.domain.enums import IdempotencyState, RequestState, ReservationStatus
from gateway.persistence.models import (
    Attempt,
    Budget,
    BudgetReservation,
    IdempotencyRecord,
    Model,
    ModelPrice,
    QuotaSnapshot,
    Request,
    RoutingPolicy,
    Validation,
)


class ModelRepository:
    """Model registry reads and writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, model: Model) -> Model:
        self.session.add(model)
        return model

    def get(self, model_id: str) -> Model | None:
        return self.session.get(Model, model_id)

    def list_enabled(self) -> list[Model]:
        """Enabled models, ordered by ID so snapshots are deterministic (§4)."""
        stmt = select(Model).where(Model.enabled.is_(True)).order_by(Model.id)
        return list(self.session.scalars(stmt))

    def price_at(self, model_id: str, when: datetime) -> ModelPrice | None:
        """The price effective for ``model_id`` at ``when``.

        Effective-dated rather than current-valued, so the cost of a past
        request stays reconstructible after a price change.
        """
        stmt = (
            select(ModelPrice)
            .where(
                ModelPrice.model_id == model_id,
                ModelPrice.effective_from <= when,
            )
            .where((ModelPrice.effective_to.is_(None)) | (ModelPrice.effective_to > when))
            .order_by(ModelPrice.effective_from.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).first()


class PolicyRepository:
    """Routing policy reads and writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, policy: RoutingPolicy) -> RoutingPolicy:
        self.session.add(policy)
        return policy

    def get_version(self, policy_id: str, version: int) -> RoutingPolicy | None:
        stmt = select(RoutingPolicy).where(
            RoutingPolicy.policy_id == policy_id, RoutingPolicy.version == version
        )
        return self.session.scalars(stmt).first()

    def active_for(self, task_class: str | None) -> RoutingPolicy | None:
        """The active policy for ``task_class``, falling back to the default.

        A class-specific policy wins over the catch-all so a high-risk class can
        tighten its gates without redefining every default.
        """
        stmt = (
            select(RoutingPolicy)
            .where(RoutingPolicy.active.is_(True), RoutingPolicy.task_class == task_class)
            .order_by(RoutingPolicy.version.desc())
            .limit(1)
        )
        specific = self.session.scalars(stmt).first()
        if specific is not None or task_class is None:
            return specific

        fallback = (
            select(RoutingPolicy)
            .where(RoutingPolicy.active.is_(True), RoutingPolicy.task_class.is_(None))
            .order_by(RoutingPolicy.version.desc())
            .limit(1)
        )
        return self.session.scalars(fallback).first()

    def list_active(self) -> list[RoutingPolicy]:
        stmt = (
            select(RoutingPolicy)
            .where(RoutingPolicy.active.is_(True))
            .order_by(RoutingPolicy.policy_id, RoutingPolicy.version)
        )
        return list(self.session.scalars(stmt))


class RequestRepository:
    """Request lifecycle reads and writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, request: Request) -> Request:
        self.session.add(request)
        return request

    def get(self, request_id: str) -> Request | None:
        return self.session.get(Request, request_id)

    def next_attempt_sequence(self, request_id: str) -> int:
        """The next free sequence number for a request's attempts.

        Advisory only. The unique constraint on ``(request_id, sequence)`` is
        the actual guarantee, because two concurrent callers can read the same
        value here.
        """
        stmt = (
            select(Attempt.sequence)
            .where(Attempt.request_id == request_id)
            .order_by(Attempt.sequence.desc())
            .limit(1)
        )
        highest = self.session.scalars(stmt).first()
        return 1 if highest is None else highest + 1

    def add_attempt(self, attempt: Attempt) -> Attempt:
        self.session.add(attempt)
        return attempt

    def add_validation(self, validation: Validation) -> Validation:
        self.session.add(validation)
        return validation

    def attempts_for(self, request_id: str) -> list[Attempt]:
        stmt = select(Attempt).where(Attempt.request_id == request_id).order_by(Attempt.sequence)
        return list(self.session.scalars(stmt))

    def validations_for(self, request_id: str) -> list[Validation]:
        stmt = (
            select(Validation)
            .where(Validation.request_id == request_id)
            .order_by(Validation.attempt_id, Validation.sequence)
        )
        return list(self.session.scalars(stmt))

    def list_in_state(self, state: RequestState, limit: int = 100) -> list[Request]:
        stmt = (
            select(Request)
            .where(Request.state == state.value)
            .order_by(Request.created_at)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class BudgetRepository:
    """Budget and reservation reads.

    Mutation of ``reserved`` and ``spent`` is intentionally absent: it must go
    through the atomic algorithm in §6, which M4 implements.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, budget: Budget) -> Budget:
        self.session.add(budget)
        return budget

    def get(self, budget_id: int) -> Budget | None:
        return self.session.get(Budget, budget_id)

    def for_scopes(self, scopes: list[str], at: datetime) -> list[Budget]:
        """Budgets applying to any of ``scopes`` at ``at``.

        Ordered by scope so callers lock in a stable order, which is what stops
        two concurrent reservations from deadlocking (§6).
        """
        stmt = (
            select(Budget)
            .where(Budget.scope.in_(scopes), Budget.window_start <= at)
            .where((Budget.window_end.is_(None)) | (Budget.window_end > at))
            .order_by(Budget.scope, Budget.window_start)
        )
        return list(self.session.scalars(stmt))

    def headroom(self, budget: Budget) -> Decimal:
        """Remaining spendable amount: ``hard_limit - spent - reserved``."""
        return budget.hard_limit - budget.spent - budget.reserved

    def add_reservation(self, reservation: BudgetReservation) -> BudgetReservation:
        self.session.add(reservation)
        return reservation

    def active_reservations(self, request_id: str) -> list[BudgetReservation]:
        stmt = select(BudgetReservation).where(
            BudgetReservation.request_id == request_id,
            BudgetReservation.status == ReservationStatus.ACTIVE.value,
        )
        return list(self.session.scalars(stmt))

    def expired_reservations(self, now: datetime, limit: int = 100) -> list[BudgetReservation]:
        """Active reservations past their expiry, for the M7 reconciler.

        Returned, not resolved: §11 requires reconciling attempt state before
        expiring a reservation, and never blindly re-invoking an ambiguous
        billable call.
        """
        stmt = (
            select(BudgetReservation)
            .where(
                BudgetReservation.status == ReservationStatus.ACTIVE.value,
                BudgetReservation.expires_at <= now,
            )
            .order_by(BudgetReservation.expires_at)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class IdempotencyRepository:
    """Idempotency record reads and writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, record: IdempotencyRecord) -> IdempotencyRecord:
        self.session.add(record)
        return record

    def find(self, client_id: str, key: str) -> IdempotencyRecord | None:
        stmt = select(IdempotencyRecord).where(
            IdempotencyRecord.client_id == client_id,
            IdempotencyRecord.idempotency_key == key,
        )
        return self.session.scalars(stmt).first()

    def find_live(self, client_id: str, key: str, now: datetime) -> IdempotencyRecord | None:
        """A record that has not yet expired.

        An expired key is reusable; treating it as live would reject a legitimate
        retry long after its response reference was dropped (§6 retention).
        """
        record = self.find(client_id, key)
        if record is None or record.expires_at <= now:
            return None
        return record

    def purge_expired(self, now: datetime) -> int:
        """Delete expired records, returning how many were removed."""
        stmt = select(IdempotencyRecord).where(IdempotencyRecord.expires_at <= now)
        expired = list(self.session.scalars(stmt))
        for record in expired:
            self.session.delete(record)
        return len(expired)

    def mark_completed(self, record: IdempotencyRecord, response_ref: str) -> None:
        record.state = IdempotencyState.COMPLETED.value
        record.response_ref = response_ref


class QuotaRepository:
    """Append-only quota snapshot reads and writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, snapshot: QuotaSnapshot) -> QuotaSnapshot:
        """Append an observation. Snapshots are never updated in place (§6)."""
        self.session.add(snapshot)
        return snapshot

    def latest(self, provider: str, model_id: str | None, window_kind: str) -> QuotaSnapshot | None:
        stmt = (
            select(QuotaSnapshot)
            .where(
                QuotaSnapshot.provider == provider,
                QuotaSnapshot.model_id.is_(None)
                if model_id is None
                else QuotaSnapshot.model_id == model_id,
                QuotaSnapshot.window_kind == window_kind,
            )
            .order_by(QuotaSnapshot.observed_at.desc(), QuotaSnapshot.id.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).first()
