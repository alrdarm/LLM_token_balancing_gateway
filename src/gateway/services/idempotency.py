"""Idempotency claims and replay (§1, §6).

The contract: *"Same canonical input replays; different input: 409."*

Two distinct protections, often conflated:

* **Replay** — the same key with the same canonical input returns the stored
  outcome without calling a provider again. This is what stops a client retry
  after a dropped connection from being billed twice.
* **Conflict** — the same key with *different* input is a client bug (a reused
  key), and answering it with the first request's response would be worse than
  an error: the caller would silently receive a reply to a question they did
  not ask.

The claim is inserted **before** any provider call and relies on the unique
constraint on ``(client_id, idempotency_key)`` rather than a read-then-write
check. Two concurrent requests with the same key race; exactly one wins the
insert, and the loser is told the request is in progress. A check-then-insert
would let both pass the check and both invoke the provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gateway.domain.enums import IdempotencyState
from gateway.domain.errors import IdempotencyConflictError, RequestInProgressError
from gateway.persistence.models import IdempotencyRecord
from gateway.persistence.repositories import IdempotencyRepository

logger = logging.getLogger(__name__)

#: §6 retention: the stored response reference lives 24 hours. After that the
#: key is reusable, because the outcome it referred to is gone.
DEFAULT_TTL = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class Claim:
    """The result of trying to claim a key."""

    record_id: int
    request_id: str
    #: Set when this call replays an earlier completed request.
    replayed_response_ref: str | None = None

    @property
    def is_replay(self) -> bool:
        return self.replayed_response_ref is not None


def claim(
    session: Session,
    *,
    client_id: str,
    key: str,
    input_hash: str,
    request_id: str,
    now: datetime | None = None,
    ttl: timedelta = DEFAULT_TTL,
) -> Claim:
    """Claim ``key`` for this request, or replay, or raise.

    Raises :class:`IdempotencyConflictError` when the key was used for
    different input, and :class:`RequestInProgressError` when another request
    holds it and has not finished.
    """
    at = now or datetime.now(UTC)
    repository = IdempotencyRepository(session)

    existing = repository.find(client_id, key)
    if existing is not None and existing.expires_at > at:
        return _resolve_existing(existing, input_hash=input_hash, request_id=request_id)

    if existing is not None:
        # Expired: the response it pointed at is gone, so the key is free
        # again. Reuse the row rather than leaving a duplicate behind.
        session.delete(existing)
        session.flush()

    record = IdempotencyRecord(
        client_id=client_id,
        idempotency_key=key,
        input_hash=input_hash,
        request_id=request_id,
        state=IdempotencyState.IN_PROGRESS.value,
        expires_at=at + ttl,
    )
    session.add(record)

    try:
        session.flush()
    except IntegrityError:
        # Lost the race. The unique constraint -- not a prior read -- is what
        # guarantees only one caller proceeds.
        session.rollback()
        winner = repository.find(client_id, key)
        if winner is None:  # pragma: no cover - the row must exist to conflict
            raise
        return _resolve_existing(winner, input_hash=input_hash, request_id=request_id)

    return Claim(record_id=record.id, request_id=request_id)


def _resolve_existing(record: IdempotencyRecord, *, input_hash: str, request_id: str) -> Claim:
    """Decide what an existing claim means for this request."""
    if record.input_hash != input_hash:
        # §1: same key, different body is a 409. Returning the stored response
        # would answer a question the caller never asked.
        logger.warning(
            "Idempotency key reused with different input", extra={"event": "idempotency_conflict"}
        )
        raise IdempotencyConflictError(
            "This Idempotency-Key was already used with a different request body.",
            param="Idempotency-Key",
        )

    if record.state == IdempotencyState.COMPLETED.value and record.response_ref:
        return Claim(
            record_id=record.id,
            request_id=record.request_id or request_id,
            replayed_response_ref=record.response_ref,
        )

    if record.state == IdempotencyState.IN_PROGRESS.value:
        # Another request owns this key right now. Retryable: the caller can
        # try again once the original finishes.
        raise RequestInProgressError(
            "A request with this Idempotency-Key is still in progress.",
            param="Idempotency-Key",
            retry_after=1,
        )

    # A previous attempt failed and stored no response. Let this one proceed
    # under the same key rather than blocking the caller forever.
    return Claim(record_id=record.id, request_id=request_id)


def complete(session: Session, record_id: int, *, response_ref: str) -> None:
    """Mark a claim completed and store the response reference."""
    record = session.get(IdempotencyRecord, record_id)
    if record is None:  # pragma: no cover - defensive
        return
    record.state = IdempotencyState.COMPLETED.value
    record.response_ref = response_ref


def fail(session: Session, record_id: int) -> None:
    """Mark a claim failed, freeing the key for a genuine retry."""
    record = session.get(IdempotencyRecord, record_id)
    if record is None:  # pragma: no cover - defensive
        return
    record.state = IdempotencyState.FAILED.value
