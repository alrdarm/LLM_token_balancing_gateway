"""ORM entities (§6).

Two rules shape every table here:

* **No raw content.** Prompts, outputs, tool arguments, and provider payloads
  are never columns. Where content must be identified, a keyed digest is
  stored instead (see :mod:`gateway.telemetry.hashing`).
* **Money is exact.** Every monetary column uses :class:`~gateway.persistence.
  types.Money`, never ``Float``.

Enum-valued columns are stored as ``String`` rather than a native database
enum, so adding a member is a code change on both backends rather than a
PostgreSQL-only ``ALTER TYPE``. Value sets are enforced in the domain layer and
by check constraints where the set is closed and small.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from gateway.domain.enums import (
    AttemptKind,
    AttemptOutcome,
    BudgetWindow,
    DataHandlingTier,
    Endpoint,
    IdempotencyState,
    Quality,
    RequestState,
    ReservationStatus,
    ValidationResult,
)
from gateway.persistence.types import JSONDocument, Money, UTCDateTime

#: Explicit naming convention so Alembic emits stable, comparable names on both
#: backends. Without it, SQLite constraints are largely unnamed and later
#: migrations cannot reference them.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


def _enum_values(enum_cls: type[StrEnum]) -> str:
    """Render a SQL ``IN`` list for a check constraint.

    Values are drawn from the domain enum so the database constraint and the
    application vocabulary cannot drift apart.
    """
    members = ", ".join(f"'{member.value}'" for member in enum_cls)
    return f"({members})"


class Base(DeclarativeBase):
    """Declarative base carrying the shared metadata."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Creation and update timestamps, defaulted by the database."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------
# Registry: models, prices, policies
# --------------------------------------------------------------------------


class Model(TimestampMixin, Base):
    """A routable gateway model (§6 ``models``)."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider_model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)

    data_handling_tier: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    quality_tier: Mapped[str] = mapped_column(String(32), nullable=False)

    capabilities: Mapped[list[str]] = mapped_column(JSONDocument, nullable=False, default=list)
    supported_endpoints: Mapped[list[str]] = mapped_column(
        JSONDocument, nullable=False, default=list
    )

    context_window_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    #: Conservative v0.1 routing priors (§10). These are *priors*, not
    #: measurements: ``priors_source`` records where they came from so later
    #: empirical values can replace them as data, without changing the scorer.
    latency_prior_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=2000)
    pass_rate_prior: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    failure_rate_prior: Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    priors_source: Mapped[str] = mapped_column(
        String(200), nullable=False, default="v0.1-conservative-prior"
    )

    prices: Mapped[list[ModelPrice]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            f"data_handling_tier IN {_enum_values(DataHandlingTier)}",
            name="data_handling_tier_known",
        ),
        CheckConstraint(f"quality_tier IN {_enum_values(Quality)}", name="quality_tier_known"),
        CheckConstraint("context_window_tokens > 0", name="context_window_positive"),
        CheckConstraint("max_output_tokens > 0", name="max_output_positive"),
        CheckConstraint("latency_prior_ms > 0", name="latency_prior_positive"),
        CheckConstraint(
            "pass_rate_prior >= 0 AND pass_rate_prior <= 1", name="pass_rate_prior_is_probability"
        ),
        CheckConstraint(
            "failure_rate_prior >= 0 AND failure_rate_prior <= 1",
            name="failure_rate_prior_is_probability",
        ),
    )


class ModelPrice(TimestampMixin, Base):
    """Effective-dated pricing (§6 ``model_prices``).

    Prices are temporal rather than mutable in place: a request's cost must be
    reconstructible from the price that applied when it ran, and the registry
    snapshot it froze.
    """

    __tablename__ = "model_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")

    input_per_1k_tokens: Mapped[Decimal] = mapped_column(Money, nullable=False)
    output_per_1k_tokens: Mapped[Decimal] = mapped_column(Money, nullable=False)
    request_fee: Mapped[Decimal] = mapped_column(Money, nullable=False, default=Decimal("0"))

    effective_from: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    effective_to: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    source: Mapped[str] = mapped_column(String(200), nullable=False, default="seed")

    model: Mapped[Model] = relationship(back_populates="prices")

    __table_args__ = (
        UniqueConstraint("model_id", "effective_from", name="uq_model_prices_model_from"),
        CheckConstraint("input_per_1k_tokens >= 0", name="input_price_non_negative"),
        CheckConstraint("output_per_1k_tokens >= 0", name="output_price_non_negative"),
        CheckConstraint("request_fee >= 0", name="request_fee_non_negative"),
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="price_window_ordered",
        ),
    )


class RoutingPolicy(TimestampMixin, Base):
    """Versioned routing policy (§6 ``routing_policies``).

    Policies are immutable once published: a request freezes ``policy_id`` and
    ``policy_version``, so editing a live row would retroactively rewrite how a
    completed request was decided.
    """

    __tablename__ = "routing_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    task_class: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    quality_floor: Mapped[str] = mapped_column(String(32), nullable=False)
    max_generation_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    max_same_model_repairs: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    #: Scoring weights (§10): cost, latency, quality shortfall, failure risk.
    weights: Mapped[dict[str, Any]] = mapped_column(JSONDocument, nullable=False, default=dict)
    #: Ordered validator names required for this class (§10).
    validation_plan: Mapped[list[str]] = mapped_column(JSONDocument, nullable=False, default=list)
    #: Provenance for the pass-rate priors the weights encode.
    priors_source: Mapped[str] = mapped_column(String(200), nullable=False, default="v0.1-prior")

    __table_args__ = (
        UniqueConstraint("policy_id", "version", name="uq_routing_policies_policy_version"),
        CheckConstraint(f"quality_floor IN {_enum_values(Quality)}", name="quality_floor_known"),
        CheckConstraint(
            "max_generation_attempts BETWEEN 1 AND 5", name="generation_attempts_in_range"
        ),
        CheckConstraint("max_same_model_repairs >= 0", name="same_model_repairs_non_negative"),
    )


# --------------------------------------------------------------------------
# Budgets and quotas
# --------------------------------------------------------------------------


class Budget(TimestampMixin, Base):
    """A spend ceiling for a scope and window (§6 ``budgets``).

    ``reserved`` and ``spent`` are separate: reservations are held against
    in-flight invocations and released or settled on every exit path, so
    headroom is ``hard_limit - spent - reserved``.
    """

    __tablename__ = "budgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(200), nullable=False)
    #: Named ``window_kind`` rather than ``window``: WINDOW is a reserved
    #: keyword in PostgreSQL and would need quoting in every raw SQL fragment.
    window_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    window_start: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    window_end: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    hard_limit: Mapped[Decimal] = mapped_column(Money, nullable=False)
    spent: Mapped[Decimal] = mapped_column(Money, nullable=False, default=Decimal("0"))
    reserved: Mapped[Decimal] = mapped_column(Money, nullable=False, default=Decimal("0"))

    __table_args__ = (
        UniqueConstraint("scope", "window_kind", "window_start", name="uq_budgets_scope_window"),
        CheckConstraint(f"window_kind IN {_enum_values(BudgetWindow)}", name="window_kind_known"),
        CheckConstraint("hard_limit >= 0", name="hard_limit_non_negative"),
        CheckConstraint("spent >= 0", name="spent_non_negative"),
        CheckConstraint("reserved >= 0", name="reserved_non_negative"),
    )


class QuotaSnapshot(Base):
    """Append-only provider quota observation (§6 ``quota_snapshots``).

    Never updated in place: staleness is itself a routing signal, so an old
    observation must remain visible rather than be overwritten.
    """

    __tablename__ = "quota_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    window_kind: Mapped[str] = mapped_column(String(32), nullable=False)

    observed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    limit_value: Mapped[int | None] = mapped_column(Integer, nullable=True)
    remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resets_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_quota_snapshots_latest", "provider", "model_id", "window_kind", "observed_at"),
        CheckConstraint("remaining IS NULL OR remaining >= 0", name="remaining_non_negative"),
    )


# --------------------------------------------------------------------------
# Request lifecycle
# --------------------------------------------------------------------------


class Request(TimestampMixin, Base):
    """One client request through its whole lifecycle (§6 ``requests``)."""

    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    client_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_model: Mapped[str] = mapped_column(String(128), nullable=False)

    #: Effective controls after precedence resolution (§2). Operational values
    #: only -- never prompt text or metadata that could carry content.
    normalized_controls_json: Mapped[dict[str, Any]] = mapped_column(
        JSONDocument, nullable=False, default=dict
    )

    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    terminal_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    deadline_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    #: Frozen at planning time so a route stays reconstructible (§10).
    policy_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    registry_snapshot_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    #: Keyed digest of the canonical input. Not reversible to prompt text.
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    estimated_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    max_cost: Mapped[Decimal] = mapped_column(Money, nullable=False)
    cost_actual: Mapped[Decimal] = mapped_column(Money, nullable=False, default=Decimal("0"))

    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)

    attempts: Mapped[list[Attempt]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        order_by="Attempt.sequence",
    )
    validations: Mapped[list[Validation]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(f"state IN {_enum_values(RequestState)}", name="state_known"),
        CheckConstraint(f"endpoint IN {_enum_values(Endpoint)}", name="endpoint_known"),
        CheckConstraint("max_cost >= 0", name="max_cost_non_negative"),
        CheckConstraint("cost_actual >= 0", name="cost_actual_non_negative"),
        CheckConstraint("estimated_input_tokens >= 0", name="input_tokens_non_negative"),
        Index("ix_requests_state_created", "state", "created_at"),
    )


class Attempt(TimestampMixin, Base):
    """One provider invocation within a request (§6 ``attempts``)."""

    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_id: Mapped[str] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Position within the request. Unique per request (§6).
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    model_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    route_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_kind: Mapped[str] = mapped_column(String(32), nullable=False)

    retry_of_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("attempts.id", ondelete="SET NULL"), nullable=True
    )
    reservation_id: Mapped[int | None] = mapped_column(
        ForeignKey("budget_reservations.id", ondelete="SET NULL"), nullable=True
    )

    #: Whether any of this attempt's output became client-visible. Once true a
    #: stream can no longer switch models (§4) and failure is FAILED_PARTIAL.
    emitted_output: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: Redacted classification only -- never the provider's error payload.
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    cost_estimated: Mapped[Decimal] = mapped_column(Money, nullable=False, default=Decimal("0"))
    cost_actual: Mapped[Decimal | None] = mapped_column(Money, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Keyed digest of the output, for replay detection and telemetry only.
    output_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    request: Mapped[Request] = relationship(back_populates="attempts")
    validations: Mapped[list[Validation]] = relationship(
        back_populates="attempt",
        cascade="all, delete-orphan",
        foreign_keys="Validation.attempt_id",
    )

    __table_args__ = (
        UniqueConstraint("request_id", "sequence", name="uq_attempts_request_sequence"),
        CheckConstraint(f"attempt_kind IN {_enum_values(AttemptKind)}", name="attempt_kind_known"),
        CheckConstraint(
            f"outcome IS NULL OR outcome IN {_enum_values(AttemptOutcome)}",
            name="outcome_known",
        ),
        CheckConstraint("sequence >= 1", name="sequence_positive"),
        CheckConstraint("route_rank >= 0", name="route_rank_non_negative"),
        CheckConstraint("cost_estimated >= 0", name="attempt_cost_estimated_non_negative"),
        CheckConstraint(
            "cost_actual IS NULL OR cost_actual >= 0", name="attempt_cost_actual_non_negative"
        ),
    )


class Validation(TimestampMixin, Base):
    """One validator verdict against one attempt (§6 ``validations``)."""

    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    validator_name: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Whether this validator was required for the request's risk level. Only
    #: required validators gate success (§8).
    required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Whether this verdict determined the aggregate outcome (§8 precedence).
    aggregate_effect: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: An LLM judge runs as its own attempt; this links the verdict to it so
    #: judge cost is attributable (§6).
    validator_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("attempts.id", ondelete="SET NULL"), nullable=True
    )

    #: Machine-readable failure codes only. Never validator prose that could
    #: quote the model output.
    detail_codes: Mapped[list[str]] = mapped_column(JSONDocument, nullable=False, default=list)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    request: Mapped[Request] = relationship(back_populates="validations")
    attempt: Mapped[Attempt] = relationship(back_populates="validations", foreign_keys=[attempt_id])

    __table_args__ = (
        UniqueConstraint(
            "attempt_id", "validator_name", "sequence", name="uq_validations_attempt_validator"
        ),
        CheckConstraint(f"result IN {_enum_values(ValidationResult)}", name="result_known"),
        CheckConstraint("sequence >= 1", name="validation_sequence_positive"),
    )


# --------------------------------------------------------------------------
# Operational tables
# --------------------------------------------------------------------------


class BudgetReservation(TimestampMixin, Base):
    """A hold on budget headroom for one attempt (§6 ``budget_reservations``).

    Unique per attempt and budget, so a retry cannot double-reserve against the
    same budget, and every row must reach a settled, released, or expired
    status on some exit path.
    """

    __tablename__ = "budget_reservations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    budget_id: Mapped[int] = mapped_column(
        ForeignKey("budgets.id", ondelete="CASCADE"), nullable=False, index=True
    )

    reserved_amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    settled_amount: Mapped[Decimal | None] = mapped_column(Money, nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("attempt_id", "budget_id", name="uq_reservations_attempt_budget"),
        CheckConstraint(f"status IN {_enum_values(ReservationStatus)}", name="status_known"),
        CheckConstraint("reserved_amount >= 0", name="reserved_amount_non_negative"),
        CheckConstraint(
            "settled_amount IS NULL OR settled_amount >= 0", name="settled_amount_non_negative"
        ),
    )


class IdempotencyRecord(TimestampMixin, Base):
    """Replay protection for non-stream POSTs (§1, §6).

    The same key with the same canonical input replays the stored outcome; the
    same key with different input is a 409 conflict, which is why
    ``input_hash`` is part of the record rather than only the key.
    """

    __tablename__ = "idempotency_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)

    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("requests.id", ondelete="SET NULL"), nullable=True
    )

    state: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Reference to a stored response, not the response body. Retention for
    #: this reference defaults to 24 hours (§6).
    response_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("client_id", "idempotency_key", name="uq_idempotency_client_key"),
        CheckConstraint(
            f"state IN {_enum_values(IdempotencyState)}", name="idempotency_state_known"
        ),
    )


class APIKey(TimestampMixin, Base):
    """A client credential (§11).

    Only a keyed digest of the secret is stored, never the secret itself: a
    database disclosure must not hand an attacker working credentials.
    Rotation is expressed by issuing a new row and expiring the old one rather
    than editing a key in place, so audit history stays intact.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False, default="")

    #: HMAC of the presented secret under the deployment hash key.
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    #: Non-secret prefix, for identifying a key in logs without revealing it.
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, default="")

    scopes: Mapped[list[str]] = mapped_column(JSONDocument, nullable=False, default=list)
    #: Client-level control overrides, applied above headers and body (§2).
    control_overrides: Mapped[dict[str, Any]] = mapped_column(
        JSONDocument, nullable=False, default=dict
    )

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    #: Populated out of band, never on the authentication path: writing here
    #: per request would make every authenticated call contend on one row.
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        Index("ix_api_keys_client_enabled", "client_id", "enabled"),
    )
