"""Domain value sets.

These mirror the specification's vocabulary exactly. They are stored as strings
rather than native database enums so a new member does not require a schema
migration on PostgreSQL, and so SQLite and PostgreSQL store identical values.

Ordering matters in two places and is encoded explicitly rather than left to
declaration order: :data:`VALIDATION_SEVERITY` (§8 aggregation) and the quality
floor comparison in :class:`Quality`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum


class RequestState(StrEnum):
    """Lifecycle states from §7."""

    RECEIVED = "RECEIVED"
    AUTHENTICATING = "AUTHENTICATING"
    NORMALIZING = "NORMALIZING"
    CLASSIFYING = "CLASSIFYING"
    PLANNING = "PLANNING"
    READY = "READY"
    RESERVING = "RESERVING"
    INVOKING = "INVOKING"
    RETRY_WAIT = "RETRY_WAIT"
    VALIDATING = "VALIDATING"
    REPAIR_PLANNING = "REPAIR_PLANNING"
    ESCALATION_PLANNING = "ESCALATION_PLANNING"

    # Terminal
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    REJECTED_NO_ROUTE = "REJECTED_NO_ROUTE"
    REJECTED_BUDGET = "REJECTED_BUDGET"
    FAILED = "FAILED"
    FAILED_PARTIAL = "FAILED_PARTIAL"
    FAILED_EXHAUSTED = "FAILED_EXHAUSTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


#: Terminal states (§7). Every terminal path must settle or release its
#: reservations and persist a terminal reason.
TERMINAL_STATES: frozenset[RequestState] = frozenset(
    {
        RequestState.SUCCEEDED,
        RequestState.REJECTED,
        RequestState.REJECTED_NO_ROUTE,
        RequestState.REJECTED_BUDGET,
        RequestState.FAILED,
        RequestState.FAILED_PARTIAL,
        RequestState.FAILED_EXHAUSTED,
        RequestState.CANCELLED,
        RequestState.EXPIRED,
    }
)


def is_terminal(state: RequestState) -> bool:
    """Whether ``state`` admits no further transitions."""
    return state in TERMINAL_STATES


class AttemptKind(StrEnum):
    """Why an attempt was made.

    ``VALIDATOR`` attempts (for example an LLM judge) are excluded from the
    ``max_attempts`` cap but their cost still counts against the ceiling (§2).
    """

    GENERATION = "GENERATION"
    REPAIR = "REPAIR"
    ESCALATION = "ESCALATION"
    VALIDATOR = "VALIDATOR"


#: Attempt kinds that count toward the generation/repair cap.
CAPPED_ATTEMPT_KINDS: frozenset[AttemptKind] = frozenset(
    {AttemptKind.GENERATION, AttemptKind.REPAIR, AttemptKind.ESCALATION}
)


class AttemptOutcome(StrEnum):
    """Provider-level result of an attempt (§8)."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    CAPABILITY_REJECTED = "CAPABILITY_REJECTED"
    CONTEXT_REJECTED = "CONTEXT_REJECTED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


class ValidationResult(StrEnum):
    """Validator verdicts (§8)."""

    PASS = "PASS"  # noqa: S105 - a validation verdict, not a credential
    INDETERMINATE = "INDETERMINATE"
    FAIL_REPAIRABLE = "FAIL_REPAIRABLE"
    FAIL_QUALITY = "FAIL_QUALITY"
    FAIL_CAPABILITY = "FAIL_CAPABILITY"
    FAIL_GROUNDING = "FAIL_GROUNDING"


#: Aggregation precedence, most severe first (§8):
#: FAIL_GROUNDING > FAIL_CAPABILITY > FAIL_QUALITY > FAIL_REPAIRABLE
#: > INDETERMINATE > PASS
VALIDATION_SEVERITY: tuple[ValidationResult, ...] = (
    ValidationResult.FAIL_GROUNDING,
    ValidationResult.FAIL_CAPABILITY,
    ValidationResult.FAIL_QUALITY,
    ValidationResult.FAIL_REPAIRABLE,
    ValidationResult.INDETERMINATE,
    ValidationResult.PASS,
)


class Quality(StrEnum):
    """Quality floor (§2). Comparable: a floor is satisfied by an equal or
    higher tier."""

    ECONOMY = "economy"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


#: Ascending strictness. Used for "strictest applicable value" resolution (§2).
QUALITY_ORDER: tuple[Quality, ...] = (
    Quality.ECONOMY,
    Quality.STANDARD,
    Quality.HIGH,
    Quality.CRITICAL,
)


class Privacy(StrEnum):
    """Caller-facing privacy level (§2, §11)."""

    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    DEPLOYMENT_STRICT = "deployment_strict"


#: Ascending strictness.
PRIVACY_ORDER: tuple[Privacy, ...] = (
    Privacy.PUBLIC,
    Privacy.CONFIDENTIAL,
    Privacy.DEPLOYMENT_STRICT,
)


class DataHandlingTier(StrEnum):
    """Provider data-handling class a model belongs to (§11).

    ``PUBLIC_ONLY`` models may handle public data only; ``ZDR`` denotes
    zero-data-retention agreements.
    """

    PUBLIC_ONLY = "public_only"
    STANDARD = "standard"
    ZDR = "zdr"


#: Which model tiers each privacy level may route to (§11 table).
PRIVACY_ELIGIBLE_TIERS: dict[Privacy, frozenset[DataHandlingTier]] = {
    Privacy.PUBLIC: frozenset(
        {DataHandlingTier.PUBLIC_ONLY, DataHandlingTier.STANDARD, DataHandlingTier.ZDR}
    ),
    Privacy.CONFIDENTIAL: frozenset({DataHandlingTier.STANDARD, DataHandlingTier.ZDR}),
    Privacy.DEPLOYMENT_STRICT: frozenset({DataHandlingTier.ZDR}),
}


class Risk(StrEnum):
    """Inferred or caller-raised risk. A caller may raise risk, never lower
    the inferred value (§2)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


#: Ascending strictness.
RISK_ORDER: tuple[Risk, ...] = (Risk.LOW, Risk.MEDIUM, Risk.HIGH, Risk.CRITICAL)


class TaskClass(StrEnum):
    """Initial task classes and their acceptance gates (§10)."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    TRANSFORMATION = "transformation"
    SUMMARIZATION = "summarization"
    REWRITING = "rewriting"
    CREATIVE_WRITING = "creative_writing"
    FACTUAL_QA = "factual_qa"
    GENERAL_REASONING = "general_reasoning"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    SQL_REVIEW = "sql_review"
    STRUCTURED_DATA = "structured_data"
    TOOL_SELECTION = "tool_selection"
    GROUNDED_QA = "grounded_qa"


class Capability(StrEnum):
    """Capabilities a request may require of a model (§2)."""

    TOOLS = "tools"
    JSON_SCHEMA = "json_schema"
    VISION = "vision"
    STREAMING = "streaming"
    REASONING = "reasoning"


class Endpoint(StrEnum):
    """Generation surfaces (§1)."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class ReservationStatus(StrEnum):
    """Budget reservation lifecycle (§6)."""

    ACTIVE = "ACTIVE"
    SETTLED = "SETTLED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"


class IdempotencyState(StrEnum):
    """Idempotency record lifecycle (§1, §6)."""

    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BudgetWindow(StrEnum):
    """Budget accounting window (§6)."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    TOTAL = "total"


def strictest[T](values: Iterable[T], order: Sequence[T]) -> T:
    """Return the strictest member of ``values`` according to ``order``.

    The spec resolves privacy, risk, and quality to the strictest applicable
    value rather than the most recently applied one (§2). ``order`` is
    ascending strictness, e.g. :data:`QUALITY_ORDER`.
    """
    members = list(values)
    if not members:
        raise ValueError("strictest() requires at least one value")
    unknown = [value for value in members if value not in order]
    if unknown:
        raise ValueError(f"values outside the ordering: {unknown!r}")
    return max(members, key=order.index)
