"""Routing domain types (§5, §10).

A route plan is *frozen*: it records the policy version and registry snapshot
time it was built from, so a completed request stays reconstructible even after
policies change or models are disabled. Nothing here mutates after
construction.

Scores are floats because they are rankings, not money. Every monetary value on
these types stays :class:`~decimal.Decimal`, and tie-breaking uses the exact
cost rather than the float score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from gateway.domain.enums import (
    Capability,
    DataHandlingTier,
    Privacy,
    Quality,
    Risk,
    TaskClass,
)


class ExclusionReason:
    """Why a model was excluded, as stable machine-readable codes (§4)."""

    PRIVACY_MISMATCH = "privacy_mismatch"
    MISSING_CAPABILITY = "missing_capability"
    ENDPOINT_UNSUPPORTED = "endpoint_unsupported"
    CONTEXT_TOO_SMALL = "context_too_small"
    OUTPUT_WINDOW_TOO_SMALL = "output_window_too_small"
    BELOW_QUALITY_FLOOR = "below_quality_floor"
    VALIDATOR_UNAVAILABLE = "validator_unavailable"
    PROVIDER_DENIED = "provider_denied"
    PROVIDER_NOT_ALLOWED = "provider_not_allowed"
    MODEL_DISABLED = "model_disabled"
    UNHEALTHY = "unhealthy"
    QUOTA_EXHAUSTED = "quota_exhausted"
    QUOTA_STALE = "quota_stale"
    OVER_REQUEST_BUDGET = "over_request_budget"
    OVER_SCOPED_BUDGET = "over_scoped_budget"
    DEADLINE_INFEASIBLE = "deadline_infeasible"
    NO_PRICE = "no_price"


class RationaleCode:
    """Why a plan looks the way it does, for inspection output."""

    SELECTOR_APPLIED = "selector_applied"
    EXPLICIT_MODEL = "explicit_model"
    FALLBACK_DISABLED = "fallback_disabled"
    QUALITY_FLOOR_RAISED = "quality_floor_raised"
    PRIVACY_FLOOR_APPLIED = "privacy_floor_applied"
    CLASSIFIER_FALLBACK = "classifier_fallback"
    NO_ELIGIBLE_CANDIDATES = "no_eligible_candidates"


@dataclass(frozen=True, slots=True)
class RequestFeatures:
    """Classifier output, frozen for the life of the request (§5).

    ``confidence`` and ``classifier_version`` travel with the features so a
    routing decision made on a low-confidence guess is auditable after the
    fact.
    """

    task_class: TaskClass
    complexity: int
    risk: Risk
    privacy: Privacy
    verifiability: str
    freshness_sensitive: bool
    required_capabilities: frozenset[Capability]
    expected_output_tokens: int
    classifier_name: str
    classifier_version: str
    confidence: float
    used_fallback: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.complexity <= 5:
            raise ValueError(f"complexity must be 1..5, got {self.complexity}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0..1, got {self.confidence}")


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """One registry entry, frozen at snapshot time (§5).

    A snapshot decouples ranking from the database: the plan is built from this
    immutable view, so a concurrent registry edit cannot change a decision
    halfway through.
    """

    model_id: str
    provider: str
    data_handling_tier: DataHandlingTier
    quality_tier: Quality
    capabilities: frozenset[Capability]
    supported_endpoints: frozenset[str]
    context_window_tokens: int
    max_output_tokens: int

    input_per_1k: Decimal
    output_per_1k: Decimal
    request_fee: Decimal

    #: Conservative v0.1 priors, sourced from policy data rather than measured.
    latency_prior_ms: int
    pass_rate_prior: float
    failure_rate_prior: float
    priors_source: str


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """The set of models a plan was built from, and when."""

    taken_at: datetime
    models: tuple[ModelSnapshot, ...]

    def by_id(self, model_id: str) -> ModelSnapshot | None:
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None


@dataclass(frozen=True, slots=True)
class ExcludedModel:
    """A model that failed at least one gate, with every reason it failed."""

    model_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    """An eligible model with its score components (§10)."""

    model_id: str
    provider: str
    rank: int

    estimated_cost: Decimal
    effective_cost: Decimal
    predicted_latency_ms: int
    predicted_pass_probability: float

    score: float
    #: Component contributions, retained so a ranking can be explained.
    score_components: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Result of running the ordered eligibility pipeline (§10)."""

    eligible: tuple[ModelSnapshot, ...]
    excluded: tuple[ExcludedModel, ...]

    @property
    def has_route(self) -> bool:
        return bool(self.eligible)


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A frozen plan for one request (§5)."""

    policy_id: str
    policy_version: int
    registry_snapshot_at: datetime
    candidates: tuple[Candidate, ...]
    excluded: tuple[ExcludedModel, ...]
    validation_plan: tuple[str, ...]
    max_generation_attempts: int
    max_same_model_repairs: int
    estimated_route_upper_bound: Decimal
    deadline_at: datetime
    rationale_codes: tuple[str, ...] = field(default=())

    @property
    def has_route(self) -> bool:
        return bool(self.candidates)

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None
