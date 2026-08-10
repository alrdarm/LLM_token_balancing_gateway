"""Planning: classify, snapshot, gate, rank (§4, §7 PLANNING).

One entry point used by both ``/route/inspect`` and, from M5, the orchestrator.
Sharing it is what makes inspect-then-execute agree on the same snapshots
(scenario T08 in §12); two parallel implementations would drift.

This module reserves no budget, calls no provider, and creates no attempt rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from gateway.domain.enums import Quality, Risk, ValidationResult  # noqa: F401
from gateway.domain.requests import CanonicalRequest, ValidationMode
from gateway.domain.routing import RegistrySnapshot, RequestFeatures, RoutePlan
from gateway.persistence.repositories import PolicyRepository
from gateway.services import policy as policy_engine
from gateway.services import registry as registry_service
from gateway.services import router
from gateway.services.classifier import classify, quality_floor_for

#: Validators available in v0.1. M5 registers real implementations; a required
#: validator missing from this set excludes every model, because the request's
#: gates could not be enforced (§10 step 5).
AVAILABLE_VALIDATORS: frozenset[str] = frozenset(
    {
        "schema_check",
        "json_parse",
        "sql_parser",
        "sql_safety",
        "code_compile",
        "citation_check",
        "independent_review",
        "rubric_judge",
        "length_check",
        "label_check",
    }
)


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Everything an inspection or an orchestration start needs."""

    features: RequestFeatures
    snapshot: RegistrySnapshot
    plan: RoutePlan
    quality_floor: Quality


def _validation_plan_for(
    request: CanonicalRequest, features: RequestFeatures, policy_plan: list[str]
) -> tuple[str, ...]:
    """Resolve the validators this request must pass (§2, §8).

    ``validation=none`` narrows the plan but can never drop a hard schema gate:
    §2 says it must never bypass one, and a JSON schema the caller supplied is
    exactly that.
    """
    planned = list(policy_plan)

    if request.output_format.requires_schema_validation and "schema_check" not in planned:
        planned.insert(0, "schema_check")

    mode = request.controls.validation
    if mode == ValidationMode.NONE:
        # Hard schema gates survive; everything discretionary is dropped.
        planned = [name for name in planned if name == "schema_check"]
    elif mode == ValidationMode.DETERMINISTIC:
        planned = [name for name in planned if name not in {"independent_review", "rubric_judge"}]
    elif mode == ValidationMode.INDEPENDENT and "independent_review" not in planned:
        planned.append("independent_review")

    # High and critical risk always get an independent check unless the caller
    # explicitly restricted validation to deterministic checks (§8).
    if (
        features.risk in (Risk.HIGH, Risk.CRITICAL)
        and mode in (ValidationMode.AUTO, ValidationMode.INDEPENDENT)
        and "independent_review" not in planned
    ):
        planned.append("independent_review")

    return tuple(planned)


def build_plan(
    session: Session,
    request: CanonicalRequest,
    *,
    at: datetime | None = None,
) -> PlanningResult:
    """Classify, snapshot, gate, and rank -- without side effects."""
    features = classify(request)
    snapshot = registry_service.snapshot(session, at=at)

    policy = PolicyRepository(session).active_for(features.task_class.value)
    if policy is None:
        policy_id, policy_version = "none", 0
        policy_weights: dict[str, float] | None = None
        policy_validators: list[str] = []
        max_attempts, max_repairs = 3, 1
        policy_floor = Quality.STANDARD
    else:
        policy_id, policy_version = policy.policy_id, policy.version
        policy_weights = dict(policy.weights or {})
        policy_validators = list(policy.validation_plan or [])
        max_attempts = policy.max_generation_attempts
        max_repairs = policy.max_same_model_repairs
        policy_floor = Quality(policy.quality_floor)

    # The floor is the strictest of: policy, caller request, and risk.
    quality_floor = quality_floor_for(features.risk, request.controls.quality)
    from gateway.domain.requests import resolve_strictest_quality

    quality_floor = resolve_strictest_quality(quality_floor, policy_floor)

    validation_plan = _validation_plan_for(request, features, policy_validators)

    eligibility = policy_engine.evaluate(
        request,
        features,
        snapshot,
        quality_floor=quality_floor,
        available_validators=AVAILABLE_VALIDATORS,
        required_validators=frozenset(validation_plan),
    )

    route_plan = router.plan(
        request,
        features,
        snapshot,
        eligibility,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_weights=policy_weights,
        validation_plan=validation_plan,
        max_generation_attempts=max_attempts,
        max_same_model_repairs=max_repairs,
        now=at,
    )

    return PlanningResult(
        features=features,
        snapshot=snapshot,
        plan=route_plan,
        quality_floor=quality_floor,
    )
