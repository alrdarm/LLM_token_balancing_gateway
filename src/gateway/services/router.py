"""Scoring and route planning (§10).

The score is::

    score = Wc*expected_total_effective_cost
          + Wl*predicted_latency
          + Wq*quality_shortfall_risk
          + Wr*provider_failure_risk

Lower is better. Components are normalised before weighting so a change in
units -- dollars versus milliseconds -- cannot silently dominate the ranking.

"Expected total" is probability-weighted: a cheap model that often fails
validation costs more than its sticker price once the retry it forces is
counted. That is the whole reason price alone does not decide the route.

**Determinism is a contract.** §4 requires the same canonical input and
snapshots to rank identically, tie-broken by score, then predicted cost, then
model ID. Ties are broken on the exact ``Decimal`` cost rather than the float
score.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.domain.requests import CanonicalRequest
from gateway.domain.routing import (
    Candidate,
    Eligibility,
    ModelSnapshot,
    RationaleCode,
    RegistrySnapshot,
    RequestFeatures,
    RoutePlan,
)
from gateway.services.policy import estimate_cost

#: Default weights when a policy supplies none (§10). Cost-leaning but not
#: cost-only, because quality shortfall drives repairs and escalations that
#: cost more than the saving.
DEFAULT_WEIGHTS: dict[str, float] = {
    "cost": 0.4,
    "latency": 0.2,
    "quality_shortfall": 0.3,
    "failure_risk": 0.1,
}

#: Selector-specific weight overrides (§1). ``auto`` uses the policy's own
#: balance; the others express an explicit preference.
SELECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "auto-cheap": {"cost": 0.8, "latency": 0.05, "quality_shortfall": 0.1, "failure_risk": 0.05},
    "auto-fast": {"cost": 0.1, "latency": 0.7, "quality_shortfall": 0.15, "failure_risk": 0.05},
    "auto-quality": {"cost": 0.05, "latency": 0.05, "quality_shortfall": 0.8, "failure_risk": 0.1},
}

#: Reference scale for the quota scarcity penalty, which must be expressed in
#: money to be added to a monetary cost (§10).
COST_NORMALISER = Decimal("1.000000000")


def _min_max(values: list[float]) -> list[float]:
    """Scale ``values`` onto 0..1 relative to each other.

    Components are normalised **across the candidate set**, not against fixed
    constants. Absolute normalisers cannot work here: a realistic request costs
    fractions of a cent while latency runs to seconds, so any constant divisor
    leaves one component's spread orders of magnitude smaller than the other's
    and the weights stop meaning what they say. Relative scaling makes a weight
    of 0.8 actually dominate.

    An all-equal set scores 0 throughout, so an indifferent component cannot
    tip the ranking.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    span = high - low
    return [(value - low) / span for value in values]


def expected_total_cost(
    model: ModelSnapshot,
    *,
    single_attempt_cost: Decimal,
    max_attempts: int,
) -> Decimal:
    """Probability-weighted cost including the retries a failure would force.

    A model that passes validation 60% of the time will, on average, be invoked
    more than once. Ranking on single-attempt price alone would systematically
    favour exactly the models that generate the most rework.
    """
    pass_rate = max(min(model.pass_rate_prior, 1.0), 0.0)
    expected_attempts = 1.0
    probability_of_reaching = 1.0 - pass_rate

    for _ in range(max(max_attempts - 1, 0)):
        expected_attempts += probability_of_reaching
        probability_of_reaching *= 1.0 - pass_rate

    multiplier = Decimal(str(round(expected_attempts, 6)))
    return (single_attempt_cost * multiplier).quantize(Decimal("0.000000001"))


def quota_scarcity_penalty(model: ModelSnapshot, scarcity: float) -> Decimal:
    """Convert quota pressure into a monetary-equivalent penalty (§10).

    ``effective_cost = monetary_cost + quota_scarcity_penalty``: a model whose
    quota is nearly gone is worth avoiding even when it is nominally cheapest,
    because exhausting it removes a route that later requests will need.
    """
    if scarcity <= 0:
        return Decimal("0")
    return (COST_NORMALISER * Decimal(str(min(scarcity, 1.0))) / Decimal(100)).quantize(
        Decimal("0.000000001")
    )


def _quality_shortfall(model: ModelSnapshot, features: RequestFeatures) -> float:
    """How likely this model is to fall short of the required quality."""
    return 1.0 - max(min(model.pass_rate_prior, 1.0), 0.0)


def effective_cost_for(
    model: ModelSnapshot,
    *,
    single_attempt_cost: Decimal,
    max_attempts: int,
    scarcity: float = 0.0,
) -> Decimal:
    """Expected total cost plus any quota scarcity penalty (§10)."""
    expected_cost = expected_total_cost(
        model, single_attempt_cost=single_attempt_cost, max_attempts=max_attempts
    )
    return expected_cost + quota_scarcity_penalty(model, scarcity)


def weights_for(requested_model: str, policy_weights: dict[str, float] | None) -> dict[str, float]:
    """Resolve the weight vector for a request.

    A selector's explicit preference wins over the policy default, because the
    caller chose it deliberately; ``auto`` defers to policy.
    """
    if requested_model in SELECTOR_WEIGHTS:
        return dict(SELECTOR_WEIGHTS[requested_model])

    weights = dict(DEFAULT_WEIGHTS)
    if policy_weights:
        weights.update({key: float(value) for key, value in policy_weights.items()})
    return weights


def rank(
    eligibility: Eligibility,
    features: RequestFeatures,
    request: CanonicalRequest,
    *,
    weights: dict[str, float],
    max_attempts: int,
) -> tuple[Candidate, ...]:
    """Score and order the eligible models.

    Sorted by (score, estimated cost, model ID). The exact Decimal cost breaks
    score ties before the ID does, so ranking never depends on float equality.
    """
    models = list(eligibility.eligible)
    if not models:
        return ()

    single_costs = [
        estimate_cost(
            model,
            input_tokens=request.estimated_input_tokens,
            output_tokens=features.expected_output_tokens,
        )
        for model in models
    ]
    effective_costs = [
        effective_cost_for(model, single_attempt_cost=single, max_attempts=max_attempts)
        for model, single in zip(models, single_costs, strict=True)
    ]

    # Normalise each component across the candidate set so the weights apply to
    # comparable magnitudes.
    normalised = {
        "cost": _min_max([float(cost) for cost in effective_costs]),
        "latency": _min_max([float(model.latency_prior_ms) for model in models]),
        "quality_shortfall": _min_max([_quality_shortfall(model, features) for model in models]),
        "failure_risk": _min_max([model.failure_rate_prior for model in models]),
    }

    scored: list[
        tuple[float, Decimal, str, ModelSnapshot, Decimal, tuple[tuple[str, float], ...]]
    ] = []

    for index, model in enumerate(models):
        parts = {
            component: weights[component] * values[index]
            for component, values in normalised.items()
        }
        score = sum(parts.values())
        scored.append(
            (
                score,
                single_costs[index],
                model.model_id,
                model,
                effective_costs[index],
                tuple(sorted(parts.items())),
            )
        )

    scored.sort(key=lambda row: (round(row[0], 9), row[1], row[2]))

    return tuple(
        Candidate(
            model_id=model.model_id,
            provider=model.provider,
            rank=position + 1,
            estimated_cost=estimated,
            effective_cost=effective,
            predicted_latency_ms=model.latency_prior_ms,
            predicted_pass_probability=model.pass_rate_prior,
            score=score,
            score_components=parts,
        )
        for position, (score, estimated, _, model, effective, parts) in enumerate(scored)
    )


def route_upper_bound(candidates: tuple[Candidate, ...], max_attempts: int) -> Decimal:
    """Worst-case spend if every allowed attempt is used (§4).

    Uses the most expensive candidates the plan could actually reach, so the
    figure is an upper bound rather than an average.
    """
    if not candidates:
        return Decimal("0")

    costs = sorted((candidate.estimated_cost for candidate in candidates), reverse=True)
    reachable = costs[:max_attempts] or [costs[0]]
    while len(reachable) < max_attempts:
        reachable.append(reachable[-1])
    return sum(reachable, Decimal("0")).quantize(Decimal("0.000000001"))


def plan(
    request: CanonicalRequest,
    features: RequestFeatures,
    snapshot: RegistrySnapshot,
    eligibility: Eligibility,
    *,
    policy_id: str,
    policy_version: int,
    policy_weights: dict[str, float] | None,
    validation_plan: tuple[str, ...],
    max_generation_attempts: int,
    max_same_model_repairs: int,
    now: datetime | None = None,
) -> RoutePlan:
    """Build the frozen plan for one request (§5)."""
    now = now or datetime.now(UTC)
    controls = request.controls

    # The plan is capped by whichever is stricter: policy or caller.
    max_attempts = min(max_generation_attempts, controls.max_attempts)

    weights = weights_for(request.requested_model, policy_weights)
    candidates = rank(eligibility, features, request, weights=weights, max_attempts=max_attempts)

    if not request.is_selector:
        # An explicit model is honoured only if it survived the gates; §1 says
        # explicit IDs still pass governance.
        candidates = (
            tuple(
                candidate
                for candidate in candidates
                if candidate.model_id == request.requested_model
            )
            or candidates
        )
        if not controls.allow_fallback:
            candidates = candidates[:1]

    rationale: list[str] = []
    if request.is_selector:
        rationale.append(RationaleCode.SELECTOR_APPLIED)
    else:
        rationale.append(RationaleCode.EXPLICIT_MODEL)
        if not controls.allow_fallback:
            rationale.append(RationaleCode.FALLBACK_DISABLED)
    if features.used_fallback:
        rationale.append(RationaleCode.CLASSIFIER_FALLBACK)
    if not candidates:
        rationale.append(RationaleCode.NO_ELIGIBLE_CANDIDATES)

    latency_budget = controls.max_latency_ms or 60_000
    deadline_at = now + timedelta(milliseconds=latency_budget)

    return RoutePlan(
        policy_id=policy_id,
        policy_version=policy_version,
        registry_snapshot_at=snapshot.taken_at,
        candidates=candidates,
        excluded=eligibility.excluded,
        validation_plan=validation_plan,
        max_generation_attempts=max_attempts,
        max_same_model_repairs=max_same_model_repairs,
        estimated_route_upper_bound=route_upper_bound(candidates, max_attempts),
        deadline_at=deadline_at,
        rationale_codes=tuple(rationale),
    )
