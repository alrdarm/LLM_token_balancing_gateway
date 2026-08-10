"""Policy engine: the ordered eligibility pipeline (§10).

The gates run in the spec's order, and every one is a **hard** gate. Price
never overrides any of them -- cost only decides ranking *among* survivors.

A model is checked against every gate rather than short-circuiting on the first
failure, because ``/route/inspect`` must be able to explain all the reasons a
model was excluded, not just the first one encountered.
"""

from __future__ import annotations

from decimal import Decimal

from gateway.domain.enums import PRIVACY_ELIGIBLE_TIERS, QUALITY_ORDER, Quality
from gateway.domain.requests import CanonicalRequest
from gateway.domain.routing import (
    Eligibility,
    ExcludedModel,
    ExclusionReason,
    ModelSnapshot,
    RegistrySnapshot,
    RequestFeatures,
)

#: Tokens of headroom required beyond the estimate, so a slightly long prompt
#: does not overflow a window that was only just large enough.
CONTEXT_SAFETY_MARGIN = 256


def estimate_cost(model: ModelSnapshot, *, input_tokens: int, output_tokens: int) -> Decimal:
    """Estimated monetary cost of one invocation (§6).

    ``price(input_estimate, expected_or_max_output) + request_fee``, in exact
    decimal throughout -- this feeds budget reservation, where rounding would
    let spend drift past a ceiling.
    """
    thousand = Decimal(1000)
    input_cost = (Decimal(input_tokens) / thousand) * model.input_per_1k
    output_cost = (Decimal(output_tokens) / thousand) * model.output_per_1k
    return (input_cost + output_cost + model.request_fee).quantize(Decimal("0.000000001"))


def _quality_satisfies(model: Quality, floor: Quality) -> bool:
    return QUALITY_ORDER.index(model) >= QUALITY_ORDER.index(floor)


def evaluate(
    request: CanonicalRequest,
    features: RequestFeatures,
    snapshot: RegistrySnapshot,
    *,
    quality_floor: Quality,
    available_validators: frozenset[str],
    required_validators: frozenset[str],
    remaining_budget: Decimal | None = None,
    unhealthy_models: frozenset[str] = frozenset(),
    exhausted_models: frozenset[str] = frozenset(),
) -> Eligibility:
    """Run every gate against every model in ``snapshot``.

    Returns the survivors and, for each rejection, all the reasons it failed.
    """
    controls = request.controls
    eligible: list[ModelSnapshot] = []
    excluded: list[ExcludedModel] = []

    eligible_tiers = PRIVACY_ELIGIBLE_TIERS[features.privacy]
    output_tokens = features.expected_output_tokens

    for model in snapshot.models:
        reasons: list[str] = []

        # 2. privacy / data-handling eligibility
        if model.data_handling_tier not in eligible_tiers:
            reasons.append(ExclusionReason.PRIVACY_MISMATCH)

        # 3. required capabilities and endpoint support
        missing = features.required_capabilities - model.capabilities
        if missing:
            reasons.append(ExclusionReason.MISSING_CAPABILITY)
        if request.endpoint.value not in model.supported_endpoints:
            reasons.append(ExclusionReason.ENDPOINT_UNSUPPORTED)

        # 4. context and output window fit
        needed = request.estimated_input_tokens + output_tokens + CONTEXT_SAFETY_MARGIN
        if needed > model.context_window_tokens:
            reasons.append(ExclusionReason.CONTEXT_TOO_SMALL)
        if output_tokens > model.max_output_tokens:
            reasons.append(ExclusionReason.OUTPUT_WINDOW_TOO_SMALL)

        # 5. risk quality floor and validator availability
        if not _quality_satisfies(model.quality_tier, quality_floor):
            reasons.append(ExclusionReason.BELOW_QUALITY_FLOOR)
        if required_validators - available_validators:
            # A required validator that cannot run means the request's gates
            # cannot be enforced, so no model is eligible to attempt it.
            reasons.append(ExclusionReason.VALIDATOR_UNAVAILABLE)

        # 6. allow / deny and override rules
        if model.provider in controls.provider_deny:
            reasons.append(ExclusionReason.PROVIDER_DENIED)
        elif controls.provider_allow and model.provider not in controls.provider_allow:
            reasons.append(ExclusionReason.PROVIDER_NOT_ALLOWED)

        # 7. health / circuit / quota
        if model.model_id in unhealthy_models:
            reasons.append(ExclusionReason.UNHEALTHY)
        if model.model_id in exhausted_models:
            reasons.append(ExclusionReason.QUOTA_EXHAUSTED)

        # 8. request and scoped budget feasibility
        cost = estimate_cost(
            model,
            input_tokens=request.estimated_input_tokens,
            output_tokens=output_tokens,
        )
        if controls.max_cost is not None and cost > controls.max_cost:
            reasons.append(ExclusionReason.OVER_REQUEST_BUDGET)
        if remaining_budget is not None and cost > remaining_budget:
            reasons.append(ExclusionReason.OVER_SCOPED_BUDGET)

        # 9. deadline feasibility
        if controls.max_latency_ms is not None and model.latency_prior_ms > controls.max_latency_ms:
            reasons.append(ExclusionReason.DEADLINE_INFEASIBLE)

        if reasons:
            excluded.append(ExcludedModel(model_id=model.model_id, reasons=tuple(reasons)))
        else:
            eligible.append(model)

    return Eligibility(eligible=tuple(eligible), excluded=tuple(excluded))
