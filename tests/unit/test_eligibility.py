"""Eligibility suite: every gate in the §10 pipeline.

Each gate is a **hard** gate. Price never overrides one, so every test here
sets up a model that would otherwise win on cost and asserts it is excluded
anyway.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from gateway.domain.enums import (
    Capability,
    DataHandlingTier,
    Endpoint,
    Privacy,
    Quality,
    Risk,
    TaskClass,
)
from gateway.domain.requests import CanonicalRequest, GatewayControls, Message
from gateway.domain.routing import (
    ExclusionReason,
    ModelSnapshot,
    RegistrySnapshot,
    RequestFeatures,
)
from gateway.services.policy import estimate_cost, evaluate

NOW_TOKENS = 100

ALL_VALIDATORS = frozenset({"schema_check", "sql_parser", "independent_review"})


def make_model(**overrides: object) -> ModelSnapshot:
    defaults = dict(
        model_id="m/test",
        provider="fake",
        data_handling_tier=DataHandlingTier.STANDARD,
        quality_tier=Quality.STANDARD,
        capabilities=frozenset({Capability.STREAMING, Capability.TOOLS}),
        supported_endpoints=frozenset({"chat_completions", "responses"}),
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        input_per_1k=Decimal("0.001"),
        output_per_1k=Decimal("0.002"),
        request_fee=Decimal("0"),
        latency_prior_ms=1000,
        pass_rate_prior=0.8,
        failure_rate_prior=0.02,
        priors_source="test",
    )
    defaults.update(overrides)
    return ModelSnapshot(**defaults)  # type: ignore[arg-type]


def make_request(**control_overrides: object) -> CanonicalRequest:
    controls = replace(GatewayControls(), **control_overrides)
    return CanonicalRequest(
        request_id="req_1",
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model="auto",
        conversation=(Message(role="user", content="hello"),),
        controls=controls,
        client_id="c",
        estimated_input_tokens=NOW_TOKENS,
    )


def make_features(**overrides: object) -> RequestFeatures:
    defaults = dict(
        task_class=TaskClass.GENERAL_REASONING,
        complexity=2,
        risk=Risk.MEDIUM,
        privacy=Privacy.CONFIDENTIAL,
        verifiability="subjective",
        freshness_sensitive=False,
        required_capabilities=frozenset(),
        expected_output_tokens=500,
        classifier_name="test",
        classifier_version="v0",
        confidence=0.9,
    )
    defaults.update(overrides)
    return RequestFeatures(**defaults)  # type: ignore[arg-type]


def run(models, request=None, features=None, **kwargs: object):
    snapshot = RegistrySnapshot(taken_at=datetime.now(UTC), models=tuple(models))
    return evaluate(
        request or make_request(),
        features or make_features(),
        snapshot,
        quality_floor=kwargs.pop("quality_floor", Quality.ECONOMY),
        available_validators=kwargs.pop("available_validators", ALL_VALIDATORS),
        required_validators=kwargs.pop("required_validators", frozenset()),
        **kwargs,
    )


def reasons_for(result, model_id: str) -> tuple[str, ...]:
    for excluded in result.excluded:
        if excluded.model_id == model_id:
            return excluded.reasons
    return ()


# --- gate 2: privacy -------------------------------------------------------


def test_confidential_excludes_public_only_models():
    """T01 in §12: a confidential request must never reach a public-only tier."""
    model = make_model(data_handling_tier=DataHandlingTier.PUBLIC_ONLY)
    result = run([model], features=make_features(privacy=Privacy.CONFIDENTIAL))

    assert not result.has_route
    assert ExclusionReason.PRIVACY_MISMATCH in reasons_for(result, "m/test")


def test_cheapest_model_is_still_excluded_on_privacy():
    """Price never overrides a hard gate (§2)."""
    cheap_public = make_model(
        model_id="m/cheap",
        data_handling_tier=DataHandlingTier.PUBLIC_ONLY,
        input_per_1k=Decimal("0.000001"),
        output_per_1k=Decimal("0.000001"),
    )
    expensive_private = make_model(
        model_id="m/expensive",
        data_handling_tier=DataHandlingTier.ZDR,
        input_per_1k=Decimal("1"),
        output_per_1k=Decimal("1"),
    )
    result = run([cheap_public, expensive_private])

    assert [model.model_id for model in result.eligible] == ["m/expensive"]


def test_deployment_strict_admits_only_zdr():
    models = [
        make_model(model_id="m/std", data_handling_tier=DataHandlingTier.STANDARD),
        make_model(model_id="m/zdr", data_handling_tier=DataHandlingTier.ZDR),
    ]
    result = run(models, features=make_features(privacy=Privacy.DEPLOYMENT_STRICT))
    assert [model.model_id for model in result.eligible] == ["m/zdr"]


def test_public_request_may_use_every_tier():
    models = [
        make_model(model_id=f"m/{tier.value}", data_handling_tier=tier) for tier in DataHandlingTier
    ]
    result = run(models, features=make_features(privacy=Privacy.PUBLIC))
    assert len(result.eligible) == 3


# --- gate 3: capabilities and endpoint ------------------------------------


def test_missing_capability_excludes():
    model = make_model(capabilities=frozenset({Capability.STREAMING}))
    result = run(
        [model], features=make_features(required_capabilities=frozenset({Capability.VISION}))
    )
    assert ExclusionReason.MISSING_CAPABILITY in reasons_for(result, "m/test")


def test_unsupported_endpoint_excludes():
    model = make_model(supported_endpoints=frozenset({"responses"}))
    result = run([model])
    assert ExclusionReason.ENDPOINT_UNSUPPORTED in reasons_for(result, "m/test")


# --- gate 4: context and output windows -----------------------------------


def test_context_window_too_small_excludes():
    model = make_model(context_window_tokens=200)
    result = run([model], features=make_features(expected_output_tokens=500))
    assert ExclusionReason.CONTEXT_TOO_SMALL in reasons_for(result, "m/test")


def test_output_window_too_small_excludes():
    model = make_model(max_output_tokens=100)
    result = run([model], features=make_features(expected_output_tokens=500))
    assert ExclusionReason.OUTPUT_WINDOW_TOO_SMALL in reasons_for(result, "m/test")


def test_context_check_includes_a_safety_margin():
    """A window that only just fits is rejected; a prompt is an estimate."""
    from gateway.services.policy import CONTEXT_SAFETY_MARGIN

    exactly = NOW_TOKENS + 500
    model = make_model(context_window_tokens=exactly + CONTEXT_SAFETY_MARGIN - 1)
    result = run([model], features=make_features(expected_output_tokens=500))
    assert ExclusionReason.CONTEXT_TOO_SMALL in reasons_for(result, "m/test")


# --- gate 5: quality floor and validators ---------------------------------


def test_below_quality_floor_excludes():
    model = make_model(quality_tier=Quality.ECONOMY)
    result = run([model], quality_floor=Quality.HIGH)
    assert ExclusionReason.BELOW_QUALITY_FLOOR in reasons_for(result, "m/test")


def test_equal_quality_tier_satisfies_the_floor():
    model = make_model(quality_tier=Quality.HIGH)
    result = run([model], quality_floor=Quality.HIGH)
    assert result.has_route


def test_higher_quality_tier_satisfies_the_floor():
    model = make_model(quality_tier=Quality.CRITICAL)
    result = run([model], quality_floor=Quality.STANDARD)
    assert result.has_route


def test_unavailable_required_validator_excludes_everything():
    """If a required gate cannot run, no model may attempt the request."""
    result = run(
        [make_model()],
        required_validators=frozenset({"nonexistent_validator"}),
        available_validators=ALL_VALIDATORS,
    )
    assert not result.has_route
    assert ExclusionReason.VALIDATOR_UNAVAILABLE in reasons_for(result, "m/test")


# --- gate 6: allow / deny --------------------------------------------------


def test_denied_provider_excludes():
    result = run([make_model()], request=make_request(provider_deny=("fake",)))
    assert ExclusionReason.PROVIDER_DENIED in reasons_for(result, "m/test")


def test_provider_not_in_allow_list_excludes():
    result = run([make_model()], request=make_request(provider_allow=("other",)))
    assert ExclusionReason.PROVIDER_NOT_ALLOWED in reasons_for(result, "m/test")


def test_deny_wins_over_allow():
    result = run(
        [make_model()], request=make_request(provider_allow=("fake",), provider_deny=("fake",))
    )
    assert ExclusionReason.PROVIDER_DENIED in reasons_for(result, "m/test")


# --- gate 7: health and quota ---------------------------------------------


def test_unhealthy_model_excluded():
    result = run([make_model()], unhealthy_models=frozenset({"m/test"}))
    assert ExclusionReason.UNHEALTHY in reasons_for(result, "m/test")


def test_quota_exhausted_model_excluded():
    result = run([make_model()], exhausted_models=frozenset({"m/test"}))
    assert ExclusionReason.QUOTA_EXHAUSTED in reasons_for(result, "m/test")


# --- gate 8: budget --------------------------------------------------------


def test_over_request_ceiling_excludes():
    model = make_model(input_per_1k=Decimal("10"), output_per_1k=Decimal("10"))
    result = run([model], request=make_request(max_cost=Decimal("0.001")))
    assert ExclusionReason.OVER_REQUEST_BUDGET in reasons_for(result, "m/test")


def test_within_request_ceiling_is_eligible():
    result = run([make_model()], request=make_request(max_cost=Decimal("1.0")))
    assert result.has_route


def test_over_scoped_budget_excludes():
    result = run([make_model()], remaining_budget=Decimal("0.0000001"))
    assert ExclusionReason.OVER_SCOPED_BUDGET in reasons_for(result, "m/test")


# --- gate 9: deadline ------------------------------------------------------


def test_model_slower_than_the_deadline_excludes():
    model = make_model(latency_prior_ms=9000)
    result = run([model], request=make_request(max_latency_ms=1000))
    assert ExclusionReason.DEADLINE_INFEASIBLE in reasons_for(result, "m/test")


# --- reporting -------------------------------------------------------------


def test_all_failing_reasons_are_reported_not_just_the_first():
    """Inspection must explain every reason, not stop at the first."""
    model = make_model(
        data_handling_tier=DataHandlingTier.PUBLIC_ONLY,
        quality_tier=Quality.ECONOMY,
        capabilities=frozenset(),
    )
    result = run(
        [model],
        quality_floor=Quality.CRITICAL,
        features=make_features(
            privacy=Privacy.CONFIDENTIAL,
            required_capabilities=frozenset({Capability.TOOLS}),
        ),
    )
    found = set(reasons_for(result, "m/test"))
    assert {
        ExclusionReason.PRIVACY_MISMATCH,
        ExclusionReason.BELOW_QUALITY_FLOOR,
        ExclusionReason.MISSING_CAPABILITY,
    } <= found


def test_eligible_and_excluded_partition_the_snapshot():
    models = [make_model(model_id=f"m/{i}") for i in range(5)]
    models[0] = make_model(model_id="m/0", data_handling_tier=DataHandlingTier.PUBLIC_ONLY)
    result = run(models)
    assert len(result.eligible) + len(result.excluded) == 5


# --- cost estimation -------------------------------------------------------


def test_cost_is_exact_decimal():
    model = make_model(input_per_1k=Decimal("0.001"), output_per_1k=Decimal("0.002"))
    cost = estimate_cost(model, input_tokens=1000, output_tokens=1000)
    assert cost == Decimal("0.003000000")
    assert isinstance(cost, Decimal)


def test_cost_includes_the_request_fee():
    model = make_model(request_fee=Decimal("0.000100000"))
    cost = estimate_cost(model, input_tokens=0, output_tokens=0)
    assert cost == Decimal("0.000100000")


@pytest.mark.parametrize("tokens", [0, 1, 999, 1000, 1_000_000])
def test_cost_never_uses_float_arithmetic(tokens):
    model = make_model()
    cost = estimate_cost(model, input_tokens=tokens, output_tokens=tokens)
    assert isinstance(cost, Decimal)
    assert cost.as_tuple().exponent == -9
