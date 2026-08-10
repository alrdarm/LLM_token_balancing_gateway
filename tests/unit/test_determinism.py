"""Determinism suite: ranking must be reproducible (§4).

"Same canonical input and snapshots must rank deterministically; tie-break by
score, predicted cost, model ID." Inspection is only useful if executing the
same request would pick the same route, so these are contract tests, not
nice-to-haves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from gateway.domain.enums import Quality
from gateway.domain.routing import Eligibility, RegistrySnapshot
from gateway.services.router import (
    DEFAULT_WEIGHTS,
    SELECTOR_WEIGHTS,
    expected_total_cost,
    plan,
    rank,
    route_upper_bound,
    weights_for,
)
from tests.unit.test_eligibility import make_features, make_model, make_request

SNAPSHOT_AT = datetime(2026, 6, 1, tzinfo=UTC)


def rank_models(models, *, weights=None, max_attempts=3, request=None, features=None):
    eligibility = Eligibility(eligible=tuple(models), excluded=())
    return rank(
        eligibility,
        features or make_features(),
        request or make_request(),
        weights=weights or dict(DEFAULT_WEIGHTS),
        max_attempts=max_attempts,
    )


def test_ranking_is_stable_across_repeated_runs():
    models = [make_model(model_id=f"m/{i}", latency_prior_ms=1000 + i) for i in range(6)]
    first = [candidate.model_id for candidate in rank_models(models)]
    for _ in range(20):
        assert [c.model_id for c in rank_models(models)] == first


def test_ranking_is_independent_of_input_order():
    """The same registry in a different order must rank identically."""
    models = [
        make_model(model_id="m/a", latency_prior_ms=1000),
        make_model(model_id="m/b", latency_prior_ms=2000),
        make_model(model_id="m/c", latency_prior_ms=3000),
    ]
    forward = [c.model_id for c in rank_models(models)]
    backward = [c.model_id for c in rank_models(list(reversed(models)))]
    assert forward == backward


def test_identical_models_tie_break_on_model_id():
    """Fully identical models must still produce a stable, explainable order."""
    models = [make_model(model_id=name) for name in ("m/zebra", "m/alpha", "m/middle")]
    ordered = [candidate.model_id for candidate in rank_models(models)]
    assert ordered == ["m/alpha", "m/middle", "m/zebra"]


def test_score_ties_break_on_exact_cost_before_id():
    """Cost is compared as Decimal, so ties never hinge on float equality."""
    cheap = make_model(model_id="m/zzz", input_per_1k=Decimal("0.001"))
    dear = make_model(model_id="m/aaa", input_per_1k=Decimal("0.002"))
    ordered = [candidate.model_id for candidate in rank_models([cheap, dear])]
    assert ordered[0] == "m/zzz"


def test_ranks_are_dense_and_one_based():
    models = [make_model(model_id=f"m/{i}") for i in range(4)]
    candidates = rank_models(models)
    assert [candidate.rank for candidate in candidates] == [1, 2, 3, 4]


def test_empty_eligibility_ranks_to_nothing():
    assert rank_models([]) == ()


# --- expected cost ---------------------------------------------------------


def test_expected_cost_exceeds_single_attempt_when_pass_rate_is_low():
    """A model that often fails costs more than its sticker price."""
    unreliable = make_model(pass_rate_prior=0.5)
    single = Decimal("0.010000000")
    expected = expected_total_cost(unreliable, single_attempt_cost=single, max_attempts=3)
    assert expected > single


def test_perfect_pass_rate_costs_exactly_one_attempt():
    reliable = make_model(pass_rate_prior=1.0)
    single = Decimal("0.010000000")
    assert expected_total_cost(reliable, single_attempt_cost=single, max_attempts=3) == single


def test_expected_cost_is_decimal():
    value = expected_total_cost(
        make_model(pass_rate_prior=0.7), single_attempt_cost=Decimal("0.01"), max_attempts=3
    )
    assert isinstance(value, Decimal)


def test_a_cheap_unreliable_model_can_lose_to_a_dearer_reliable_one():
    """The point of expected-cost ranking (§10)."""
    cheap_flaky = make_model(
        model_id="m/flaky",
        input_per_1k=Decimal("0.0005"),
        output_per_1k=Decimal("0.0005"),
        pass_rate_prior=0.30,
        failure_rate_prior=0.20,
    )
    dearer_solid = make_model(
        model_id="m/solid",
        input_per_1k=Decimal("0.0010"),
        output_per_1k=Decimal("0.0010"),
        pass_rate_prior=0.95,
        failure_rate_prior=0.01,
    )
    ordered = [c.model_id for c in rank_models([cheap_flaky, dearer_solid])]
    assert ordered[0] == "m/solid"


# --- selector weighting ----------------------------------------------------


@pytest.mark.parametrize("selector", ["auto-cheap", "auto-fast", "auto-quality"])
def test_selectors_use_their_own_weights(selector):
    assert weights_for(selector, {"cost": 0.99}) == SELECTOR_WEIGHTS[selector]


def test_auto_defers_to_policy_weights():
    weights = weights_for("auto", {"cost": 0.9})
    assert weights["cost"] == 0.9


def test_auto_cheap_prefers_the_cheapest_model():
    cheap = make_model(model_id="m/cheap", input_per_1k=Decimal("0.0001"), latency_prior_ms=9000)
    fast = make_model(model_id="m/fast", input_per_1k=Decimal("0.05"), latency_prior_ms=200)
    ordered = [
        c.model_id for c in rank_models([cheap, fast], weights=SELECTOR_WEIGHTS["auto-cheap"])
    ]
    assert ordered[0] == "m/cheap"


def test_auto_fast_prefers_the_lowest_latency():
    cheap = make_model(model_id="m/cheap", input_per_1k=Decimal("0.0001"), latency_prior_ms=9000)
    fast = make_model(model_id="m/fast", input_per_1k=Decimal("0.05"), latency_prior_ms=200)
    ordered = [
        c.model_id for c in rank_models([cheap, fast], weights=SELECTOR_WEIGHTS["auto-fast"])
    ]
    assert ordered[0] == "m/fast"


def test_auto_quality_prefers_the_highest_pass_rate():
    weak = make_model(model_id="m/weak", pass_rate_prior=0.5, input_per_1k=Decimal("0.00001"))
    strong = make_model(model_id="m/strong", pass_rate_prior=0.97, input_per_1k=Decimal("0.02"))
    ordered = [
        c.model_id for c in rank_models([weak, strong], weights=SELECTOR_WEIGHTS["auto-quality"])
    ]
    assert ordered[0] == "m/strong"


# --- plan ------------------------------------------------------------------


def build_plan_for(models, request=None, **kwargs: object):
    snapshot = RegistrySnapshot(taken_at=SNAPSHOT_AT, models=tuple(models))
    eligibility = Eligibility(eligible=tuple(models), excluded=())
    return plan(
        request or make_request(),
        make_features(),
        snapshot,
        eligibility,
        policy_id=kwargs.pop("policy_id", "default-v1"),
        policy_version=kwargs.pop("policy_version", 1),
        policy_weights=kwargs.pop("policy_weights", None),
        validation_plan=kwargs.pop("validation_plan", ("schema_check",)),
        max_generation_attempts=kwargs.pop("max_generation_attempts", 3),
        max_same_model_repairs=kwargs.pop("max_same_model_repairs", 1),
        now=SNAPSHOT_AT,
    )


def test_plan_freezes_policy_and_snapshot_time():
    """§10: every route freezes policy version and snapshot time."""
    result = build_plan_for([make_model()], policy_id="sql-v3", policy_version=3)
    assert result.policy_id == "sql-v3"
    assert result.policy_version == 3
    assert result.registry_snapshot_at == SNAPSHOT_AT


def test_plan_is_reproducible():
    models = [make_model(model_id=f"m/{i}") for i in range(4)]
    first, second = build_plan_for(models), build_plan_for(models)
    assert [c.model_id for c in first.candidates] == [c.model_id for c in second.candidates]
    assert first.estimated_route_upper_bound == second.estimated_route_upper_bound


def test_attempt_cap_is_the_stricter_of_policy_and_caller():
    result = build_plan_for(
        [make_model()], request=make_request(max_attempts=2), max_generation_attempts=5
    )
    assert result.max_generation_attempts == 2


def test_explicit_model_without_fallback_yields_one_candidate():
    """§1: fallback off an explicit ID requires allow_fallback=true."""
    models = [make_model(model_id="m/a"), make_model(model_id="m/b")]
    request = make_request(allow_fallback=False)
    from dataclasses import replace

    request = replace(request, requested_model="m/a")
    result = build_plan_for(models, request=request)
    assert len(result.candidates) == 1
    assert result.candidates[0].model_id == "m/a"
    assert "fallback_disabled" in result.rationale_codes


def test_selector_request_is_marked_as_such():
    result = build_plan_for([make_model()])
    assert "selector_applied" in result.rationale_codes


def test_no_candidates_is_reported_in_the_rationale():
    result = build_plan_for([])
    assert not result.has_route
    assert "no_eligible_candidates" in result.rationale_codes


def test_upper_bound_covers_every_allowed_attempt():
    """§4's advisory ceiling must be an upper bound, not an average."""
    models = [
        make_model(model_id="m/a", input_per_1k=Decimal("0.001")),
        make_model(model_id="m/b", input_per_1k=Decimal("0.010")),
    ]
    candidates = rank_models(models)
    bound = route_upper_bound(candidates, 3)
    assert bound >= max(c.estimated_cost for c in candidates)
    assert isinstance(bound, Decimal)


def test_upper_bound_of_no_candidates_is_zero():
    assert route_upper_bound((), 3) == Decimal("0")


def test_quality_floor_helper_raises_for_risk():
    from gateway.domain.enums import Risk
    from gateway.services.classifier import quality_floor_for

    assert quality_floor_for(Risk.CRITICAL, Quality.ECONOMY) is Quality.CRITICAL
    assert quality_floor_for(Risk.LOW, Quality.HIGH) is Quality.HIGH
