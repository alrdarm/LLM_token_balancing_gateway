"""Control precedence resolution (§2).

The most safety-critical logic in M2: getting precedence backwards would let a
caller weaken a deployment's privacy floor or cost ceiling by sending a header.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from gateway.domain.enums import Capability, Privacy, Quality, Risk
from gateway.domain.errors import InvalidGatewayControlError
from gateway.domain.requests import GatewayControls
from gateway.services.controls import (
    ControlLayer,
    layer_from_headers,
    resolve_controls,
)

DEFAULTS = GatewayControls()


def resolve(**layers: ControlLayer) -> GatewayControls:
    return resolve_controls(
        deployment=layers.get("deployment", ControlLayer()),
        client=layers.get("client", ControlLayer()),
        headers=layers.get("headers", ControlLayer()),
        body=layers.get("body", ControlLayer()),
        alias_defaults=layers.get("alias_defaults", ControlLayer()),
        system_defaults=layers.get("system_defaults", DEFAULTS),
        requested_model=layers.get("requested_model", "auto"),  # type: ignore[arg-type]
    )


# --- strictest-wins dimensions --------------------------------------------


def test_privacy_defaults_to_confidential():
    """A caller who says nothing must not reach public-data tiers."""
    assert resolve().privacy is Privacy.CONFIDENTIAL


def test_body_cannot_loosen_the_deployment_privacy_floor():
    resolved = resolve(
        deployment=ControlLayer(privacy=Privacy.DEPLOYMENT_STRICT),
        body=ControlLayer(privacy=Privacy.PUBLIC),
    )
    assert resolved.privacy is Privacy.DEPLOYMENT_STRICT


def test_header_cannot_loosen_the_deployment_privacy_floor():
    """Headers outrank the body but are still bound by strictest-wins."""
    resolved = resolve(
        deployment=ControlLayer(privacy=Privacy.CONFIDENTIAL),
        headers=ControlLayer(privacy=Privacy.PUBLIC),
    )
    assert resolved.privacy is Privacy.CONFIDENTIAL


def test_a_lower_layer_may_tighten_privacy():
    """Strictest wins in both directions: the body can exceed the floor."""
    resolved = resolve(
        deployment=ControlLayer(privacy=Privacy.PUBLIC),
        body=ControlLayer(privacy=Privacy.CONFIDENTIAL),
    )
    assert resolved.privacy is Privacy.CONFIDENTIAL


def test_auto_private_selector_forces_confidential():
    resolved = resolve(
        body=ControlLayer(privacy=Privacy.PUBLIC),
        requested_model="auto-private",  # type: ignore[arg-type]
    )
    assert resolved.privacy is Privacy.CONFIDENTIAL


def test_quality_takes_the_strictest_floor():
    resolved = resolve(
        client=ControlLayer(quality=Quality.HIGH),
        body=ControlLayer(quality=Quality.ECONOMY),
    )
    assert resolved.quality is Quality.HIGH


def test_risk_may_be_raised_but_never_lowered():
    """§2: a caller may raise inferred risk, never lower it."""
    resolved = resolve(
        client=ControlLayer(risk=Risk.HIGH),
        body=ControlLayer(risk=Risk.LOW),
    )
    assert resolved.risk is Risk.HIGH


def test_risk_is_none_when_nothing_sets_it():
    assert resolve().risk is None


# --- highest-precedence-wins dimensions -----------------------------------


def test_cost_ceiling_takes_the_lowest_value_not_the_top_layer():
    """A ceiling is a limit, so the strictest wins wherever it came from.

    Regression: a deployment *default* placed in the top layer used to win
    outright, so a caller could never set a tighter ceiling than the default.
    """
    resolved = resolve(
        headers=ControlLayer(max_cost=Decimal("5.00")),
        body=ControlLayer(max_cost=Decimal("0.10")),
    )
    assert resolved.max_cost == Decimal("0.10")


def test_caller_cannot_raise_a_ceiling_set_higher_up():
    resolved = resolve(
        client=ControlLayer(max_cost=Decimal("0.01")),
        body=ControlLayer(max_cost=Decimal("2.00")),
    )
    assert resolved.max_cost == Decimal("0.01")


def test_system_default_cost_ceiling_still_applies():
    resolved = resolve(
        system_defaults=GatewayControls(max_cost=Decimal("1.00")),
        body=ControlLayer(max_cost=Decimal("50.00")),
    )
    assert resolved.max_cost == Decimal("1.00")


def test_latency_ceiling_takes_the_lowest_value():
    resolved = resolve(
        deployment=ControlLayer(max_latency_ms=1000),
        client=ControlLayer(max_latency_ms=5000),
        headers=ControlLayer(max_latency_ms=9000),
    )
    assert resolved.max_latency_ms == 1000


# --- caps and lists --------------------------------------------------------


def test_max_attempts_takes_the_lowest_cap():
    """A caller must not be able to raise a policy ceiling."""
    resolved = resolve(
        client=ControlLayer(max_attempts=2),
        body=ControlLayer(max_attempts=5),
    )
    assert resolved.max_attempts == 2


def test_provider_allow_lists_intersect():
    """No layer may widen access a stricter layer granted."""
    resolved = resolve(
        client=ControlLayer(provider_allow=("a", "b")),
        body=ControlLayer(provider_allow=("b", "c")),
    )
    assert resolved.provider_allow == ("b",)


def test_provider_deny_lists_union():
    resolved = resolve(
        client=ControlLayer(provider_deny=("a",)),
        body=ControlLayer(provider_deny=("b",)),
    )
    assert set(resolved.provider_deny) == {"a", "b"}


def test_deny_beats_allow():
    resolved = resolve(body=ControlLayer(provider_allow=("a", "b"), provider_deny=("b",)))
    assert resolved.provider_allowed("a")
    assert not resolved.provider_allowed("b")


def test_unlisted_provider_is_excluded_when_allow_is_set():
    resolved = resolve(body=ControlLayer(provider_allow=("a",)))
    assert not resolved.provider_allowed("z")


def test_no_allow_list_permits_any_undenied_provider():
    assert resolve().provider_allowed("anything")


def test_required_capabilities_union_across_layers():
    resolved = resolve(
        client=ControlLayer(required_capabilities=frozenset({Capability.TOOLS})),
        body=ControlLayer(required_capabilities=frozenset({Capability.VISION})),
    )
    assert resolved.required_capabilities == {Capability.TOOLS, Capability.VISION}


# --- header parsing --------------------------------------------------------


def test_headers_are_parsed_case_insensitively():
    layer = layer_from_headers({"X-LLM-Quality": "high", "x-llm-privacy": "confidential"})
    assert layer.quality is Quality.HIGH
    assert layer.privacy is Privacy.CONFIDENTIAL


def test_header_money_is_exact():
    layer = layer_from_headers({"X-LLM-Max-Cost": "0.050000000"})
    assert layer.max_cost == Decimal("0.050000000")
    assert isinstance(layer.max_cost, Decimal)


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("X-LLM-Quality", "cheap"),
        ("X-LLM-Privacy", "secret"),
        ("X-LLM-Max-Cost", "not-a-number"),
        ("X-LLM-Max-Cost", "-1"),
        ("X-LLM-Max-Latency", "soon"),
        ("X-LLM-Max-Latency", "0"),
        ("X-LLM-Risk", "spicy"),
        ("X-LLM-Validation", "maybe"),
    ],
)
def test_malformed_headers_are_rejected_not_ignored(header, value):
    """Silently dropping a control would apply weaker limits than requested."""
    with pytest.raises(InvalidGatewayControlError):
        layer_from_headers({header: value})


def test_deployment_strict_is_not_caller_selectable():
    with pytest.raises(InvalidGatewayControlError):
        layer_from_headers({"X-LLM-Privacy": "deployment_strict"})


def test_absent_headers_produce_an_empty_layer():
    layer = layer_from_headers({"Authorization": "Bearer x"})
    assert layer.quality is None
    assert layer.privacy is None
    assert layer.max_cost is None
