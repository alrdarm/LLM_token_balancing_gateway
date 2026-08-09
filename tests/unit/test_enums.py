"""Domain value sets and ordering."""

from __future__ import annotations

import pytest

from gateway.domain.enums import (
    PRIVACY_ELIGIBLE_TIERS,
    PRIVACY_ORDER,
    QUALITY_ORDER,
    RISK_ORDER,
    TERMINAL_STATES,
    VALIDATION_SEVERITY,
    DataHandlingTier,
    Privacy,
    Quality,
    RequestState,
    Risk,
    TaskClass,
    ValidationResult,
    is_terminal,
)


def test_every_terminal_state_is_a_request_state():
    assert TERMINAL_STATES <= set(RequestState)


def test_terminal_states_match_the_spec():
    assert {state.value for state in TERMINAL_STATES} == {
        "SUCCEEDED",
        "REJECTED",
        "REJECTED_NO_ROUTE",
        "REJECTED_BUDGET",
        "FAILED",
        "FAILED_PARTIAL",
        "FAILED_EXHAUSTED",
        "CANCELLED",
        "EXPIRED",
    }


@pytest.mark.parametrize(
    "state",
    [RequestState.RECEIVED, RequestState.INVOKING, RequestState.VALIDATING],
)
def test_working_states_are_not_terminal(state):
    assert not is_terminal(state)


def test_validation_severity_covers_every_result_exactly_once():
    assert set(VALIDATION_SEVERITY) == set(ValidationResult)
    assert len(VALIDATION_SEVERITY) == len(ValidationResult)


def test_validation_severity_order_matches_the_spec():
    """FAIL_GROUNDING > FAIL_CAPABILITY > FAIL_QUALITY > FAIL_REPAIRABLE
    > INDETERMINATE > PASS."""
    assert [result.value for result in VALIDATION_SEVERITY] == [
        "FAIL_GROUNDING",
        "FAIL_CAPABILITY",
        "FAIL_QUALITY",
        "FAIL_REPAIRABLE",
        "INDETERMINATE",
        "PASS",
    ]


@pytest.mark.parametrize(
    ("order", "enum_cls"),
    [(QUALITY_ORDER, Quality), (PRIVACY_ORDER, Privacy), (RISK_ORDER, Risk)],
)
def test_orderings_are_total(order, enum_cls):
    assert set(order) == set(enum_cls)
    assert len(order) == len(enum_cls)


def test_confidential_excludes_public_only_models():
    """§11: confidential requests may not reach public-data handling tiers."""
    assert DataHandlingTier.PUBLIC_ONLY not in PRIVACY_ELIGIBLE_TIERS[Privacy.CONFIDENTIAL]


def test_deployment_strict_allows_only_zdr():
    assert PRIVACY_ELIGIBLE_TIERS[Privacy.DEPLOYMENT_STRICT] == {DataHandlingTier.ZDR}


def test_privacy_eligibility_narrows_monotonically():
    """A stricter privacy level can never permit a tier a looser one forbids."""
    public = PRIVACY_ELIGIBLE_TIERS[Privacy.PUBLIC]
    confidential = PRIVACY_ELIGIBLE_TIERS[Privacy.CONFIDENTIAL]
    strict = PRIVACY_ELIGIBLE_TIERS[Privacy.DEPLOYMENT_STRICT]
    assert strict <= confidential <= public


def test_spec_task_classes_are_present():
    """§10 lists the initial classes; §14 requires at least ten to be routable."""
    assert len(TaskClass) >= 10
    for expected in ("classification", "extraction", "sql_review", "grounded_qa"):
        assert expected in {task.value for task in TaskClass}
