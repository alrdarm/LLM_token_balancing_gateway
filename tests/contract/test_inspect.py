"""``POST /route/inspect`` contract (§4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from gateway.persistence.models import Attempt, BudgetReservation, Request
from tests.conftest import TEST_DEBUG_API_KEY

PATH = "/route/inspect"

CHAT_BODY = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarise this report."}],
}
SQL_BODY = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Review this SQL: SELECT * FROM users"}],
    "gateway": {"privacy": "confidential", "max_cost": "0.20"},
}
RESPONSES_BODY = {"model": "auto-quality", "input": "Review this SQL query please."}


@pytest.fixture
def debug_client(api_app):
    """A client whose credential carries the ``debug`` scope."""
    with TestClient(api_app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_DEBUG_API_KEY}"})
        yield client


def test_requires_authentication(anonymous_client):
    assert anonymous_client.post(PATH, json=CHAT_BODY).status_code == 401


def test_accepts_a_chat_shaped_payload(api_client):
    response = api_client.post(PATH, json=CHAT_BODY)
    assert response.status_code == 200
    assert response.json()["object"] == "gateway.route_inspection"


def test_accepts_a_responses_shaped_payload(api_client):
    """§4 accepts either wire shape."""
    response = api_client.post(PATH, json=RESPONSES_BODY)
    assert response.status_code == 200
    assert response.json()["object"] == "gateway.route_inspection"


def test_request_id_uses_the_inspect_prefix(api_client):
    body = api_client.post(PATH, json=CHAT_BODY).json()
    assert body["request_id"].startswith("inspect_")


def test_document_carries_every_required_section(api_client):
    body = api_client.post(PATH, json=SQL_BODY).json()
    for key in (
        "classification",
        "effective_controls",
        "policy",
        "candidates",
        "excluded",
        "planned_validation",
        "estimated_route_upper_bound",
        "warnings",
    ):
        assert key in body, f"missing section: {key}"


def test_classification_reports_task_class_and_risk(api_client):
    classification = api_client.post(PATH, json=SQL_BODY).json()["classification"]
    assert classification["task_class"] == "sql_review"
    assert classification["risk"] == "high"
    assert 1 <= classification["complexity"] <= 5
    assert 0.0 <= classification["confidence"] <= 1.0


def test_policy_is_frozen_by_id_and_version(api_client):
    policy = api_client.post(PATH, json=SQL_BODY).json()["policy"]
    assert policy["id"]
    assert isinstance(policy["version"], int)


def test_high_risk_plans_an_independent_review(api_client):
    """§8: high risk requires independent validation."""
    plan = api_client.post(PATH, json=SQL_BODY).json()["planned_validation"]
    assert "independent_review" in plan


def test_confidential_request_excludes_public_only_models(debug_client):
    """T01 in §12, visible through inspection."""
    body = debug_client.post(PATH, json=SQL_BODY).json()

    excluded = {entry["model"]: entry["reasons"] for entry in body["excluded"]}
    assert "fake/flash" in excluded
    assert "privacy_mismatch" in excluded["fake/flash"]

    chosen = {candidate["model"] for candidate in body["candidates"]}
    assert "fake/flash" not in chosen


def test_costs_are_serialised_as_exact_strings(api_client):
    """Money crosses the wire as a string so it cannot become a float."""
    body = api_client.post(PATH, json=CHAT_BODY).json()
    assert isinstance(body["estimated_route_upper_bound"], str)
    for candidate in body["candidates"]:
        assert isinstance(candidate["estimated_cost"], str)
        assert isinstance(candidate["effective_cost"], str)


def test_ranking_is_deterministic_across_calls(api_client):
    """§4: the same input and snapshots must rank identically."""
    first = api_client.post(PATH, json=SQL_BODY).json()
    for _ in range(5):
        again = api_client.post(PATH, json=SQL_BODY).json()
        assert [c["rank"] for c in again["candidates"]] == [c["rank"] for c in first["candidates"]]
        assert again["estimated_route_upper_bound"] == first["estimated_route_upper_bound"]


def test_candidate_ranks_are_dense_and_ordered(api_client):
    candidates = api_client.post(PATH, json=CHAT_BODY).json()["candidates"]
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))


# --- disclosure ------------------------------------------------------------


def test_scores_and_model_ids_require_debug_scope(api_client):
    """§4: disclosure of scores and provider IDs requires debug scope."""
    body = api_client.post(PATH, json=CHAT_BODY).json()
    for candidate in body["candidates"]:
        assert "score" not in candidate
        assert "model" not in candidate
        assert "provider" not in candidate
    for entry in body["excluded"]:
        assert "model" not in entry


def test_debug_scope_discloses_scores_and_models(debug_client):
    body = debug_client.post(PATH, json=CHAT_BODY).json()
    assert body["candidates"], "expected at least one candidate"
    for candidate in body["candidates"]:
        assert "model" in candidate
        assert "provider" in candidate
        assert isinstance(candidate["score"], float)
        assert set(candidate["score_components"]) == {
            "cost",
            "latency",
            "quality_shortfall",
            "failure_risk",
        }
    assert "registry_snapshot_at" in body
    assert "rationale_codes" in body


def test_exclusion_reasons_are_disclosed_without_scope(api_client):
    """Reasons are useful and non-sensitive; the model identity is not."""
    body = api_client.post(PATH, json=SQL_BODY).json()
    assert body["excluded"]
    for entry in body["excluded"]:
        assert entry["reasons"]


# --- no side effects -------------------------------------------------------


def test_inspection_creates_no_rows(api_client, api_settings):
    """§4: inspect reserves no budget, calls no provider, creates no attempts."""
    from gateway.persistence.engine import create_db_engine, create_session_factory

    engine = create_db_engine(api_settings.database_url)
    factory = create_session_factory(engine)

    def counts() -> tuple[int, int, int]:
        with factory() as session:
            return (
                session.scalar(select(func.count()).select_from(Request)) or 0,
                session.scalar(select(func.count()).select_from(Attempt)) or 0,
                session.scalar(select(func.count()).select_from(BudgetReservation)) or 0,
            )

    before = counts()
    for _ in range(3):
        assert api_client.post(PATH, json=SQL_BODY).status_code == 200
    after = counts()
    engine.dispose()

    assert before == after == (0, 0, 0)


# --- validation ------------------------------------------------------------


def test_invalid_control_is_rejected(api_client):
    response = api_client.post(PATH, json={**CHAT_BODY, "gateway": {"quality": "supreme"}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_gateway_control"


def test_unknown_explicit_model_is_404(api_client):
    response = api_client.post(PATH, json={**CHAT_BODY, "model": "nobody/nothing"})
    assert response.status_code in (400, 404)


def test_impossible_constraints_report_no_route(api_client):
    """A valid request with unsatisfiable constraints is advisory, not an error."""
    response = api_client.post(
        PATH,
        json={
            **CHAT_BODY,
            "gateway": {"max_cost": "0.000000001", "privacy": "confidential"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["candidates"] == []
    assert "no_eligible_route" in body["warnings"]


def test_inspect_and_models_agree_on_available_models(debug_client):
    """Inspection must not offer a model /v1/models does not list."""
    listed = {entry["id"] for entry in debug_client.get("/v1/models").json()["data"]}
    body = debug_client.post(PATH, json=CHAT_BODY).json()
    for candidate in body["candidates"]:
        assert candidate["model"] in listed
