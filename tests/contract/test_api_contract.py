"""Golden contracts for the public API surface (§1, §3, §9).

These pin the wire behaviour a client depends on: status codes, the error
envelope, and what is and is not disclosed.
"""

from __future__ import annotations

import pytest

from tests.conftest import TEST_DEBUG_API_KEY

CHAT_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"

MINIMAL_CHAT = {"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}
MINIMAL_RESPONSES = {"model": "auto", "input": "Hello"}


def envelope_is_wellformed(body: dict) -> bool:
    """Every error must match the §9 shape exactly."""
    return (
        set(body) == {"error", "gateway"}
        and set(body["error"]) == {"message", "type", "param", "code"}
        and {"request_id", "retryable"} <= set(body["gateway"])
    )


# --- authentication --------------------------------------------------------


@pytest.mark.parametrize("path", [CHAT_PATH, RESPONSES_PATH, "/v1/models"])
def test_endpoints_require_authentication(anonymous_client, path):
    """§1: authenticate all non-health endpoints."""
    response = (
        anonymous_client.get(path)
        if path == "/v1/models"
        else anonymous_client.post(path, json=MINIMAL_CHAT)
    )
    assert response.status_code == 401
    body = response.json()
    assert envelope_is_wellformed(body)
    assert body["error"]["code"] == "invalid_api_key"


@pytest.mark.parametrize(
    "header",
    ["Bearer wrong-key", "Basic abc", "bearer", "", "Bearer "],
)
def test_bad_credentials_are_uniformly_401(anonymous_client, header):
    """Distinct messages would let an attacker enumerate valid keys."""
    response = anonymous_client.post(
        CHAT_PATH, json=MINIMAL_CHAT, headers={"Authorization": header}
    )
    assert response.status_code == 401


def test_valid_key_authenticates(api_client):
    assert api_client.get("/v1/models").status_code == 200


def test_auth_failure_does_not_echo_the_credential(anonymous_client):
    secret = "sk-super-secret-value"
    response = anonymous_client.post(
        CHAT_PATH, json=MINIMAL_CHAT, headers={"Authorization": f"Bearer {secret}"}
    )
    assert secret not in response.text


# --- /v1/models ------------------------------------------------------------


def test_models_lists_selectors_and_enabled_models(api_client):
    body = api_client.get("/v1/models").json()
    assert body["object"] == "list"

    ids = {entry["id"] for entry in body["data"]}
    assert {"auto", "auto-cheap", "auto-fast", "auto-quality", "auto-private"} <= ids
    assert "fake/general" in ids

    for entry in body["data"]:
        assert set(entry) == {"id", "object", "created", "owned_by"}
        assert entry["object"] == "model"


# --- content type and body -------------------------------------------------


def test_non_json_content_type_is_415(api_client):
    response = api_client.post(
        CHAT_PATH, content="model=auto", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_malformed_json_is_400(api_client):
    response = api_client.post(
        CHAT_PATH, content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_malformed_json_error_does_not_echo_the_body(api_client):
    response = api_client.post(
        CHAT_PATH,
        content='{"messages": "secret prompt text',
        headers={"Content-Type": "application/json"},
    )
    assert "secret prompt text" not in response.text


def test_non_object_body_is_400(api_client):
    response = api_client.post(CHAT_PATH, json=[1, 2, 3])
    assert response.status_code == 400


# --- schema validation -----------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected_param"),
    [
        ({"messages": [{"role": "user", "content": "x"}]}, "model"),
        ({"model": "auto"}, "messages"),
        ({"model": "auto", "messages": []}, "messages"),
    ],
)
def test_missing_required_fields_are_400(api_client, body, expected_param):
    response = api_client.post(CHAT_PATH, json=body)
    assert response.status_code == 400
    assert expected_param in (response.json()["error"]["param"] or "")


def test_n_other_than_one_is_rejected(api_client):
    """v0.1 rejects n != 1 (§3)."""
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "n": 2})
    assert response.status_code == 400
    assert "n must be 1" in response.json()["error"]["message"]


def test_mutually_exclusive_token_limits_are_rejected(api_client):
    response = api_client.post(
        CHAT_PATH, json={**MINIMAL_CHAT, "max_tokens": 10, "max_completion_tokens": 10}
    )
    assert response.status_code == 400
    assert "mutually exclusive" in response.json()["error"]["message"]


def test_validation_error_does_not_echo_prompt_text(api_client):
    """Pydantic's default body can quote submitted values; ours must not."""
    secret = "confidential-patient-record-42"
    response = api_client.post(
        CHAT_PATH,
        json={"model": "auto", "messages": [{"role": "user", "content": secret}], "n": 7},
    )
    assert response.status_code == 400
    assert secret not in response.text


# --- gateway controls ------------------------------------------------------


def test_unknown_gateway_control_is_rejected(api_client):
    """The control block is strict: a typo must not silently weaken limits."""
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "gateway": {"quallity": "high"}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_gateway_control"


@pytest.mark.parametrize(
    "controls",
    [
        {"quality": "supreme"},
        {"privacy": "secret"},
        {"max_cost": -1},
        {"max_attempts": 0},
        {"max_attempts": 6},
        {"max_latency_ms": 0},
        {"validation": "sometimes"},
        {"risk": "spicy"},
    ],
)
def test_invalid_gateway_controls_are_400(api_client, controls):
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "gateway": controls})
    assert response.status_code == 400


def test_float_max_cost_is_rejected(api_client):
    """JSON floats cannot represent decimal cents exactly (§2)."""
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "gateway": {"max_cost": 0.05}})
    assert response.status_code == 400
    assert "string" in response.json()["error"]["message"]


def test_string_max_cost_is_accepted(api_client):
    response = api_client.post(
        CHAT_PATH, json={**MINIMAL_CHAT, "gateway": {"max_cost": "0.050000000"}}
    )
    assert response.status_code == 200


def test_contradictory_provider_lists_are_rejected(api_client):
    response = api_client.post(
        CHAT_PATH,
        json={
            **MINIMAL_CHAT,
            "gateway": {"provider_allow": ["a"], "provider_deny": ["a"]},
        },
    )
    assert response.status_code == 400


def test_oversized_metadata_is_rejected(api_client):
    response = api_client.post(
        CHAT_PATH,
        json={**MINIMAL_CHAT, "gateway": {"metadata": {str(i): "x" for i in range(40)}}},
    )
    assert response.status_code == 400


def test_debug_requires_scope(api_client):
    """§2: debug requires scope. Denying beats silently downgrading."""
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "gateway": {"debug": True}})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "policy_denied"


def test_debug_allowed_with_scope(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app, raise_server_exceptions=False) as client:
        response = client.post(
            CHAT_PATH,
            json={**MINIMAL_CHAT, "gateway": {"debug": True}},
            headers={"Authorization": f"Bearer {TEST_DEBUG_API_KEY}"},
        )
    assert response.status_code == 200
    # §1: route and cost detail only for an authorized caller.
    assert response.json()["gateway"]["resolved_model"]


def test_malformed_control_header_is_400(api_client):
    response = api_client.post(CHAT_PATH, json=MINIMAL_CHAT, headers={"X-LLM-Max-Cost": "free"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_gateway_control"


# --- models and routing state ---------------------------------------------


def test_unknown_explicit_model_is_404(api_client):
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "model": "nobody/nothing"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_known_model_passes_the_lookup(api_client):
    response = api_client.post(CHAT_PATH, json={**MINIMAL_CHAT, "model": "fake/general"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "path,body", [(CHAT_PATH, MINIMAL_CHAT), (RESPONSES_PATH, MINIMAL_RESPONSES)]
)
def test_valid_request_generates_a_response(api_client, path, body):
    """Since M5 the pipeline runs end to end against the deterministic fake."""
    response = api_client.post(path, json=body)
    assert response.status_code == 200

    payload = response.json()
    assert payload["gateway"]["request_id"]
    assert payload["usage"]["total_tokens"] > 0


def test_responses_rejects_background_mode(api_client):
    response = api_client.post(RESPONSES_PATH, json={**MINIMAL_RESPONSES, "background": True})
    assert response.status_code == 400


# --- cross-cutting ---------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        (CHAT_PATH, MINIMAL_CHAT, 200),
        (CHAT_PATH, {"model": "auto"}, 400),
        ("/v1/models", None, 200),
    ],
)
def test_every_response_carries_a_request_id(api_client, path, body, expected):
    response = api_client.get(path) if body is None else api_client.post(path, json=body)
    assert response.status_code == expected
    assert response.headers["X-LLM-Request-ID"]


def test_client_request_id_is_echoed(api_client):
    response = api_client.post(
        CHAT_PATH, json=MINIMAL_CHAT, headers={"X-Request-ID": "trace-contract-1"}
    )
    assert response.headers["X-LLM-Request-ID"] == "trace-contract-1"
    assert response.json()["gateway"]["request_id"] == "trace-contract-1"


def test_openapi_document_is_served(api_client):
    document = api_client.get("/openapi.json").json()
    assert CHAT_PATH in document["paths"]
    assert RESPONSES_PATH in document["paths"]
    assert "/v1/models" in document["paths"]
