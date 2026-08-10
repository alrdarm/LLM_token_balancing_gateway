"""SDK smoke tests against the real ``openai`` client (§13 M2 exit gate).

The compatibility claim is that unmodified OpenAI SDK code works against this
gateway. Hand-rolled HTTP assertions cannot demonstrate that: they test what we
believe the SDK sends, not what it actually sends, and they never exercise its
response parsing or error mapping at all.

No credentials and no spend are involved -- the client is pointed at our own
in-process app through a transport shim, so nothing leaves the test process.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
)

from tests.conftest import TEST_API_KEY

pytestmark = pytest.mark.contract


def _sdk_for(api_app, api_key: str) -> Iterator[OpenAI]:
    """Bind an ``openai`` client to the in-process gateway.

    ``TestClient`` is used as the SDK's HTTP transport because it is itself an
    ``httpx.Client`` *and* it runs the app's lifespan. A bare ``ASGITransport``
    skips lifespan, so the database engine would never be created.
    """
    with TestClient(api_app, raise_server_exceptions=False) as http_client:
        yield OpenAI(
            api_key=api_key,
            base_url=f"{http_client.base_url}/v1",
            http_client=http_client,
            max_retries=0,
        )


@pytest.fixture
def sdk(api_app) -> Iterator[OpenAI]:
    """An authenticated SDK client pointed at the gateway."""
    yield from _sdk_for(api_app, TEST_API_KEY)


def test_sdk_lists_models(sdk: OpenAI):
    """``client.models.list()`` must parse into the SDK's own model objects."""
    page = sdk.models.list()
    ids = {model.id for model in page.data}

    assert "auto" in ids
    assert "fake/general" in ids
    for model in page.data:
        assert model.object == "model"
        assert model.owned_by


def test_sdk_retrieves_nothing_it_was_not_given(sdk: OpenAI):
    """An unknown model surfaces as the SDK's NotFoundError, not a raw 404."""
    with pytest.raises(NotFoundError) as excinfo:
        sdk.chat.completions.create(
            model="nobody/nothing",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert excinfo.value.status_code == 404


def test_sdk_chat_completion_reaches_the_gateway(sdk: OpenAI):
    """The SDK's request shape is accepted and normalized.

    M2 has no provider, so the call ends in 503. What matters here is that it
    got that far through the SDK's own serialisation, and that the SDK could
    parse our error envelope.
    """
    with pytest.raises(APIStatusError) as excinfo:
        sdk.chat.completions.create(
            model="auto",
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Return JSON with a title."},
            ],
            max_tokens=300,
            temperature=0.2,
        )

    error = excinfo.value
    assert error.status_code == 503
    # The SDK parsed our envelope rather than treating it as opaque text.
    assert error.body["code"] == "no_provider_available"
    assert "adapter" in error.message


def test_sdk_responses_endpoint_reaches_the_gateway(sdk: OpenAI):
    with pytest.raises(APIStatusError) as excinfo:
        sdk.responses.create(
            model="auto-quality",
            instructions="Be concise.",
            input="Review this SQL.",
            max_output_tokens=1200,
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.body["code"] == "no_provider_available"


def test_sdk_maps_authentication_failure(api_app):
    for client in _sdk_for(api_app, "sk-not-a-real-key"):
        with pytest.raises(AuthenticationError) as excinfo:
            client.models.list()
        assert excinfo.value.status_code == 401


def test_sdk_maps_validation_failure(sdk: OpenAI):
    """A rejected control must reach the SDK as BadRequestError, not a 422."""
    with pytest.raises(BadRequestError) as excinfo:
        sdk.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            n=3,
        )
    assert excinfo.value.status_code == 400


def test_sdk_passes_gateway_controls_through_extra_body(sdk: OpenAI):
    """Clients that cannot extend the body use headers; SDKs use extra_body.

    Proves the control block survives the SDK's serialisation intact -- if it
    did not, the request would fail as an unknown-field error instead of 503.
    """
    with pytest.raises(APIStatusError) as excinfo:
        sdk.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={
                "gateway": {
                    "quality": "high",
                    "privacy": "confidential",
                    "max_cost": "0.050000000",
                    "max_latency_ms": 12000,
                }
            },
        )
    assert excinfo.value.status_code == 503


def test_sdk_rejects_a_bad_gateway_control(sdk: OpenAI):
    with pytest.raises(BadRequestError) as excinfo:
        sdk.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"gateway": {"quality": "supreme"}},
        )
    assert excinfo.value.body["code"] == "invalid_gateway_control"


def test_sdk_control_headers_are_honoured(sdk: OpenAI):
    """``X-LLM-*`` headers work for clients that cannot extend the body (§2)."""
    with pytest.raises(APIStatusError) as excinfo:
        sdk.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            extra_headers={"X-LLM-Max-Cost": "not-a-number"},
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.body["code"] == "invalid_gateway_control"


def test_sdk_receives_the_request_id_header(sdk: OpenAI):
    """SDK users correlate support requests by this header."""
    with pytest.raises(APIStatusError) as excinfo:
        sdk.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    assert excinfo.value.response.headers["x-llm-request-id"]
