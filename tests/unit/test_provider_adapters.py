"""Provider adapter contract and failure classification (§8, §11).

Every adapter is held to the same contract, including the deliberately alien
one — that is what proves the abstraction is not quietly OpenAI-shaped.
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from gateway.domain.enums import AttemptOutcome, Endpoint
from gateway.domain.requests import CanonicalRequest, GatewayControls, Message
from gateway.providers.base import (
    AdapterRegistry,
    ProviderAdapter,
    ProviderFailure,
    ProviderInvocation,
)
from gateway.providers.fake import (
    AlienProvider,
    FakeProvider,
    ScriptedBehaviour,
    auth_error,
    partial_stream_failure,
    rate_limited,
    transient_server_error,
)
from gateway.providers.openai_http import (
    OpenAICompatibleAdapter,
    OutboundNotAllowedError,
    _classify,
)


def make_invocation(**overrides: object) -> ProviderInvocation:
    request = CanonicalRequest(
        request_id="req_1",
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model="auto",
        conversation=(Message(role="user", content="hello"),),
        controls=GatewayControls(),
        client_id="c",
        system_instructions="Be brief.",
        estimated_input_tokens=10,
        input_hash="a" * 64,
    )
    defaults: dict[str, object] = {
        "request": request,
        "provider_model_id": "test-model",
        "gateway_model_id": "fake/general",
        "max_output_tokens": 100,
        "timeout_seconds": 5.0,
        "estimated_cost": Decimal("0.001"),
    }
    defaults.update(overrides)
    return ProviderInvocation(**defaults)  # type: ignore[arg-type]


# --- the shared contract ---------------------------------------------------


@pytest.fixture(params=["fake", "alien"])
def adapter(request: pytest.FixtureRequest) -> ProviderAdapter:
    return FakeProvider() if request.param == "fake" else AlienProvider()


async def test_every_adapter_satisfies_the_protocol(adapter):
    assert isinstance(adapter, ProviderAdapter)
    assert adapter.name


async def test_every_adapter_returns_a_normalized_result(adapter):
    """Whatever the native shape, the result must be gateway-shaped."""
    result = await adapter.generate(make_invocation())

    assert isinstance(result.text, str) and result.text
    assert result.finish_reason in {"stop", "length", "content_filter", "tool_calls"}
    assert result.usage.prompt_tokens is not None
    assert result.usage.completion_tokens is not None
    assert result.model_id == "fake/general"


async def test_every_adapter_is_deterministic(adapter):
    """The same request must produce the same output."""
    first = await adapter.generate(make_invocation())
    second = await adapter.generate(make_invocation())
    assert first.text == second.text


async def test_adapters_do_not_retry_internally(adapter):
    """Retry belongs to orchestration; an adapter retrying spends unreserved money."""
    adapter.behaviour = ScriptedBehaviour().fail_next(transient_server_error())

    with pytest.raises(ProviderFailure):
        await adapter.generate(make_invocation())

    assert len(adapter.calls) == 1


# --- the alien adapter specifically ----------------------------------------


async def test_alien_native_payload_is_not_openai_shaped():
    """Guards the guard: if this ever looks like OpenAI, it stops being useful."""
    alien = AlienProvider()
    payload = alien._native_payload(make_invocation())

    assert "choices" not in payload
    assert "usage" not in payload
    assert payload["output"]["segments"][0]["kind"] == "text"
    assert set(payload["meter"]) == {"tokens_in", "tokens_out"}


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("COMPLETE", "stop"),
        ("TRUNCATED", "length"),
        ("BLOCKED", "content_filter"),
        ("TOOL_REQUESTED", "tool_calls"),
    ],
)
async def test_alien_finish_reasons_are_translated(native, expected):
    alien = AlienProvider(native_state=native)
    result = await alien.generate(make_invocation())
    assert result.finish_reason == expected


async def test_alien_usage_fields_are_translated():
    result = await AlienProvider().generate(make_invocation())
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens >= 1


async def test_alien_streaming_is_rejected_as_a_capability_failure():
    alien = AlienProvider()
    with pytest.raises(ProviderFailure) as excinfo:
        async for _ in alien.stream(make_invocation()):
            pass
    assert excinfo.value.outcome is AttemptOutcome.CAPABILITY_REJECTED
    assert not excinfo.value.retryable


async def test_alien_declares_narrower_capabilities():
    assert not AlienProvider().capabilities().supports_streaming
    assert FakeProvider().capabilities().supports_streaming


# --- fake provider behaviour ----------------------------------------------


async def test_scripted_failure_then_success():
    fake = FakeProvider(behaviour=ScriptedBehaviour().fail_next(rate_limited()).succeed_next())

    with pytest.raises(ProviderFailure):
        await fake.generate(make_invocation())

    result = await fake.generate(make_invocation())
    assert result.text
    assert len(fake.calls) == 2


async def test_streaming_yields_deltas_then_a_finish():
    chunks = [chunk async for chunk in FakeProvider().stream(make_invocation())]

    assert any(chunk.delta for chunk in chunks)
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None


# --- failure classification (§8) ------------------------------------------


@pytest.mark.parametrize(
    ("failure", "outcome", "retryable"),
    [
        (rate_limited(), AttemptOutcome.RATE_LIMITED, True),
        (transient_server_error(), AttemptOutcome.PROVIDER_ERROR, True),
        (auth_error(), AttemptOutcome.AUTH_ERROR, False),
    ],
)
async def test_failures_carry_their_classification(failure, outcome, retryable):
    assert failure.outcome is outcome
    assert failure.retryable is retryable


async def test_partial_stream_failure_is_terminal_and_billable():
    """§4: past visible output, failure is terminal — but usage still bills."""
    failure = partial_stream_failure(prompt_tokens=50, completion_tokens=20)

    assert failure.emitted_output
    assert not failure.retryable
    assert failure.usage.prompt_tokens == 50
    assert failure.usage.completion_tokens == 20


# --- OpenAI-compatible HTTP adapter ---------------------------------------


@pytest.mark.parametrize(
    ("status", "outcome", "retryable"),
    [
        (401, AttemptOutcome.AUTH_ERROR, False),
        (403, AttemptOutcome.AUTH_ERROR, False),
        (413, AttemptOutcome.CONTEXT_REJECTED, False),
        (429, AttemptOutcome.RATE_LIMITED, True),
        (500, AttemptOutcome.PROVIDER_ERROR, True),
        (503, AttemptOutcome.PROVIDER_ERROR, True),
        (418, AttemptOutcome.PROVIDER_ERROR, False),
    ],
)
def test_http_status_classification(status, outcome, retryable):
    failure = _classify(status, None)
    assert failure.outcome is outcome
    assert failure.retryable is retryable


def test_retry_after_header_is_honoured():
    assert _classify(429, "2.5").retry_after_seconds == 2.5


def test_unparseable_retry_after_is_ignored_not_fatal():
    assert _classify(429, "Wed, 21 Oct 2026 07:28:00 GMT").retry_after_seconds is None


def test_outbound_allowlist_blocks_unknown_hosts():
    """§11: allowlist outbound destinations."""
    with pytest.raises(OutboundNotAllowedError):
        OpenAICompatibleAdapter(
            base_url="https://evil.example.com/v1",
            api_key="k",
            allowed_hosts=frozenset({"api.openai.com"}),
        )


def test_outbound_allowlist_permits_listed_hosts():
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        api_key="k",
        allowed_hosts=frozenset({"api.openai.com"}),
    )
    assert adapter.base_url == "https://api.openai.com/v1"


def test_plaintext_http_is_rejected_for_remote_hosts():
    """TLS verification is required; plaintext would expose prompts in transit."""
    with pytest.raises(OutboundNotAllowedError):
        OpenAICompatibleAdapter(base_url="http://remote.example.com/v1", api_key="k")


def test_plaintext_http_is_allowed_for_localhost():
    """Local fake servers are how CI tests this adapter without credentials."""
    adapter = OpenAICompatibleAdapter(base_url="http://127.0.0.1:9999/v1", api_key="k")
    assert adapter.base_url.startswith("http://127.0.0.1")


def _mock_transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_http_adapter_maps_a_successful_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1",
        api_key="test-key",
        client=_mock_transport(handler),
    )
    result = await adapter.generate(make_invocation())

    assert result.text == "hi there"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 3


async def test_http_adapter_never_leaks_the_api_key_in_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key sk-secret-value"}})

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1",
        api_key="sk-secret-value",
        client=_mock_transport(handler),
    )
    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert "sk-secret-value" not in str(excinfo.value)
    assert excinfo.value.outcome is AttemptOutcome.AUTH_ERROR


async def test_http_adapter_rejects_malformed_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1", api_key="k", client=_mock_transport(handler)
    )
    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.INVALID_RESPONSE
    assert not excinfo.value.retryable


async def test_http_adapter_rejects_a_response_missing_choices():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1", api_key="k", client=_mock_transport(handler)
    )
    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())
    assert excinfo.value.outcome is AttemptOutcome.INVALID_RESPONSE


async def test_http_adapter_maps_timeouts():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1", api_key="k", client=_mock_transport(handler)
    )
    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.TIMEOUT
    assert excinfo.value.retryable


async def test_http_adapter_sends_json_schema_response_format():
    from dataclasses import replace

    from gateway.domain.requests import OutputFormat

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    invocation = make_invocation()
    invocation = replace(
        invocation,
        request=replace(
            invocation.request,
            output_format=OutputFormat(
                kind="json_schema", json_schema={"type": "object"}, schema_name="r", strict=True
            ),
        ),
    )

    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:9999/v1", api_key="k", client=_mock_transport(handler)
    )
    await adapter.generate(invocation)

    assert captured["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert captured["response_format"]["json_schema"]["strict"] is True  # type: ignore[index]


# --- registry --------------------------------------------------------------


def test_registry_holds_adapters_by_name():
    registry = AdapterRegistry()
    registry.register(FakeProvider())
    registry.register(AlienProvider())

    assert registry.names() == ("alien", "fake")
    assert registry.get("fake") is not None
    assert registry.get("nope") is None
    assert len(registry) == 2
