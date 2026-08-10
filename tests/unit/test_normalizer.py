"""Normalization of both endpoint shapes into one canonical request (§3, §5)."""

from __future__ import annotations

import pytest

from gateway.api.schemas import ChatCompletionRequest, ResponsesRequest
from gateway.domain.enums import Capability, Endpoint
from gateway.domain.errors import InvalidRequestError
from gateway.domain.requests import GatewayControls
from gateway.services.normalizer import (
    normalize_chat_request,
    normalize_responses_request,
)

HASH_KEY = "test-hash-key"


def chat(body: dict) -> object:
    payload = ChatCompletionRequest.model_validate(body)
    return normalize_chat_request(
        payload,
        raw_body=body,
        request_id="req_1",
        client_id="client_a",
        controls=GatewayControls(),
        hash_key=HASH_KEY,
    )


def responses(body: dict) -> object:
    payload = ResponsesRequest.model_validate(body)
    return normalize_responses_request(
        payload,
        raw_body=body,
        request_id="req_1",
        client_id="client_a",
        controls=GatewayControls(),
        hash_key=HASH_KEY,
    )


def test_chat_normalizes_to_canonical():
    canonical = chat(
        {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
        }
    )
    assert canonical.endpoint is Endpoint.CHAT_COMPLETIONS
    assert canonical.system_instructions == "Be brief."
    assert [m.role for m in canonical.conversation] == ["user"]
    assert canonical.conversation[0].content == "Hello"


def test_responses_normalizes_to_the_same_shape():
    canonical = responses({"model": "auto", "instructions": "Be brief.", "input": "Hello"})
    assert canonical.endpoint is Endpoint.RESPONSES
    assert canonical.system_instructions == "Be brief."
    assert canonical.conversation[0].content == "Hello"


def test_both_endpoints_agree_on_the_input_hash():
    """The same semantic request must hash identically across endpoints.

    Idempotency replay compares these digests, so a divergence would make the
    same conversation look like two different requests.
    """
    from gateway.services.normalizer import _canonical_digest_source

    left = chat(
        {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
        }
    )
    right = responses({"model": "auto", "instructions": "Be brief.", "input": "Hello"})

    # Endpoint is part of the digest by design, so compare the content portion.
    assert left.conversation == right.conversation
    assert left.system_instructions == right.system_instructions
    assert _canonical_digest_source(
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model=left.requested_model,
        system_instructions=left.system_instructions,
        conversation=list(left.conversation),
        output_format=left.output_format,
    ) == _canonical_digest_source(
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model=right.requested_model,
        system_instructions=right.system_instructions,
        conversation=list(right.conversation),
        output_format=right.output_format,
    )


def test_identical_requests_hash_identically():
    body = {"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}
    assert chat(body).input_hash == chat(dict(body)).input_hash


def test_different_content_hashes_differently():
    a = chat({"model": "auto", "messages": [{"role": "user", "content": "A"}]})
    b = chat({"model": "auto", "messages": [{"role": "user", "content": "B"}]})
    assert a.input_hash != b.input_hash


def test_input_hash_does_not_contain_the_prompt():
    secret = "my confidential prompt text"
    canonical = chat({"model": "auto", "messages": [{"role": "user", "content": secret}]})
    assert secret not in canonical.input_hash
    assert len(canonical.input_hash) == 64


def test_tools_infer_the_tools_capability():
    canonical = chat(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }
    )
    assert Capability.TOOLS in canonical.controls.required_capabilities


def test_json_schema_infers_the_schema_capability():
    canonical = chat(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "r", "schema": {"type": "object"}},
            },
        }
    )
    assert Capability.JSON_SCHEMA in canonical.controls.required_capabilities
    assert canonical.output_format.requires_schema_validation


def test_streaming_infers_the_streaming_capability():
    canonical = chat(
        {"model": "auto", "messages": [{"role": "user", "content": "x"}], "stream": True}
    )
    assert Capability.STREAMING in canonical.controls.required_capabilities


def test_image_content_infers_vision():
    canonical = chat(
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        }
    )
    assert Capability.VISION in canonical.controls.required_capabilities


def test_caller_cannot_opt_out_of_an_inferred_capability():
    """Inference unions with declared requirements rather than replacing them."""
    payload = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }
    )
    canonical = normalize_chat_request(
        payload,
        raw_body={},
        request_id="req_1",
        client_id="c",
        controls=GatewayControls(required_capabilities=frozenset({Capability.VISION})),
        hash_key=HASH_KEY,
    )
    assert {Capability.TOOLS, Capability.VISION} <= canonical.controls.required_capabilities


def test_max_completion_tokens_wins_over_max_tokens():
    canonical = chat(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 256,
        }
    )
    assert canonical.max_output_tokens == 256


def test_unknown_standard_fields_are_recorded_as_ignored():
    """§1 permits ignoring safely ignorable fields, but with telemetry."""
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "x"}],
        "logit_bias": {"1": 1},
        "service_tier": "auto",
    }
    canonical = chat(body)
    assert canonical.ignored_fields == ("logit_bias", "service_tier")


def test_system_only_conversation_is_rejected():
    with pytest.raises(InvalidRequestError):
        chat({"model": "auto", "messages": [{"role": "system", "content": "only"}]})


def test_responses_rejects_background_mode():
    with pytest.raises(InvalidRequestError):
        responses({"model": "auto", "input": "hi", "background": True})


def test_responses_structured_input_is_flattened():
    canonical = responses(
        {
            "model": "auto",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Review this"}]}],
        }
    )
    assert canonical.conversation[0].content == "Review this"


def test_stop_string_becomes_a_tuple():
    canonical = chat(
        {"model": "auto", "messages": [{"role": "user", "content": "x"}], "stop": "END"}
    )
    assert canonical.sampling.stop == ("END",)


def test_estimated_tokens_are_positive():
    canonical = chat({"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
    assert canonical.estimated_input_tokens >= 1


def test_canonical_request_is_immutable():
    """A control mutated after planning would invalidate frozen decisions."""
    canonical = chat({"model": "auto", "messages": [{"role": "user", "content": "x"}]})
    with pytest.raises((AttributeError, TypeError)):
        canonical.requested_model = "other"  # type: ignore[misc]
