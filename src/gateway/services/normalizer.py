"""Normalization: both endpoint shapes to one :class:`CanonicalRequest` (§3, §5).

This module is the only place that knows either wire format. Everything
downstream reads the canonical form, which is what keeps two API shapes from
leaking into routing, budgets, and validation.

Token estimation is deliberately crude and clearly labelled: it exists to seed
budget reservation, and M4 replaces it with a real tokenizer per model family.
Over-estimating is the safe direction -- it reserves more than needed and
releases the difference.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gateway.api.schemas import (
    KNOWN_CHAT_FIELDS,
    KNOWN_RESPONSES_FIELDS,
    REJECTED_RESPONSES_FIELDS,
    ChatCompletionRequest,
    ResponsesRequest,
)
from gateway.domain.enums import Capability, Endpoint
from gateway.domain.errors import InvalidRequestError
from gateway.domain.requests import (
    CanonicalRequest,
    GatewayControls,
    Message,
    OutputFormat,
    Sampling,
)
from gateway.telemetry.hashing import keyed_digest

#: Rough characters-per-token ratio for English prose. Used only for the
#: pre-invocation estimate; actual usage is settled from provider-reported
#: counts.
CHARS_PER_TOKEN = 4

#: Added per message to cover role and framing overhead.
PER_MESSAGE_TOKEN_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text``."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _flatten_content(content: Any) -> str:
    """Reduce either content form to plain text for hashing and estimation.

    Only text parts contribute. Image parts are counted by the capability gate,
    not here, because their token cost is model-specific.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if "text" in part and isinstance(part["text"], str):
            parts.append(part["text"])
    return "\n".join(parts)


def _detect_capabilities(
    *,
    tools: list[dict[str, Any]] | None,
    output_format: OutputFormat,
    stream: bool,
    has_image: bool,
) -> set[Capability]:
    """Infer capabilities the request implies (§2).

    Inferred requirements are unioned with any the caller stated, so a request
    that needs tools cannot be routed to a model without them just because the
    caller did not spell it out.
    """
    required: set[Capability] = set()
    if tools:
        required.add(Capability.TOOLS)
    if output_format.kind == "json_schema":
        required.add(Capability.JSON_SCHEMA)
    if stream:
        required.add(Capability.STREAMING)
    if has_image:
        required.add(Capability.VISION)
    return required


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in {"image_url", "input_image", "image"}
        for part in content
    )


def _output_format_from_chat(payload: ChatCompletionRequest) -> OutputFormat:
    response_format = payload.response_format
    if response_format is None:
        return OutputFormat()

    if response_format.type == "json_schema":
        schema_block = response_format.json_schema or {}
        return OutputFormat(
            kind="json_schema",
            json_schema=schema_block.get("schema"),
            schema_name=schema_block.get("name"),
            strict=bool(schema_block.get("strict", False)),
        )
    return OutputFormat(kind=response_format.type)


def _output_format_from_responses(payload: ResponsesRequest) -> OutputFormat:
    if payload.text is None or payload.text.format is None:
        return OutputFormat()

    text_format = payload.text.format
    kind = text_format.get("type", "text")
    if kind == "json_schema":
        return OutputFormat(
            kind="json_schema",
            json_schema=text_format.get("schema"),
            schema_name=text_format.get("name"),
            strict=bool(text_format.get("strict", False)),
        )
    return OutputFormat(kind=kind)


def _ignored_fields(raw_body: dict[str, Any], known: frozenset[str]) -> tuple[str, ...]:
    """Standard fields accepted but not acted on (§1)."""
    return tuple(sorted(key for key in raw_body if key not in known))


def _normalize_stop(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def normalize_chat_request(
    payload: ChatCompletionRequest,
    *,
    raw_body: dict[str, Any],
    request_id: str,
    client_id: str,
    controls: GatewayControls,
    hash_key: str,
    idempotency_key: str | None = None,
) -> CanonicalRequest:
    """Normalize a Chat Completions body."""
    conversation: list[Message] = []
    system_parts: list[str] = []
    has_image = False

    for message in payload.messages:
        text = _flatten_content(message.content)
        has_image = has_image or _content_has_image(message.content)

        # system/developer turns become system_instructions so both endpoints
        # present instructions to the router the same way.
        if message.role in {"system", "developer"}:
            system_parts.append(text)
            continue

        conversation.append(
            Message(
                role=message.role,
                content=text,
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=tuple(message.tool_calls or ()),
            )
        )

    if not conversation:
        raise InvalidRequestError("At least one non-system message is required.", param="messages")

    output_format = _output_format_from_chat(payload)
    required = _detect_capabilities(
        tools=payload.tools,
        output_format=output_format,
        stream=payload.stream,
        has_image=has_image,
    )

    system_instructions = "\n\n".join(part for part in system_parts if part) or None

    return _build(
        request_id=request_id,
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model=payload.model,
        conversation=conversation,
        system_instructions=system_instructions,
        controls=controls,
        client_id=client_id,
        tools=payload.tools,
        tool_choice=payload.tool_choice,
        parallel_tool_calls=payload.parallel_tool_calls,
        output_format=output_format,
        sampling=Sampling(
            temperature=payload.temperature,
            top_p=payload.top_p,
            frequency_penalty=payload.frequency_penalty,
            presence_penalty=payload.presence_penalty,
            seed=payload.seed,
            stop=_normalize_stop(payload.stop),
        ),
        max_output_tokens=payload.max_completion_tokens or payload.max_tokens,
        stream=payload.stream,
        inferred_capabilities=required,
        ignored_fields=_ignored_fields(raw_body, KNOWN_CHAT_FIELDS),
        hash_key=hash_key,
        idempotency_key=idempotency_key,
    )


def normalize_responses_request(
    payload: ResponsesRequest,
    *,
    raw_body: dict[str, Any],
    request_id: str,
    client_id: str,
    controls: GatewayControls,
    hash_key: str,
    idempotency_key: str | None = None,
) -> CanonicalRequest:
    """Normalize a Responses body."""
    for field, reason in REJECTED_RESPONSES_FIELDS.items():
        if raw_body.get(field) not in (None, False):
            raise InvalidRequestError(reason, param=field)

    conversation: list[Message] = []
    has_image = False

    if isinstance(payload.input, str):
        conversation.append(Message(role="user", content=payload.input))
    else:
        for item in payload.input:
            role = item.get("role", "user")
            content = item.get("content")
            has_image = has_image or _content_has_image(content)
            conversation.append(Message(role=role, content=_flatten_content(content)))

    if not conversation:
        raise InvalidRequestError("input must contain at least one item.", param="input")

    output_format = _output_format_from_responses(payload)
    required = _detect_capabilities(
        tools=payload.tools,
        output_format=output_format,
        stream=payload.stream,
        has_image=has_image,
    )
    if payload.reasoning:
        required.add(Capability.REASONING)

    return _build(
        request_id=request_id,
        endpoint=Endpoint.RESPONSES,
        requested_model=payload.model,
        conversation=conversation,
        system_instructions=payload.instructions,
        controls=controls,
        client_id=client_id,
        tools=payload.tools,
        tool_choice=payload.tool_choice,
        parallel_tool_calls=payload.parallel_tool_calls,
        output_format=output_format,
        sampling=Sampling(temperature=payload.temperature, top_p=payload.top_p),
        max_output_tokens=payload.max_output_tokens,
        stream=payload.stream,
        inferred_capabilities=required,
        ignored_fields=_ignored_fields(raw_body, KNOWN_RESPONSES_FIELDS),
        hash_key=hash_key,
        idempotency_key=idempotency_key,
    )


def _build(
    *,
    request_id: str,
    endpoint: Endpoint,
    requested_model: str,
    conversation: list[Message],
    system_instructions: str | None,
    controls: GatewayControls,
    client_id: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    parallel_tool_calls: bool | None,
    output_format: OutputFormat,
    sampling: Sampling,
    max_output_tokens: int | None,
    stream: bool,
    inferred_capabilities: set[Capability],
    ignored_fields: tuple[str, ...],
    hash_key: str,
    idempotency_key: str | None,
) -> CanonicalRequest:
    """Assemble the canonical request shared by both endpoints."""
    # Inferred capabilities union with declared ones: a caller cannot opt out
    # of a requirement their own payload creates.
    effective_controls = replace(
        controls,
        required_capabilities=frozenset(controls.required_capabilities | inferred_capabilities),
    )

    digest_source = _canonical_digest_source(
        endpoint=endpoint,
        requested_model=requested_model,
        system_instructions=system_instructions,
        conversation=conversation,
        output_format=output_format,
    )

    text_length = sum(len(message.content) for message in conversation)
    text_length += len(system_instructions or "")
    estimated = text_length // CHARS_PER_TOKEN + PER_MESSAGE_TOKEN_OVERHEAD * len(conversation)

    return CanonicalRequest(
        request_id=request_id,
        endpoint=endpoint,
        requested_model=requested_model,
        conversation=tuple(conversation),
        controls=effective_controls,
        client_id=client_id,
        system_instructions=system_instructions,
        tools=tuple(tools or ()),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        output_format=output_format,
        sampling=sampling,
        max_output_tokens=max_output_tokens,
        stream=stream,
        input_hash=keyed_digest(digest_source, key=hash_key),
        estimated_input_tokens=max(estimated, 1),
        idempotency_key=idempotency_key,
        ignored_fields=ignored_fields,
    )


def _canonical_digest_source(
    *,
    endpoint: Endpoint,
    requested_model: str,
    system_instructions: str | None,
    conversation: list[Message],
    output_format: OutputFormat,
) -> str:
    """Build the string that identifies this request's semantic content.

    Used only as HMAC input. It must be stable for identical requests, because
    idempotency replay (§1) compares these digests -- and it deliberately
    excludes the request ID and timestamps, which differ between a request and
    its legitimate retry.
    """
    parts = [
        endpoint.value,
        requested_model,
        system_instructions or "",
        output_format.kind,
    ]
    for message in conversation:
        parts.append(f"{message.role}:{message.content}")
    return "\x1f".join(parts)
