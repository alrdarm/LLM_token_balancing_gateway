"""OpenAI-compatible HTTP adapter (§14).

One adapter covers every vendor speaking the OpenAI wire format -- vLLM,
Together, Groq, a local server -- by changing ``base_url`` alone.

The mapping is close to identity, which is the reason an intentionally
different fake exists alongside it: near-identity would otherwise let an
OpenAI-shaped assumption hide inside the abstraction.

Security posture (§11): outbound destinations are checked against an allowlist
before any connection, TLS verification is never disabled, and the API key is
read from configuration and never logged. Errors are classified into the
gateway's own vocabulary rather than forwarded, because §9 forbids exposing
raw provider payloads.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import (
    ProviderCapabilities,
    ProviderFailure,
    ProviderInvocation,
    ProviderResult,
    ProviderUsage,
    StreamChunk,
)

logger = logging.getLogger(__name__)

#: HTTP statuses that are transient before any output (§8).
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OutboundNotAllowedError(ProviderFailure):
    """The configured base URL is not on the allowlist (§11)."""


def _check_destination(base_url: str, allowed_hosts: frozenset[str]) -> None:
    """Refuse to call a host the deployment has not allowlisted.

    §11 requires an outbound allowlist and TLS verification. Checking here,
    before the client is constructed, means a misconfiguration fails closed
    rather than silently exfiltrating a prompt to an unexpected host.
    """
    parsed = urlparse(base_url)

    if parsed.scheme not in ("https", "http"):
        raise OutboundNotAllowedError(
            "Provider base URL must use http or https.",
            outcome=AttemptOutcome.AUTH_ERROR,
        )

    host = parsed.hostname or ""
    if allowed_hosts and host not in allowed_hosts:
        # The host is named because it is deployment configuration, not caller
        # data, and an operator needs it to fix the allowlist.
        raise OutboundNotAllowedError(
            f"Outbound host is not allowlisted: {host}",
            outcome=AttemptOutcome.AUTH_ERROR,
        )

    if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise OutboundNotAllowedError(
            "Plaintext HTTP is permitted only for local endpoints.",
            outcome=AttemptOutcome.AUTH_ERROR,
        )


def _classify(status: int, retry_after: str | None) -> ProviderFailure:
    """Map an HTTP status onto the gateway's outcome vocabulary (§8)."""
    seconds: float | None = None
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = None

    if status in (401, 403):
        return ProviderFailure(
            "Provider rejected the credential.",
            outcome=AttemptOutcome.AUTH_ERROR,
            retryable=False,
        )
    if status == 429:
        return ProviderFailure(
            "Provider rate limited the request.",
            outcome=AttemptOutcome.RATE_LIMITED,
            retryable=True,
            retry_after_seconds=seconds,
        )
    if status == 413:
        return ProviderFailure(
            "Provider rejected the request as too large.",
            outcome=AttemptOutcome.CONTEXT_REJECTED,
            retryable=False,
        )
    if status in RETRYABLE_STATUSES:
        return ProviderFailure(
            "Provider returned a transient error.",
            outcome=AttemptOutcome.PROVIDER_ERROR,
            retryable=True,
            retry_after_seconds=seconds,
        )
    return ProviderFailure(
        "Provider returned an error.",
        outcome=AttemptOutcome.PROVIDER_ERROR,
        retryable=False,
    )


class OpenAICompatibleAdapter:
    """Calls any OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        allowed_hosts: frozenset[str] = frozenset(),
        client: httpx.AsyncClient | None = None,
        name: str | None = None,
    ) -> None:
        _check_destination(base_url, allowed_hosts)

        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        if name:
            self.name = name

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_json_schema=True,
            supports_vision=True,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, invocation: ProviderInvocation, *, stream: bool) -> dict[str, Any]:
        """Translate the canonical request into the OpenAI wire format."""
        request = invocation.request
        messages: list[dict[str, Any]] = []

        if request.system_instructions:
            messages.append({"role": "system", "content": request.system_instructions})
        for message in request.conversation:
            messages.append({"role": message.role, "content": message.content})

        payload: dict[str, Any] = {
            "model": invocation.provider_model_id,
            "messages": messages,
            "stream": stream,
        }

        if invocation.max_output_tokens is not None:
            payload["max_tokens"] = invocation.max_output_tokens

        sampling = request.sampling
        if sampling.temperature is not None:
            payload["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            payload["top_p"] = sampling.top_p
        if sampling.stop:
            payload["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            payload["seed"] = sampling.seed

        if request.tools:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        output_format = request.output_format
        if output_format.kind == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif output_format.requires_schema_validation:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.schema_name or "response",
                    "schema": output_format.json_schema,
                    "strict": output_format.strict,
                },
            }

        return payload

    async def _post(
        self, invocation: ProviderInvocation, payload: dict[str, Any]
    ) -> httpx.Response:
        client = self._client or httpx.AsyncClient(verify=True)
        owns_client = self._client is None
        try:
            return await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=invocation.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderFailure(
                "Provider timed out.",
                outcome=AttemptOutcome.TIMEOUT,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            # The transport message can contain URLs and internal detail, so it
            # is logged rather than surfaced.
            logger.warning("Provider transport error", extra={"event": "provider_error"})
            raise ProviderFailure(
                "Provider connection failed.",
                outcome=AttemptOutcome.PROVIDER_ERROR,
                retryable=True,
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def generate(self, invocation: ProviderInvocation) -> ProviderResult:
        started = time.monotonic()
        response = await self._post(invocation, self._payload(invocation, stream=False))

        if response.status_code >= 400:
            raise _classify(response.status_code, response.headers.get("retry-after"))

        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            usage = body.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderFailure(
                "Provider returned a response the adapter could not interpret.",
                outcome=AttemptOutcome.INVALID_RESPONSE,
                retryable=False,
            ) from exc

        return ProviderResult(
            text=text,
            finish_reason=choice.get("finish_reason") or "stop",
            usage=ProviderUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ),
            model_id=invocation.gateway_model_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            tool_calls=tuple(message.get("tool_calls") or ()),
        )

    async def stream(self, invocation: ProviderInvocation) -> AsyncIterator[StreamChunk]:
        """Stream via SSE.

        Once a delta has been yielded the caller may have shown it, so any
        later failure is reported with ``emitted_output`` set and becomes
        terminal (§4).
        """
        import json

        client = self._client or httpx.AsyncClient(verify=True)
        owns_client = self._client is None
        emitted = False

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self._payload(invocation, stream=True),
                headers=self._headers(),
                timeout=invocation.timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _classify(response.status_code, response.headers.get("retry-after"))

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                        choice = event["choices"][0]
                        delta = (choice.get("delta") or {}).get("content") or ""
                        finish = choice.get("finish_reason")
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise ProviderFailure(
                            "Provider sent a malformed stream event.",
                            outcome=AttemptOutcome.INVALID_RESPONSE,
                            retryable=not emitted,
                            emitted_output=emitted,
                        ) from exc

                    if delta:
                        emitted = True
                        yield StreamChunk(delta=delta)
                    if finish:
                        yield StreamChunk(finish_reason=finish)

        except httpx.TimeoutException as exc:
            raise ProviderFailure(
                "Provider timed out during streaming.",
                outcome=AttemptOutcome.PARTIAL if emitted else AttemptOutcome.TIMEOUT,
                retryable=not emitted,
                emitted_output=emitted,
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
