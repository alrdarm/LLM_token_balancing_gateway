"""Deterministic fake providers (§13 build order).

Two of them, and the second is the point.

:class:`FakeProvider` is the well-behaved reference: deterministic output,
scriptable failures, no I/O. It exists so state and budget correctness can be
established before provider variability enters.

:class:`AlienProvider` is deliberately *un*-OpenAI-shaped -- nested content
parts, different usage field names, a different finish-reason vocabulary, no
streaming. It exists to keep :class:`~gateway.providers.base.ProviderAdapter`
honest. The first real adapter is OpenAI-compatible, whose wire format is
near-identical to the gateway's own; without something structurally different
alongside it, an OpenAI-shaped assumption could sit undetected in the
abstraction until a third-party adapter is written against it.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import (
    ProviderCapabilities,
    ProviderFailure,
    ProviderInvocation,
    ProviderResult,
    ProviderUsage,
    StreamChunk,
)


def _deterministic_text(invocation: ProviderInvocation, *, prefix: str) -> str:
    """Stable output for a given request.

    Derived from the request's input hash so the same request always produces
    the same output, which is what lets orchestration tests assert on results
    without pinning provider behaviour to a literal string.
    """
    seed = hashlib.sha256(
        f"{prefix}:{invocation.gateway_model_id}:{invocation.request.input_hash}".encode()
    ).hexdigest()[:12]
    return f"{prefix} response {seed}"


def _estimate_usage(invocation: ProviderInvocation, text: str) -> ProviderUsage:
    return ProviderUsage(
        prompt_tokens=max(invocation.request.estimated_input_tokens, 1),
        completion_tokens=max(len(text) // 4, 1),
    )


class ScriptedBehaviour:
    """A queue of outcomes a fake provider should produce, in order.

    Lets a test drive a precise sequence -- 429, then success -- without
    patching internals, which keeps failure tests readable and makes the
    retry/fallback logic the thing under test rather than the mock.
    """

    def __init__(self) -> None:
        self._queue: deque[ProviderFailure | None] = deque()

    def fail_next(self, failure: ProviderFailure) -> ScriptedBehaviour:
        self._queue.append(failure)
        return self

    def succeed_next(self) -> ScriptedBehaviour:
        self._queue.append(None)
        return self

    def pop(self) -> ProviderFailure | None:
        return self._queue.popleft() if self._queue else None

    @property
    def remaining(self) -> int:
        return len(self._queue)


class FakeProvider:
    """A deterministic, well-behaved provider."""

    name = "fake"

    def __init__(
        self,
        *,
        latency_ms: int = 5,
        behaviour: ScriptedBehaviour | None = None,
        text: str | None = None,
    ) -> None:
        self.latency_ms = latency_ms
        self.behaviour = behaviour or ScriptedBehaviour()
        self.forced_text = text
        #: Every invocation seen, so tests can assert how many calls were made
        #: -- the difference between "retried once" and "retried twice" is
        #: money.
        self.calls: list[ProviderInvocation] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_json_schema=True,
            supports_vision=True,
        )

    async def generate(self, invocation: ProviderInvocation) -> ProviderResult:
        self.calls.append(invocation)

        scripted = self.behaviour.pop()
        if scripted is not None:
            raise scripted

        await asyncio.sleep(self.latency_ms / 1000)
        text = self.forced_text or _deterministic_text(invocation, prefix="fake")

        return ProviderResult(
            text=text,
            finish_reason="stop",
            usage=_estimate_usage(invocation, text),
            model_id=invocation.gateway_model_id,
            latency_ms=self.latency_ms,
        )

    async def stream(self, invocation: ProviderInvocation) -> AsyncIterator[StreamChunk]:
        self.calls.append(invocation)

        scripted = self.behaviour.pop()
        if scripted is not None:
            raise scripted

        text = self.forced_text or _deterministic_text(invocation, prefix="fake")
        words = text.split(" ")
        for index, word in enumerate(words):
            await asyncio.sleep(self.latency_ms / 1000 / max(len(words), 1))
            yield StreamChunk(delta=word if index == 0 else f" {word}")

        yield StreamChunk(finish_reason="stop", usage=_estimate_usage(invocation, text))


class AlienProvider:
    """A provider whose shape shares nothing with the gateway's own format.

    Its native payload uses nested segments, ``tokens_in``/``tokens_out``, and
    completion states like ``COMPLETE``/``TRUNCATED``. Everything foreign is
    translated here, at the boundary, so nothing above it needs to know.
    """

    name = "alien"

    #: The alien vocabulary, mapped onto the gateway's finish reasons.
    FINISH_REASONS: ClassVar[dict[str, str]] = {
        "COMPLETE": "stop",
        "TRUNCATED": "length",
        "BLOCKED": "content_filter",
        "TOOL_REQUESTED": "tool_calls",
    }

    def __init__(
        self,
        *,
        latency_ms: int = 5,
        behaviour: ScriptedBehaviour | None = None,
        native_state: str = "COMPLETE",
    ) -> None:
        self.latency_ms = latency_ms
        self.behaviour = behaviour or ScriptedBehaviour()
        self.native_state = native_state
        self.calls: list[ProviderInvocation] = []

    def capabilities(self) -> ProviderCapabilities:
        # Deliberately narrower than the fake: an adapter that cannot stream
        # must be routable only for non-streaming requests.
        return ProviderCapabilities(
            supports_streaming=False,
            supports_tools=False,
            supports_json_schema=False,
            supports_vision=False,
        )

    def _native_payload(self, invocation: ProviderInvocation) -> dict[str, Any]:
        """What this provider would actually return over the wire."""
        text = _deterministic_text(invocation, prefix="alien")
        return {
            "output": {"segments": [{"kind": "text", "body": text}]},
            "completion_state": self.native_state,
            "meter": {
                "tokens_in": max(invocation.request.estimated_input_tokens, 1),
                "tokens_out": max(len(text) // 4, 1),
            },
        }

    async def generate(self, invocation: ProviderInvocation) -> ProviderResult:
        self.calls.append(invocation)

        scripted = self.behaviour.pop()
        if scripted is not None:
            raise scripted

        await asyncio.sleep(self.latency_ms / 1000)
        payload = self._native_payload(invocation)

        # Translation happens here and nowhere else.
        segments = payload["output"]["segments"]
        text = "".join(segment["body"] for segment in segments if segment.get("kind") == "text")
        meter = payload["meter"]
        state = str(payload["completion_state"])

        return ProviderResult(
            text=text,
            finish_reason=self.FINISH_REASONS.get(state, "stop"),
            usage=ProviderUsage(
                prompt_tokens=int(meter["tokens_in"]),
                completion_tokens=int(meter["tokens_out"]),
            ),
            model_id=invocation.gateway_model_id,
            latency_ms=self.latency_ms,
        )

    async def stream(self, invocation: ProviderInvocation) -> AsyncIterator[StreamChunk]:
        """Not supported; the capability gate should have excluded this model."""
        raise ProviderFailure(
            "This provider does not support streaming.",
            outcome=AttemptOutcome.CAPABILITY_REJECTED,
            retryable=False,
        )
        # Unreachable, and deliberately so: the bare ``yield`` is what makes
        # this an async generator, which the ProviderAdapter protocol
        # requires. Without it, ``async for`` over the result would fail
        # with a confusing TypeError instead of the capability failure.
        yield StreamChunk()  # type: ignore[unreachable]  # pragma: no cover


# --- convenience failures for tests and fixtures ---------------------------


def rate_limited(retry_after: float | None = 0.01) -> ProviderFailure:
    """A 429 before any output: retryable within budget and deadline (§8)."""
    return ProviderFailure(
        "Provider rate limited the request.",
        outcome=AttemptOutcome.RATE_LIMITED,
        retryable=True,
        retry_after_seconds=retry_after,
    )


def transient_server_error() -> ProviderFailure:
    """A retryable 5xx before output (§8)."""
    return ProviderFailure(
        "Provider returned a transient error.",
        outcome=AttemptOutcome.PROVIDER_ERROR,
        retryable=True,
    )


def auth_error() -> ProviderFailure:
    """Configuration fault: §8 requires opening the circuit, never blind retry."""
    return ProviderFailure(
        "Provider rejected the credential.",
        outcome=AttemptOutcome.AUTH_ERROR,
        retryable=False,
    )


def timeout_before_output() -> ProviderFailure:
    return ProviderFailure(
        "Provider timed out before producing output.",
        outcome=AttemptOutcome.TIMEOUT,
        retryable=True,
    )


def partial_stream_failure(prompt_tokens: int, completion_tokens: int) -> ProviderFailure:
    """A stream cut after output was visible.

    Terminal by §4, and still billable: the usage already incurred must be
    settled rather than released.
    """
    return ProviderFailure(
        "Stream failed after output was emitted.",
        outcome=AttemptOutcome.PARTIAL,
        retryable=False,
        emitted_output=True,
        usage=ProviderUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
