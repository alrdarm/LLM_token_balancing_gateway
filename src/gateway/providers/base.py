"""The provider abstraction boundary (§5).

``ProviderAdapter`` is where provider variety stops. Everything above it -- the
router, the budget manager, the orchestrator -- sees only
:class:`ProviderResult` and :class:`ProviderFailure`, never a vendor payload.

Adapters translate; they do not decide. Retry, fallback, and escalation are
orchestration concerns (§8), so an adapter reports *what happened* in the
vocabulary of :class:`~gateway.domain.enums.AttemptOutcome` and lets the
orchestrator choose. An adapter that retried internally would spend money the
budget manager never reserved.

Errors carry a classification, not a provider message: §9 forbids exposing raw
provider payloads, and §8 needs the distinction between "retryable before
output" and "terminal" to be explicit rather than inferred from a string.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from gateway.domain.enums import AttemptOutcome, Capability
from gateway.domain.requests import CanonicalRequest


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What an adapter can do, independent of any one model."""

    supports_streaming: bool = False
    supports_tools: bool = False
    supports_json_schema: bool = False
    supports_vision: bool = False
    supports_cancellation: bool = True

    def as_capability_set(self) -> frozenset[Capability]:
        mapping = {
            Capability.STREAMING: self.supports_streaming,
            Capability.TOOLS: self.supports_tools,
            Capability.JSON_SCHEMA: self.supports_json_schema,
            Capability.VISION: self.supports_vision,
        }
        return frozenset(capability for capability, present in mapping.items() if present)


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """Token counts as reported by the provider.

    ``None`` means the provider did not report a figure -- distinct from zero,
    because settling an unknown usage as zero would understate real spend.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """A completed generation, normalized."""

    text: str
    finish_reason: str
    usage: ProviderUsage
    model_id: str
    latency_ms: int
    tool_calls: tuple[dict[str, Any], ...] = ()
    #: True once any part of this output was made visible to the client. After
    #: that point §4 forbids switching models.
    emitted_output: bool = False
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One incremental piece of a streamed response."""

    delta: str = ""
    finish_reason: str | None = None
    usage: ProviderUsage | None = None


class ProviderFailure(Exception):
    """A provider call that did not produce a usable result.

    ``outcome`` classifies it in the spec's vocabulary so §8's transition table
    can be applied without parsing messages. ``emitted_output`` decides whether
    a retry is even permitted: once output is client-visible, §4 makes failure
    terminal.
    """

    def __init__(
        self,
        message: str,
        *,
        outcome: AttemptOutcome,
        retryable: bool = False,
        emitted_output: bool = False,
        retry_after_seconds: float | None = None,
        usage: ProviderUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.outcome = outcome
        self.retryable = retryable
        self.emitted_output = emitted_output
        self.retry_after_seconds = retry_after_seconds
        #: Usage incurred before failing. A stream cut mid-flight still bills.
        self.usage = usage or ProviderUsage()


@dataclass(frozen=True, slots=True)
class ProviderInvocation:
    """Everything an adapter needs for one call.

    Carries the canonical request plus the resolved model, so an adapter never
    has to consult the registry or make a routing decision of its own.
    """

    request: CanonicalRequest
    provider_model_id: str
    gateway_model_id: str
    max_output_tokens: int | None
    timeout_seconds: float
    #: Estimated cost, for logging and reconciliation only. Adapters must not
    #: use it to decide anything.
    estimated_cost: Decimal = Decimal("0")


@runtime_checkable
class ProviderAdapter(Protocol):
    """The contract every provider must satisfy (§5)."""

    name: str

    def capabilities(self) -> ProviderCapabilities:
        """What this adapter supports."""
        ...

    async def generate(self, invocation: ProviderInvocation) -> ProviderResult:
        """Run one non-streaming generation.

        Raises :class:`ProviderFailure` for anything that is not a usable
        result. Must not retry internally.
        """
        ...

    def stream(self, invocation: ProviderInvocation) -> AsyncIterator[StreamChunk]:
        """Run one streaming generation.

        Declared without ``async`` deliberately: an async *generator* function
        returns its iterator directly, whereas ``async def` here would type as a
        coroutine that must first be awaited to obtain one.
        """
        ...


class AdapterRegistry:
    """Adapters available to this deployment, keyed by provider name."""

    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, provider: str) -> ProviderAdapter | None:
        return self._adapters.get(provider)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def clear(self) -> None:
        self._adapters.clear()

    def __contains__(self, provider: object) -> bool:
        return provider in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)
