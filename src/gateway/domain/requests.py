"""The canonical request and its controls (§2, §5).

Both generation endpoints normalize to one immutable :class:`CanonicalRequest`.
Everything downstream -- routing, budgets, validation, escalation -- reads only
this, never a framework object or a provider payload. That is what stops two
API shapes from leaking into the core.

Immutability is enforced (frozen dataclasses) rather than assumed: a route plan
is frozen against a specific request, and a control mutated after planning
would silently invalidate the decisions already made from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gateway.domain.enums import (
    PRIVACY_ORDER,
    QUALITY_ORDER,
    RISK_ORDER,
    Capability,
    Endpoint,
    Privacy,
    Quality,
    Risk,
    TaskClass,
    strictest,
)

#: Model selectors (§1). Anything else is treated as an explicit model ID.
SELECTORS: frozenset[str] = frozenset(
    {"auto", "auto-cheap", "auto-fast", "auto-quality", "auto-private"}
)

#: ``auto-private`` forces confidential handling regardless of what was asked.
SELECTOR_FORCED_PRIVACY: dict[str, Privacy] = {"auto-private": Privacy.CONFIDENTIAL}


class ValidationMode:
    """Requested validation depth (§2)."""

    AUTO = "auto"
    NONE = "none"
    DETERMINISTIC = "deterministic"
    INDEPENDENT = "independent"

    ALL = frozenset({AUTO, NONE, DETERMINISTIC, INDEPENDENT})


@dataclass(frozen=True, slots=True)
class Sampling:
    """Sampling parameters passed through to the provider."""

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GatewayControls:
    """Effective controls after precedence resolution (§2).

    These are *resolved* values: by the time this object exists, the precedence
    chain has already been applied and privacy, risk, and quality hold the
    strictest applicable value.
    """

    quality: Quality = Quality.STANDARD
    #: Defaults to confidential. The spec's default is the safe one: a caller
    #: who says nothing must not have their prompt sent to public-data tiers.
    privacy: Privacy = Privacy.CONFIDENTIAL
    max_cost: Decimal | None = None
    max_latency_ms: int | None = None
    provider_allow: tuple[str, ...] = ()
    provider_deny: tuple[str, ...] = ()
    required_capabilities: frozenset[Capability] = frozenset()
    allow_fallback: bool = True
    max_attempts: int = 3
    validation: str = ValidationMode.AUTO
    task_class: TaskClass | None = None
    risk: Risk | None = None
    dry_run: bool = False
    metadata: tuple[tuple[str, str], ...] = ()
    debug: bool = False

    def provider_allowed(self, provider: str) -> bool:
        """Whether ``provider`` survives the allow/deny rules.

        Deny always wins, and a non-empty allow list is an intersection: an
        unlisted provider is excluded rather than tolerated.
        """
        if provider in self.provider_deny:
            return False
        if self.provider_allow and provider not in self.provider_allow:
            return False
        return True


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn, normalized across both endpoint shapes."""

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class OutputFormat:
    """Requested response format, normalized from both endpoints.

    ``kind`` is ``text``, ``json_object``, or ``json_schema``. A JSON schema is
    a hard validation gate: §2 forbids ``validation=none`` from bypassing it.
    """

    kind: str = "text"
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    strict: bool = False

    @property
    def requires_schema_validation(self) -> bool:
        return self.kind == "json_schema" and self.json_schema is not None


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    """The single immutable representation both endpoints normalize to (§5)."""

    request_id: str
    endpoint: Endpoint
    requested_model: str
    conversation: tuple[Message, ...]
    controls: GatewayControls
    client_id: str

    system_instructions: str | None = None
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    output_format: OutputFormat = field(default_factory=OutputFormat)
    sampling: Sampling = field(default_factory=Sampling)
    max_output_tokens: int | None = None
    stream: bool = False

    #: Keyed digest of the canonical input. Never the input itself.
    input_hash: str = ""
    estimated_input_tokens: int = 0
    idempotency_key: str | None = None
    #: Standard fields accepted but ignored, recorded for telemetry (§1).
    ignored_fields: tuple[str, ...] = ()

    @property
    def is_selector(self) -> bool:
        """Whether the caller asked for a selector rather than a specific model."""
        return self.requested_model in SELECTORS

    @property
    def requires_streaming_capability(self) -> bool:
        return self.stream


def resolve_strictest_privacy(*values: Privacy | None) -> Privacy:
    """Return the strictest privacy level among ``values`` (§2)."""
    present = [value for value in values if value is not None]
    return strictest(present, PRIVACY_ORDER) if present else Privacy.CONFIDENTIAL


def resolve_strictest_quality(*values: Quality | None) -> Quality:
    """Return the strictest quality floor among ``values`` (§2)."""
    present = [value for value in values if value is not None]
    return strictest(present, QUALITY_ORDER) if present else Quality.STANDARD


def resolve_strictest_risk(*values: Risk | None) -> Risk | None:
    """Return the strictest risk among ``values``.

    A caller may raise inferred risk but never lower it, so taking the maximum
    is the whole rule.
    """
    present = [value for value in values if value is not None]
    return strictest(present, RISK_ORDER) if present else None
