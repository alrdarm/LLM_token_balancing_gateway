"""Control precedence resolution (§2).

The chain, strictest first::

    deployment security policy
      > authenticated client policy
      > X-LLM-* headers
      > gateway body
      > model-alias defaults
      > system defaults

"Precedence" is not simply "later wins". For privacy, risk, and quality the
effective value is the **strictest** applicable one, so a lower-precedence
layer asking for something stricter still wins on those three. Everything else
takes the highest-precedence value that was set.

Getting this backwards would let a caller downgrade a deployment's privacy
floor by sending a header, which is the single most dangerous mistake
available in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from gateway.domain.enums import Capability, Privacy, Quality, Risk, TaskClass
from gateway.domain.errors import InvalidGatewayControlError
from gateway.domain.requests import (
    SELECTOR_FORCED_PRIVACY,
    GatewayControls,
    ValidationMode,
    resolve_strictest_privacy,
    resolve_strictest_quality,
    resolve_strictest_risk,
)
from gateway.persistence.types import MoneyError, to_money

HEADER_PREFIX = "x-llm-"


@dataclass(frozen=True, slots=True)
class ControlLayer:
    """One layer of the precedence chain. ``None`` means "not set here"."""

    quality: Quality | None = None
    privacy: Privacy | None = None
    max_cost: Decimal | None = None
    max_latency_ms: int | None = None
    provider_allow: tuple[str, ...] | None = None
    provider_deny: tuple[str, ...] | None = None
    required_capabilities: frozenset[Capability] | None = None
    allow_fallback: bool | None = None
    max_attempts: int | None = None
    validation: str | None = None
    task_class: TaskClass | None = None
    risk: Risk | None = None
    dry_run: bool | None = None
    metadata: tuple[tuple[str, str], ...] | None = None
    debug: bool | None = None


def _first_set(*values: Any) -> Any:
    """Return the first value that was actually set, highest precedence first."""
    for value in values:
        if value is not None:
            return value
    return None


def _lowest(*values: Any) -> Any:
    """Return the smallest value that was set, or ``None`` if none were.

    Used for ceilings, where the strictest limit must win regardless of which
    layer supplied it.
    """
    present = [value for value in values if value is not None]
    return min(present) if present else None


def resolve_controls(
    *,
    deployment: ControlLayer,
    client: ControlLayer,
    headers: ControlLayer,
    body: ControlLayer,
    alias_defaults: ControlLayer,
    system_defaults: GatewayControls,
    requested_model: str,
) -> GatewayControls:
    """Collapse the precedence chain into one effective control set."""
    ordered = (deployment, client, headers, body, alias_defaults)

    # Strictest-wins dimensions: every layer participates, not just the winner.
    privacy = resolve_strictest_privacy(
        *(layer.privacy for layer in ordered), system_defaults.privacy
    )
    quality = resolve_strictest_quality(
        *(layer.quality for layer in ordered), system_defaults.quality
    )
    risk = resolve_strictest_risk(*(layer.risk for layer in ordered))

    # A selector may force a stricter privacy level than anything requested.
    forced = SELECTOR_FORCED_PRIVACY.get(requested_model)
    if forced is not None:
        privacy = resolve_strictest_privacy(privacy, forced)

    # Ceilings take the *lowest* applicable value, never the highest-precedence
    # one. "Never exceed the caller's request cost ceiling" (§2) has to hold in
    # both directions: a deployment default must not silently raise a limit the
    # caller set, and a caller must not raise one the deployment set.
    max_cost = _lowest(*(layer.max_cost for layer in ordered), system_defaults.max_cost)
    max_latency_ms = _lowest(
        *(layer.max_latency_ms for layer in ordered), system_defaults.max_latency_ms
    )

    # Highest-precedence-wins dimensions.
    allow_fallback = _first_set(*(layer.allow_fallback for layer in ordered))
    validation = _first_set(*(layer.validation for layer in ordered))
    task_class = _first_set(*(layer.task_class for layer in ordered))
    dry_run = _first_set(*(layer.dry_run for layer in ordered))
    debug = _first_set(*(layer.debug for layer in ordered))
    metadata = _first_set(*(layer.metadata for layer in ordered))

    # max_attempts takes the *lowest* applicable cap: a policy ceiling must not
    # be raised by a caller asking for more.
    attempt_caps = [layer.max_attempts for layer in ordered if layer.max_attempts is not None]
    max_attempts = min(attempt_caps) if attempt_caps else system_defaults.max_attempts

    # Allow lists intersect and deny lists union, so no layer can widen access
    # granted by a stricter one.
    allow: tuple[str, ...] | None = None
    for layer in ordered:
        if layer.provider_allow is None:
            continue
        allow = (
            tuple(layer.provider_allow)
            if allow is None
            else tuple(sorted(set(allow) & set(layer.provider_allow)))
        )

    deny: set[str] = set()
    for layer in ordered:
        if layer.provider_deny:
            deny |= set(layer.provider_deny)

    capabilities: set[Capability] = set()
    for layer in ordered:
        if layer.required_capabilities:
            capabilities |= set(layer.required_capabilities)

    resolved = replace(
        system_defaults,
        quality=quality,
        privacy=privacy,
        max_cost=max_cost,
        max_latency_ms=max_latency_ms,
        provider_allow=allow or (),
        provider_deny=tuple(sorted(deny)),
        required_capabilities=frozenset(capabilities),
        allow_fallback=(
            allow_fallback if allow_fallback is not None else system_defaults.allow_fallback
        ),
        max_attempts=max_attempts,
        validation=validation or system_defaults.validation,
        task_class=task_class,
        risk=risk,
        dry_run=bool(dry_run) if dry_run is not None else system_defaults.dry_run,
        metadata=metadata or (),
        debug=bool(debug) if debug is not None else system_defaults.debug,
    )
    return resolved


def layer_from_body(payload: Any) -> ControlLayer:
    """Build a layer from a validated ``gateway`` control block."""
    if payload is None:
        return ControlLayer()

    return ControlLayer(
        quality=Quality(payload.quality) if payload.quality else None,
        privacy=Privacy(payload.privacy) if payload.privacy else None,
        max_cost=payload.max_cost,
        max_latency_ms=payload.max_latency_ms,
        provider_allow=tuple(payload.provider_allow) if payload.provider_allow else None,
        provider_deny=tuple(payload.provider_deny) if payload.provider_deny else None,
        required_capabilities=(
            frozenset(Capability(item) for item in payload.required_capabilities)
            if payload.required_capabilities
            else None
        ),
        allow_fallback=payload.allow_fallback,
        max_attempts=payload.max_attempts,
        validation=payload.validation,
        task_class=_parse_task_class(payload.task_class),
        risk=Risk(payload.risk) if payload.risk else None,
        dry_run=payload.dry_run,
        metadata=tuple(sorted(payload.metadata.items())) if payload.metadata else None,
        debug=payload.debug,
    )


def _parse_task_class(value: str | None) -> TaskClass | None:
    if value is None:
        return None
    try:
        return TaskClass(value)
    except ValueError as exc:
        known = ", ".join(sorted(task.value for task in TaskClass))
        raise InvalidGatewayControlError(
            f"Unknown task_class. Supported values: {known}.",
            param="gateway.task_class",
        ) from exc


def layer_from_headers(headers: dict[str, str]) -> ControlLayer:
    """Build a layer from ``X-LLM-*`` headers (§2).

    Headers exist for clients that cannot extend the request body, and they
    outrank the body. A malformed header is rejected rather than ignored: a
    caller who sets ``X-LLM-Max-Cost`` and has it silently dropped would be
    billed under a ceiling they did not choose.
    """
    lowered = {key.lower(): value for key, value in headers.items()}

    def get(name: str) -> str | None:
        value = lowered.get(f"{HEADER_PREFIX}{name}")
        return value.strip() if value is not None else None

    quality = _enum_header(get("quality"), Quality, "X-LLM-Quality")
    privacy = _enum_header(get("privacy"), Privacy, "X-LLM-Privacy")

    if privacy is Privacy.DEPLOYMENT_STRICT:
        # Deployment-strict is a deployment posture, not a caller-selectable
        # level; §2 offers callers public|confidential only.
        raise InvalidGatewayControlError(
            "X-LLM-Privacy accepts 'public' or 'confidential'.",
            param="X-LLM-Privacy",
        )

    max_cost: Decimal | None = None
    raw_cost = get("max-cost")
    if raw_cost is not None:
        try:
            max_cost = to_money(raw_cost)
        except MoneyError as exc:
            raise InvalidGatewayControlError(
                f"X-LLM-Max-Cost is not a valid decimal amount: {exc}",
                param="X-LLM-Max-Cost",
            ) from exc
        if max_cost < 0:
            raise InvalidGatewayControlError(
                "X-LLM-Max-Cost must not be negative.", param="X-LLM-Max-Cost"
            )

    max_latency_ms: int | None = None
    raw_latency = get("max-latency")
    if raw_latency is not None:
        try:
            max_latency_ms = int(raw_latency)
        except ValueError as exc:
            raise InvalidGatewayControlError(
                "X-LLM-Max-Latency must be an integer number of milliseconds.",
                param="X-LLM-Max-Latency",
            ) from exc
        if max_latency_ms <= 0:
            raise InvalidGatewayControlError(
                "X-LLM-Max-Latency must be positive.", param="X-LLM-Max-Latency"
            )

    return ControlLayer(
        quality=quality,
        privacy=privacy,
        max_cost=max_cost,
        max_latency_ms=max_latency_ms,
        risk=_enum_header(get("risk"), Risk, "X-LLM-Risk"),
        validation=_validation_header(get("validation")),
    )


def _enum_header(raw: str | None, enum_cls: Any, header_name: str) -> Any:
    if raw is None:
        return None
    try:
        return enum_cls(raw.lower())
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise InvalidGatewayControlError(
            f"{header_name} must be one of: {allowed}.", param=header_name
        ) from exc


def _validation_header(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.lower()
    if value not in ValidationMode.ALL:
        allowed = ", ".join(sorted(ValidationMode.ALL))
        raise InvalidGatewayControlError(
            f"X-LLM-Validation must be one of: {allowed}.", param="X-LLM-Validation"
        )
    return value
