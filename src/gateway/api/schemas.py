"""Wire schemas for both generation endpoints (§3).

Two validation postures, deliberately different:

* The ``gateway`` control block is **strict** -- an unknown key there is a
  typo in something that governs privacy or spend, and silently ignoring it
  would apply weaker controls than the caller believed they set.
* The surrounding OpenAI-compatible body is **tolerant** -- §1 allows safely
  ignorable standard fields to be ignored with telemetry, because SDKs send
  fields the gateway has no opinion about.

Money is parsed from string or int through ``Decimal``. ``float`` is rejected:
JSON floats cannot represent decimal cents exactly.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from gateway.persistence.types import MoneyError, to_money

#: Fields the gateway understands on the chat body. Anything else is reported
#: as ignored rather than rejected.
KNOWN_CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "n",
        "stream",
        "stream_options",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "metadata",
        "gateway",
    }
)

KNOWN_RESPONSES_FIELDS = frozenset(
    {
        "model",
        "instructions",
        "input",
        "max_output_tokens",
        "temperature",
        "top_p",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "text",
        "reasoning",
        "metadata",
        "user",
        "gateway",
    }
)

#: Rejected outright rather than ignored: each would change what the gateway
#: does in a way it cannot honour (§3).
REJECTED_RESPONSES_FIELDS = {
    "background": "Background mode is not supported.",
    "previous_response_id": "Stored conversation state is not supported.",
    "store": "Provider-side response storage is not supported.",
}

MAX_METADATA_ENTRIES = 16
MAX_METADATA_KEY_LENGTH = 64
MAX_METADATA_VALUE_LENGTH = 512


def _parse_money(value: Any) -> Any:
    """Coerce a JSON scalar to an exact ``Decimal``."""
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise ValueError('monetary values must be sent as a string to stay exact, e.g. "0.05"')
    if isinstance(value, str | int):
        try:
            return to_money(value)
        except MoneyError as exc:
            raise ValueError(str(exc)) from exc
    raise ValueError("must be a decimal string")


MoneyField = Annotated[Decimal, BeforeValidator(_parse_money)]


class GatewayControlsPayload(BaseModel):
    """The top-level ``gateway`` control block (§2)."""

    model_config = ConfigDict(extra="forbid")

    quality: Literal["economy", "standard", "high", "critical"] | None = None
    privacy: Literal["public", "confidential"] | None = None
    max_cost: MoneyField | None = None
    max_latency_ms: int | None = Field(default=None, gt=0)
    provider_allow: list[str] | None = None
    provider_deny: list[str] | None = None
    required_capabilities: (
        list[Literal["tools", "json_schema", "vision", "streaming", "reasoning"]] | None
    ) = None
    allow_fallback: bool | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    validation: Literal["auto", "none", "deterministic", "independent"] | None = None
    task_class: str | None = None
    risk: Literal["low", "medium", "high", "critical"] | None = None
    dry_run: bool | None = None
    metadata: dict[str, str] | None = None
    debug: bool | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> GatewayControlsPayload:
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must not be negative")
        if self.metadata is not None:
            _validate_metadata(self.metadata)
        if self.provider_allow is not None and self.provider_deny is not None:
            overlap = set(self.provider_allow) & set(self.provider_deny)
            if overlap:
                raise ValueError(f"provider_allow and provider_deny both list: {sorted(overlap)}")
        return self


def _validate_metadata(metadata: dict[str, str]) -> None:
    """Reject oversized metadata (§2).

    Metadata is caller-controlled and lands in telemetry, so it is bounded
    before it can bloat every log line or smuggle prompt text into storage.
    """
    if len(metadata) > MAX_METADATA_ENTRIES:
        raise ValueError(f"metadata supports at most {MAX_METADATA_ENTRIES} entries")
    for key, value in metadata.items():
        if len(key) > MAX_METADATA_KEY_LENGTH:
            raise ValueError(f"metadata key too long: {key[:20]}...")
        if len(value) > MAX_METADATA_VALUE_LENGTH:
            raise ValueError(f"metadata value too long for key: {key}")


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One Chat Completions message."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    #: Either a plain string or the content-part array form.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` body (§3)."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    n: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = None
    response_format: ResponseFormat | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None
    gateway: GatewayControlsPayload | None = None

    @model_validator(mode="after")
    def _check_semantics(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("max_tokens and max_completion_tokens are mutually exclusive")
        if self.n is not None and self.n != 1:
            # v0.1 rejects n != 1 (§3): multiple candidates would each need
            # their own validation and budget accounting.
            raise ValueError("n must be 1; multiple candidates are not supported in v0.1")
        if self.metadata is not None:
            _validate_metadata(self.metadata)
        return self


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ResponsesTextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    """``POST /v1/responses`` body (§3)."""

    model_config = ConfigDict(extra="allow")

    model: str
    instructions: str | None = None
    #: A bare string or the structured input-item array.
    input: str | list[dict[str, Any]]

    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    text: ResponsesTextConfig | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    user: str | None = None
    gateway: GatewayControlsPayload | None = None

    @model_validator(mode="after")
    def _check_semantics(self) -> ResponsesRequest:
        if self.metadata is not None:
            _validate_metadata(self.metadata)
        return self


# ---------------------------------------------------------------------------
# Responses out
# ---------------------------------------------------------------------------


class GatewayCost(BaseModel):
    """Monetary summary. Serialised as a string to stay exact on the wire."""

    currency: str = "USD"
    actual: str


class GatewaySummary(BaseModel):
    """The ``gateway`` block attached to successful responses (§3).

    Route and cost detail is disclosed only to authorized callers (§1), so the
    optional fields stay unset without the ``debug`` scope.
    """

    request_id: str
    resolved_model: str | None = None
    attempts: int | None = None
    validation: str | None = None
    cost: GatewayCost | None = None
    latency_ms: int | None = None


class ModelCard(BaseModel):
    """One entry in ``GET /v1/models``."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
