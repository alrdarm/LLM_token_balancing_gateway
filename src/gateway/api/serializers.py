"""Endpoint serializers (§3, §5).

Pure adapters from one canonical outcome to two wire shapes. They read
:class:`~gateway.services.orchestrator.OrchestrationResult` and nothing else --
no database, no provider object -- which is what keeps two API formats from
leaking back into the core.

Disclosure follows §1: the ``gateway`` block always carries the request ID, but
route, cost, attempt, and latency detail appear only for callers holding the
``debug`` scope. Cost crosses the wire as a **string** so it cannot be silently
turned into a float by a JSON parser.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from gateway.domain.enums import RequestState
from gateway.domain.errors import (
    DeadlineExceededError,
    GatewayError,
    InternalError,
    NoEligibleRouteError,
    ProviderError,
)
from gateway.domain.requests import CanonicalRequest
from gateway.services.orchestrator import OrchestrationResult

#: Terminal states that are failures, and the §9 error each maps to.
_FAILURE_ERRORS: dict[RequestState, type[GatewayError]] = {
    RequestState.REJECTED_NO_ROUTE: NoEligibleRouteError,
    RequestState.EXPIRED: DeadlineExceededError,
    RequestState.FAILED_EXHAUSTED: ProviderError,
    RequestState.FAILED_PARTIAL: ProviderError,
    RequestState.FAILED: ProviderError,
}

_FAILURE_MESSAGES: dict[RequestState, str] = {
    RequestState.REJECTED_NO_ROUTE: (
        "No model satisfies this request's privacy, capability, quality, cost, "
        "and deadline constraints."
    ),
    RequestState.EXPIRED: "The request deadline elapsed before completion.",
    RequestState.FAILED_EXHAUSTED: (
        "Every eligible model failed to produce a response that passed validation."
    ),
    RequestState.FAILED_PARTIAL: (
        "The response failed after output had already been sent, so it could not "
        "be retried on another model."
    ),
    RequestState.FAILED: "The request failed.",
}


def error_for(outcome: OrchestrationResult) -> GatewayError:
    """Map a failed terminal state onto its §9 error.

    The orchestrator reports states; this decides what a client is told. Keeping
    the mapping here means the state machine never has to know about HTTP.
    """
    if outcome.state is RequestState.REJECTED_BUDGET:
        from gateway.domain.errors import BudgetExceededError

        return BudgetExceededError(
            "This request would exceed its cost ceiling or a scoped budget.",
            param="gateway.max_cost",
        )

    error_cls = _FAILURE_ERRORS.get(outcome.state, InternalError)
    message = _FAILURE_MESSAGES.get(outcome.state, "The request failed.")
    return error_cls(message)


def gateway_block(
    outcome: OrchestrationResult, *, request_id: str, disclose: bool
) -> dict[str, Any]:
    """The ``gateway`` summary attached to a successful response (§3)."""
    block: dict[str, Any] = {"request_id": request_id}

    if not disclose:
        # §1: route and cost summary only when authorized.
        return block

    block["resolved_model"] = outcome.resolved_model
    block["attempts"] = len(outcome.attempts)
    block["validation"] = (
        outcome.validation.aggregate.value if outcome.validation is not None else None
    )
    block["cost"] = {"currency": "USD", "actual": str(outcome.total_cost)}
    block["latency_ms"] = outcome.latency_ms
    return block


def _usage(outcome: OrchestrationResult) -> dict[str, int]:
    result = outcome.result
    prompt = result.usage.prompt_tokens if result else None
    completion = result.usage.completion_tokens if result else None
    prompt = prompt or 0
    completion = completion or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def to_chat_completion(
    request: CanonicalRequest,
    outcome: OrchestrationResult,
    *,
    disclose: bool,
) -> dict[str, Any]:
    """Render a Chat Completions response (§3)."""
    result = outcome.result
    return {
        "id": f"chatcmpl_{uuid4()}",
        "object": "chat.completion",
        "created": _created(),
        "model": request.requested_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text if result else "",
                },
                "finish_reason": result.finish_reason if result else "stop",
            }
        ],
        "usage": _usage(outcome),
        "gateway": gateway_block(outcome, request_id=request.request_id, disclose=disclose),
    }


def to_response(
    request: CanonicalRequest,
    outcome: OrchestrationResult,
    *,
    disclose: bool,
) -> dict[str, Any]:
    """Render a Responses-API response (§3)."""
    result = outcome.result
    text = result.text if result else ""
    return {
        "id": f"resp_{uuid4()}",
        "object": "response",
        "created_at": _created(),
        "model": request.requested_model,
        "status": "completed",
        "output": [
            {
                "id": f"msg_{uuid4()}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": _usage(outcome)["prompt_tokens"],
            "output_tokens": _usage(outcome)["completion_tokens"],
            "total_tokens": _usage(outcome)["total_tokens"],
        },
        "gateway": gateway_block(outcome, request_id=request.request_id, disclose=disclose),
    }


def _created() -> int:
    from datetime import UTC, datetime

    return int(datetime.now(UTC).timestamp())
