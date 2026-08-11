"""The streaming response path for both endpoints (§4).

Kept separate from :mod:`gateway.api.generation` because the ordering rules
differ: a non-streaming request may fail with any status right up to the last
moment, while a streaming one commits its status the instant headers go out.
Everything that could legitimately produce a 4xx -- validation, planning,
eligibility, reservation -- must therefore finish first.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from gateway.api.serializers import error_for, gateway_block
from gateway.api.sse import ChatStreamWriter, ResponsesStreamWriter
from gateway.domain.enums import Endpoint
from gateway.domain.requests import CanonicalRequest
from gateway.services.orchestrator import StreamEvent, StreamingOrchestrator
from gateway.services.planning import PlanningResult
from gateway.services.streaming import StreamPlan

logger = logging.getLogger(__name__)

#: Headers that stop proxies buffering or caching an event stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _usage_payload(event: StreamEvent) -> dict[str, int]:
    outcome = event.outcome
    result = outcome.result if outcome else None
    prompt = (result.usage.prompt_tokens if result else 0) or 0
    completion = (result.usage.completion_tokens if result else 0) or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


async def stream_response(
    *,
    canonical: CanonicalRequest,
    planning: PlanningResult,
    stream_plan: StreamPlan,
    orchestrator: StreamingOrchestrator,
    session: Session,
    scopes: list[str],
    deadline_at: datetime,
    disclose: bool,
) -> StreamingResponse:
    """Build the SSE response for whichever protocol the caller used."""
    is_chat = canonical.endpoint is Endpoint.CHAT_COMPLETIONS
    generator = (_chat_events if is_chat else _responses_events)(
        canonical=canonical,
        planning=planning,
        stream_plan=stream_plan,
        orchestrator=orchestrator,
        scopes=scopes,
        deadline_at=deadline_at,
        disclose=disclose,
    )
    del session
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )


async def _chat_events(
    *,
    canonical: CanonicalRequest,
    planning: PlanningResult,
    stream_plan: StreamPlan,
    orchestrator: StreamingOrchestrator,
    scopes: list[str],
    deadline_at: datetime,
    disclose: bool,
) -> AsyncIterator[str]:
    """Chat Completions SSE: bare data frames terminated by ``[DONE]``."""
    writer = ChatStreamWriter(model=canonical.requested_model, request_id=canonical.request_id)
    yield writer.role()

    async for event in orchestrator.run_stream(
        canonical,
        planning,
        scopes=scopes,
        deadline_at=deadline_at,
        buffered=stream_plan.buffers,
    ):
        if event.delta:
            yield writer.delta(event.delta)
            continue

        outcome = event.outcome
        if outcome is None:  # pragma: no cover - defensive
            continue

        if outcome.succeeded:
            yield writer.finish(outcome.result.finish_reason if outcome.result else "stop")
            yield writer.usage(
                _usage_payload(event),
                gateway_block(outcome, request_id=canonical.request_id, disclose=disclose),
            )
        else:
            # Status was committed with the headers, so the failure is in-band.
            yield writer.error(event.failure or error_for(outcome), request_id=canonical.request_id)

    yield writer.done()


async def _responses_events(
    *,
    canonical: CanonicalRequest,
    planning: PlanningResult,
    stream_plan: StreamPlan,
    orchestrator: StreamingOrchestrator,
    scopes: list[str],
    deadline_at: datetime,
    disclose: bool,
) -> AsyncIterator[str]:
    """Responses SSE: named ``response.*`` events."""
    writer = ResponsesStreamWriter(model=canonical.requested_model, request_id=canonical.request_id)
    yield writer.created()

    collected: list[str] = []

    async for event in orchestrator.run_stream(
        canonical,
        planning,
        scopes=scopes,
        deadline_at=deadline_at,
        buffered=stream_plan.buffers,
    ):
        if event.delta:
            collected.append(event.delta)
            yield writer.delta(event.delta)
            continue

        outcome = event.outcome
        if outcome is None:  # pragma: no cover - defensive
            continue

        if outcome.succeeded:
            yield writer.completed(
                "".join(collected),
                _usage_payload(event),
                gateway_block(outcome, request_id=canonical.request_id, disclose=disclose),
            )
        else:
            yield writer.error(event.failure or error_for(outcome), request_id=canonical.request_id)


def stream_debug_block(stream_plan: StreamPlan) -> dict[str, Any]:
    """Streaming decisions worth surfacing to an authorized caller."""
    return {
        "stream_mode": stream_plan.mode.value,
        "buffering_validators": list(stream_plan.buffering_validators),
    }
