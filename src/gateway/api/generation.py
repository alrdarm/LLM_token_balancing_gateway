"""Generation endpoints (§3).

Both endpoints run one pipeline: authenticate, validate, resolve controls,
normalize, claim idempotency, plan, orchestrate, serialize. They differ only in
which schema they validate and which serializer they use, which is what keeps
two wire formats from diverging in behaviour.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from gateway.api.auth import SCOPE_DEBUG, AuthenticatedClient
from gateway.api.dependencies import (
    get_client,
    get_session,
    get_settings_dep,
    read_json_body,
    resolve_request_controls,
)
from gateway.api.errors import redacted_validation_message
from gateway.api.request_id import request_id_of
from gateway.api.schemas import ChatCompletionRequest, ResponsesRequest
from gateway.api.serializers import error_for, to_chat_completion, to_response
from gateway.api.streaming_endpoints import stream_response
from gateway.config import Settings
from gateway.domain.enums import RequestState
from gateway.domain.errors import (
    GatewayError,
    IdempotencyConflictError,
    InvalidGatewayControlError,
    InvalidRequestError,
    ModelNotFoundError,
    NoProviderAvailableError,
)
from gateway.domain.requests import SELECTORS, CanonicalRequest, GatewayControls
from gateway.persistence.models import Request as RequestRow
from gateway.persistence.repositories import ModelRepository
from gateway.services import idempotency as idempotency_service
from gateway.services import streaming as streaming_service
from gateway.services.normalizer import (
    normalize_chat_request,
    normalize_responses_request,
)
from gateway.services.orchestrator import (
    OrchestrationResult,
    Orchestrator,
    StreamingOrchestrator,
)
from gateway.services.planning import build_plan

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generation"])


def _validate_body[BodyT: BaseModel](model_cls: type[BodyT], body: dict[str, Any]) -> BodyT:
    """Validate a body, converting pydantic errors into the gateway envelope.

    Pydantic's own error payload can echo submitted values, which for these
    endpoints means prompt text, so only the field location survives.
    """
    try:
        return model_cls.model_validate(body)
    except ValidationError as exc:
        details = exc.errors()
        first: dict[str, Any] = dict(details[0]) if details else {}
        location = [str(part) for part in first.get("loc", ())]
        param = ".".join(location) or None
        message = redacted_validation_message(first, param)

        error_cls: type[GatewayError] = (
            InvalidGatewayControlError
            if param and param.startswith("gateway")
            else InvalidRequestError
        )
        raise error_cls(message, param=param) from exc


def _check_model_exists(session: Session, requested_model: str) -> None:
    """Reject an unknown explicit model ID with 404 (§9).

    Selectors are not looked up: they resolve at routing time. An explicit ID
    that does not exist is a caller error worth reporting immediately rather
    than surfacing later as "no eligible route".
    """
    if requested_model in SELECTORS:
        return

    if ModelRepository(session).get(requested_model) is None:
        raise ModelNotFoundError(
            f"Unknown model: {requested_model}. "
            "Use GET /v1/models to list available models and selectors.",
            param="model",
        )


def _prepare(
    *,
    request: Request,
    session: Session,
    settings: Settings,
    client: AuthenticatedClient,
    requested_model: str,
    body_controls: Any,
) -> GatewayControls:
    """Run the checks and control resolution both endpoints share."""
    _check_model_exists(session, requested_model)

    controls = resolve_request_controls(
        settings=settings,
        client=client,
        headers=dict(request.headers),
        body_controls=body_controls,
        requested_model=requested_model,
    )

    if controls.debug and not client.has_scope(SCOPE_DEBUG):
        # §2: debug requires scope. Denying loudly beats silently downgrading,
        # because a caller relying on debug output would otherwise get a
        # response that looks complete but omits route and cost detail.
        client.require_scope(SCOPE_DEBUG)

    return controls


def _report_ignored(request: Request, canonical: CanonicalRequest) -> None:
    """§1 allows ignoring safely ignorable fields, but only *with telemetry*."""
    if canonical.ignored_fields:
        logger.info(
            "Ignored unsupported request fields",
            extra={"event": "fields_ignored", "path": request.url.path},
        )


def _controls_json(controls: GatewayControls) -> dict[str, Any]:
    """Operational controls only -- never metadata that could carry content."""
    return {
        "quality": controls.quality.value,
        "privacy": controls.privacy.value,
        "max_cost": str(controls.max_cost) if controls.max_cost is not None else None,
        "max_latency_ms": controls.max_latency_ms,
        "max_attempts": controls.max_attempts,
        "allow_fallback": controls.allow_fallback,
        "validation": controls.validation,
        "dry_run": controls.dry_run,
    }


def _persist_request(session: Session, canonical: CanonicalRequest, deadline_at: datetime) -> None:
    """Record the request before any billable work begins.

    Written first so an in-flight request is always visible: a reservation
    references its request, and the reconciler needs both to decide what an
    orphaned hold means.
    """
    session.add(
        RequestRow(
            id=canonical.request_id,
            client_id=canonical.client_id,
            endpoint=canonical.endpoint.value,
            requested_model=canonical.requested_model,
            normalized_controls_json=_controls_json(canonical.controls),
            state=RequestState.READY.value,
            deadline_at=deadline_at,
            input_hash=canonical.input_hash,
            estimated_input_tokens=canonical.estimated_input_tokens,
            max_cost=canonical.controls.max_cost,
            idempotency_key=canonical.idempotency_key,
        )
    )
    session.flush()


def _budget_scopes(canonical: CanonicalRequest) -> list[str]:
    """Budget scopes this request draws on."""
    return ["global", f"client:{canonical.client_id}"]


def _dry_run_document(canonical: CanonicalRequest, session: Session) -> dict[str, Any]:
    """§2: ``dry_run`` makes the generation endpoint behave as inspection.

    Reuses the inspection serializer, so a dry run and ``/route/inspect`` cannot
    describe the same request differently -- and, like inspection, it reserves
    no budget and calls no provider.
    """
    from gateway.api.inspect import _serialize

    result = build_plan(session, canonical)
    return _serialize(result, canonical, request_id=canonical.request_id, disclose=False)


async def _stream(
    request: Request,
    session: Session,
    canonical: CanonicalRequest,
    *,
    disclose: bool,
) -> StreamingResponse:
    """Serve a streaming request (§4).

    Everything that could still produce a 4xx -- planning, eligibility, the
    buffer-or-reject decision, and the budget reservation inside the
    orchestrator -- happens before the response is returned, because once
    headers are committed the status can no longer change.
    """
    orchestrator: StreamingOrchestrator | None = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:  # pragma: no cover - defensive
        raise NoProviderAvailableError(
            "No provider adapter is currently available to serve this request."
        )

    deadline_at = datetime.now(UTC) + timedelta(
        milliseconds=canonical.controls.max_latency_ms or 60_000
    )

    _persist_request(session, canonical, deadline_at)
    planning = build_plan(session, canonical)

    # Raises validation_requires_buffering (400) when a judge gate is planned,
    # which must happen before any byte is committed.
    stream_plan = streaming_service.decide(
        canonical, planning.features, planning.plan.validation_plan
    )
    session.commit()

    return await stream_response(
        canonical=canonical,
        planning=planning,
        stream_plan=stream_plan,
        orchestrator=orchestrator,
        session=session,
        scopes=_budget_scopes(canonical),
        deadline_at=deadline_at,
        disclose=disclose,
    )


async def _run(
    request: Request,
    session: Session,
    canonical: CanonicalRequest,
) -> OrchestrationResult:
    """Claim idempotency, persist, plan, and orchestrate."""
    orchestrator: Orchestrator | None = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise NoProviderAvailableError(
            "No provider adapter is currently available to serve this request."
        )

    deadline_at = datetime.now(UTC) + timedelta(
        milliseconds=canonical.controls.max_latency_ms or 60_000
    )

    # The request row is written first because the idempotency record
    # references it. Both live in the same uncommitted transaction, so a
    # conflict rolls back the row as well and leaves nothing behind -- and the
    # claim still precedes every *billable* step, which is what §1 requires.
    _persist_request(session, canonical, deadline_at)

    claim = None
    if canonical.idempotency_key and not canonical.stream:
        # §1 applies idempotency to non-stream POSTs. Claimed before any
        # provider call, so a duplicate cannot start a second invocation.
        claim = idempotency_service.claim(
            session,
            client_id=canonical.client_id,
            key=canonical.idempotency_key,
            input_hash=canonical.input_hash,
            request_id=canonical.request_id,
        )
        if claim.is_replay:
            logger.info("Replayed an idempotent request", extra={"event": "idempotency_replay"})
            # §6 stores a *reference*, not the body, and v0.1 does not retain
            # responses. Saying the original already completed is honest;
            # fabricating a body would not be.
            raise IdempotencyConflictError(
                "This Idempotency-Key already completed. Responses are not "
                "retained for replay in v0.1.",
                param="Idempotency-Key",
            )

    planning = build_plan(session, canonical)
    session.commit()

    outcome = await orchestrator.run(
        canonical,
        planning,
        scopes=_budget_scopes(canonical),
        deadline_at=deadline_at,
    )

    if claim is not None:
        if outcome.succeeded:
            idempotency_service.complete(
                session, claim.record_id, response_ref=canonical.request_id
            )
        else:
            # Free the key so a genuine retry is not blocked by a failure.
            idempotency_service.fail(session, claim.record_id)
        session.commit()

    return outcome


@router.post("/v1/chat/completions", summary="Chat Completions-compatible generation")
async def create_chat_completion(
    request: Request,
    body: dict[str, Any] = Depends(read_json_body),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    client: AuthenticatedClient = Depends(get_client),
) -> Any:
    """Generate a Chat Completions response, streamed or complete."""
    payload = _validate_body(ChatCompletionRequest, body)
    controls = _prepare(
        request=request,
        session=session,
        settings=settings,
        client=client,
        requested_model=payload.model,
        body_controls=payload.gateway,
    )
    canonical = normalize_chat_request(
        payload,
        raw_body=body,
        request_id=request_id_of(request),
        client_id=client.client_id,
        controls=controls,
        hash_key=settings.hash_key,
        idempotency_key=request.headers.get("idempotency-key"),
    )
    _report_ignored(request, canonical)

    if controls.dry_run:
        return _dry_run_document(canonical, session)

    if canonical.stream:
        return await _stream(request, session, canonical, disclose=client.has_scope(SCOPE_DEBUG))

    outcome = await _run(request, session, canonical)
    if not outcome.succeeded:
        raise error_for(outcome)

    return to_chat_completion(canonical, outcome, disclose=client.has_scope(SCOPE_DEBUG))


@router.post("/v1/responses", summary="Responses-compatible generation")
async def create_response(
    request: Request,
    body: dict[str, Any] = Depends(read_json_body),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    client: AuthenticatedClient = Depends(get_client),
) -> Any:
    """Generate a Responses-API response, streamed or complete."""
    payload = _validate_body(ResponsesRequest, body)
    controls = _prepare(
        request=request,
        session=session,
        settings=settings,
        client=client,
        requested_model=payload.model,
        body_controls=payload.gateway,
    )
    canonical = normalize_responses_request(
        payload,
        raw_body=body,
        request_id=request_id_of(request),
        client_id=client.client_id,
        controls=controls,
        hash_key=settings.hash_key,
        idempotency_key=request.headers.get("idempotency-key"),
    )
    _report_ignored(request, canonical)

    if controls.dry_run:
        return _dry_run_document(canonical, session)

    if canonical.stream:
        return await _stream(request, session, canonical, disclose=client.has_scope(SCOPE_DEBUG))

    outcome = await _run(request, session, canonical)
    if not outcome.succeeded:
        raise error_for(outcome)

    return to_response(canonical, outcome, disclose=client.has_scope(SCOPE_DEBUG))
