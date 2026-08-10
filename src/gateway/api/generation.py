"""Generation endpoints (§3).

Both endpoints run the same pipeline -- authenticate, validate, resolve
controls, normalize -- and then hand the canonical request to the orchestrator.

**M2 has no orchestrator and no provider adapters**, so the pipeline ends in
``503 no_provider_available``. That is the honest §9 code for the actual
condition rather than a placeholder: nothing is registered that could serve the
request. M4 registers adapters and M5 wires the state machine, at which point
this becomes a real invocation without the surrounding contract changing.

Everything before that point is complete and observable, which is what the M2
exit gate covers.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

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
from gateway.config import Settings
from gateway.domain.errors import (
    GatewayError,
    InvalidGatewayControlError,
    InvalidRequestError,
    ModelNotFoundError,
    NoProviderAvailableError,
)
from gateway.domain.requests import SELECTORS, CanonicalRequest, GatewayControls
from gateway.persistence.repositories import ModelRepository
from gateway.services.normalizer import (
    normalize_chat_request,
    normalize_responses_request,
)

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


def _not_yet_routable(canonical: CanonicalRequest) -> NoProviderAvailableError:
    """The terminal state of the M2 pipeline."""
    logger.info(
        "No provider adapter registered",
        extra={"event": "no_provider_available", "path": canonical.endpoint.value},
    )
    return NoProviderAvailableError(
        "No provider adapter is currently available to serve this request.",
    )


@router.post("/v1/chat/completions", summary="Chat Completions-compatible generation")
async def create_chat_completion(
    request: Request,
    body: dict[str, Any] = Depends(read_json_body),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    client: AuthenticatedClient = Depends(get_client),
) -> dict[str, Any]:
    """Accept, validate, and normalize a Chat Completions request."""
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
    raise _not_yet_routable(canonical)


@router.post("/v1/responses", summary="Responses-compatible generation")
async def create_response(
    request: Request,
    body: dict[str, Any] = Depends(read_json_body),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    client: AuthenticatedClient = Depends(get_client),
) -> dict[str, Any]:
    """Accept, validate, and normalize a Responses request."""
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
    raise _not_yet_routable(canonical)
