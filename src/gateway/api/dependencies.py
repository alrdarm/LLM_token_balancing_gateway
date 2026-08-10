"""Shared endpoint dependencies.

Authentication, content-type enforcement, and canonical normalization are the
same for both generation endpoints, so they live here rather than being
duplicated (and drifting) across two routers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from gateway.api.auth import AuthenticatedClient, authenticate
from gateway.config import Settings
from gateway.domain.errors import (
    InvalidRequestError,
    RequestTooLargeError,
    UnsupportedMediaTypeError,
)
from gateway.domain.requests import GatewayControls
from gateway.services.controls import (
    ControlLayer,
    layer_from_headers,
    resolve_controls,
)

#: Payload ceiling applied before parsing (§11: apply size limits early).
MAX_BODY_BYTES = 4 * 1024 * 1024


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_session_factory(request: Request) -> sessionmaker[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    return factory


def get_session(
    factory: sessionmaker[Session] = Depends(get_session_factory),
) -> Iterator[Session]:
    """Yield a session scoped to one request."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def require_json_content_type(request: Request) -> None:
    """Enforce ``application/json`` (§1: unsupported type is 415)."""
    header = request.headers.get("content-type", "")
    media_type = header.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise UnsupportedMediaTypeError(
            "Content-Type must be application/json.", param="Content-Type"
        )


async def read_json_body(request: Request) -> dict[str, Any]:
    """Read and parse the body, enforcing the size limit first.

    Size is checked before parsing so an oversized payload is rejected without
    being materialised into Python objects.
    """
    require_json_content_type(request)

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise RequestTooLargeError(
            f"Request body exceeds the {MAX_BODY_BYTES} byte limit.",
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        # The parser's message can quote body content, so it is not surfaced.
        raise InvalidRequestError("Request body is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise InvalidRequestError("Request body must be a JSON object.")
    return parsed


def get_client(
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> AuthenticatedClient:
    """Authenticate the caller (§1: all non-health endpoints)."""
    return authenticate(
        session,
        request.headers.get("authorization"),
        hash_key=settings.hash_key,
    )


def deployment_layer(settings: Settings) -> ControlLayer:
    """The deployment's own control floor -- the highest-precedence layer.

    Only values the deployment actually pins appear here; an unset field must
    stay ``None`` so it does not silently outrank a caller's stricter choice.
    """
    return ControlLayer(
        privacy=settings.deployment_privacy_floor,
        max_cost=settings.default_max_cost,
        max_latency_ms=settings.default_max_latency_ms,
    )


def client_layer(client: AuthenticatedClient) -> ControlLayer:
    """Control overrides attached to the credential."""
    overrides = client.control_overrides
    if not overrides:
        return ControlLayer()

    from gateway.api.schemas import GatewayControlsPayload
    from gateway.services.controls import layer_from_body

    # Validated through the same schema as a request body, so a bad override
    # cannot inject a value the wire format would have rejected.
    return layer_from_body(GatewayControlsPayload.model_validate(overrides))


def resolve_request_controls(
    *,
    settings: Settings,
    client: AuthenticatedClient,
    headers: dict[str, str],
    body_controls: Any,
    requested_model: str,
) -> GatewayControls:
    """Apply the full precedence chain for one request (§2)."""
    from gateway.services.controls import layer_from_body

    return resolve_controls(
        deployment=deployment_layer(settings),
        client=client_layer(client),
        headers=layer_from_headers(headers),
        body=layer_from_body(body_controls),
        alias_defaults=ControlLayer(),
        system_defaults=GatewayControls(
            max_cost=settings.default_max_cost,
            max_latency_ms=settings.default_max_latency_ms,
        ),
        requested_model=requested_model,
    )
