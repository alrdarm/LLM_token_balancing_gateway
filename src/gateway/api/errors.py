"""Error envelope.

The spec mandates an OpenAI-style envelope that never exposes secrets, raw
provider payloads, internal paths, SQL, or prompt text. M0 ships only the
envelope and the two handlers a bare skeleton can actually hit; the full status
and code taxonomy lands with M2.

The unhandled-exception handler exists in M0 specifically so the skeleton never
returns a framework traceback, which would leak internal paths.
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from gateway.api.request_id import OUTBOUND_HEADER, request_id_of
from gateway.domain.errors import GatewayError
from gateway.telemetry.context import reset_request_id, set_request_id

logger = logging.getLogger(__name__)

GENERIC_INTERNAL_MESSAGE = "The gateway encountered an internal error."


def error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    request_id: str,
    param: str | None = None,
    retryable: bool = False,
) -> JSONResponse:
    """Build a spec-shaped error response."""
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        },
        "gateway": {
            "request_id": request_id,
            "retryable": retryable,
        },
    }
    # RequestIDMiddleware normally stamps the outbound header and replaces any
    # duplicate. It is set here too because Starlette's ServerErrorMiddleware
    # sits *outside* user middleware: a 500 it emits never passes through that
    # wrapper, so this is the only chance to attach the ID.
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={OUTBOUND_HEADER: request_id},
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a redacted 500 for any unhandled exception.

    The exception is logged (type and message only, per the JSON formatter) and
    deliberately not surfaced to the client.
    """
    request_id = request_id_of(request)

    # This handler runs in ServerErrorMiddleware, after RequestIDMiddleware has
    # already unwound its context. Re-bind the ID so the error log is still
    # correlated with the request that produced it.
    token = set_request_id(request_id)
    try:
        logger.exception("Unhandled gateway error", extra={"event": "internal_error"})
    finally:
        reset_request_id(token)

    return error_response(
        status_code=500,
        message=GENERIC_INTERNAL_MESSAGE,
        error_type="gateway_internal_error",
        code="internal_error",
        request_id=request_id,
    )


async def handle_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Return an enveloped 404 instead of Starlette's default body."""
    return error_response(
        status_code=404,
        message="Unknown endpoint.",
        error_type="invalid_request_error",
        code="not_found",
        request_id=request_id_of(request),
    )


async def handle_gateway_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a domain :class:`GatewayError` through the spec envelope.

    The domain raises; the API renders. This keeps §9's status/code mapping in
    one place instead of scattered across endpoints.
    """
    if not isinstance(exc, GatewayError):  # pragma: no cover - defensive
        return await handle_unexpected_error(request, exc)

    logger.info(
        "Request rejected",
        extra={"event": exc.code, "status_code": exc.status_code},
    )

    response = error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_type=exc.error_type,
        code=exc.code,
        request_id=request_id_of(request),
        param=exc.param,
        retryable=exc.retryable,
    )
    if exc.retry_after is not None:
        # §9: honour Retry-After when the wait is known.
        response.headers["Retry-After"] = str(exc.retry_after)
    return response


async def handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a schema rejection into the gateway envelope.

    Pydantic's default 422 body is neither the spec's shape nor its status, and
    its error entries can echo submitted values -- which for these endpoints
    means prompt text. Only the field location is surfaced.
    """
    param: str | None = None
    message = "Request body failed validation."

    errors = getattr(exc, "errors", None)
    if callable(errors):
        details = errors()
        if details:
            first = details[0]
            location = [str(part) for part in first.get("loc", ()) if part != "body"]
            param = ".".join(location) or None
            message = redacted_validation_message(first, param)

    is_control = param is not None and param.startswith("gateway")
    return error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        code="invalid_gateway_control" if is_control else "invalid_request",
        request_id=request_id_of(request),
        param=param,
    )


def redacted_validation_message(detail: dict[str, Any], param: str | None) -> str:
    """Build a message from the error type, never the submitted value."""
    reason = str(detail.get("msg", "is invalid"))
    # Pydantic prefixes custom errors with "Value error, "; drop the noise.
    reason = reason.removeprefix("Value error, ")
    where = param or "request body"
    return f"Invalid value for {where}: {reason}"
