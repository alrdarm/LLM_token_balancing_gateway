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
