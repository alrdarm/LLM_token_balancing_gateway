"""Application factory.

``create_app`` is a factory rather than a module-level singleton so tests can
build isolated instances with their own settings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from gateway import __version__
from gateway.api import health
from gateway.api.errors import error_response, handle_not_found, handle_unexpected_error
from gateway.api.request_id import RequestIDMiddleware, request_id_of
from gateway.config import Settings, get_settings
from gateway.telemetry.logging import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop process-wide resources.

    Empty in M0; database engines and provider clients attach here.
    """
    yield


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Wrap Starlette HTTP exceptions in the spec envelope.

    M0 only distinguishes 404. The full status and code taxonomy lands in M2.
    """
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover - defensive
        return await handle_unexpected_error(request, exc)

    if exc.status_code == 404:
        return await handle_not_found(request, exc)

    return error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
        error_type="invalid_request_error",
        code="invalid_request",
        request_id=request_id_of(request),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="LLM Token-Balancing Gateway",
        version=__version__,
        summary="OpenAI-compatible gateway with routing, budgets, and output validation.",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Outermost middleware: every response, including those built by exception
    # handlers, must carry the request ID.
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router)

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, handle_unexpected_error)

    return app
