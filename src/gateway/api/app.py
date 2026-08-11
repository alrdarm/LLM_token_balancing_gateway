"""Application factory.

``create_app`` is a factory rather than a module-level singleton so tests can
build isolated instances with their own settings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from gateway import __version__
from gateway.api import generation, health, inspect, models_endpoint
from gateway.api.errors import (
    error_response,
    handle_gateway_error,
    handle_not_found,
    handle_request_validation_error,
    handle_unexpected_error,
)
from gateway.api.request_id import RequestIDMiddleware, request_id_of
from gateway.config import Settings, get_settings
from gateway.domain.errors import GatewayError
from gateway.persistence.engine import create_db_engine, create_session_factory
from gateway.persistence.migrations_config import alembic_config
from gateway.persistence.readiness import register_persistence_probes
from gateway.providers.base import AdapterRegistry
from gateway.providers.fake import FakeProvider
from gateway.services.orchestrator import StreamingOrchestrator
from gateway.services.resilience import CircuitBreaker
from gateway.telemetry.logging import configure_logging
from gateway.validators.deterministic import default_registry


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop process-wide resources.

    The engine is created here and disposed on shutdown so connections are not
    leaked across reloads. Migrations are deliberately *not* run on startup:
    §11 makes pending migrations a readiness failure, which means an operator
    applies them explicitly rather than a booting process mutating the schema
    of a database other replicas are already serving.
    """
    settings: Settings = app.state.settings
    engine = create_db_engine(settings.database_url, echo=settings.database_echo)

    app.state.engine = engine
    session_factory = create_session_factory(engine)
    app.state.session_factory = session_factory

    # Adapters are wired here rather than at import time so a deployment can
    # register a different set without touching the app factory. v0.1 ships the
    # deterministic fake; real adapters are configured per deployment.
    adapters = AdapterRegistry()
    adapters.register(FakeProvider())

    app.state.adapters = adapters
    app.state.breaker = CircuitBreaker()
    app.state.orchestrator = StreamingOrchestrator(
        session_factory=session_factory,
        adapters=adapters,
        validators=default_registry(),
        breaker=app.state.breaker,
        hash_key=settings.hash_key,
    )

    register_persistence_probes(health.readiness.register, engine, alembic_config())

    try:
        yield
    finally:
        engine.dispose()


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
    app.include_router(models_endpoint.router)
    app.include_router(generation.router)
    app.include_router(inspect.router)

    # Domain errors carry their own §9 status and code; register them before
    # the catch-all so they are never flattened into a 500.
    app.add_exception_handler(GatewayError, handle_gateway_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(ValidationError, handle_request_validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, handle_unexpected_error)

    return app
