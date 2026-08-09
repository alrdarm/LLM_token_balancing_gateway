"""Liveness and readiness.

Liveness answers only "is this process running"; it must never depend on a
database or provider, or a transient dependency outage would get the process
killed rather than drained.

Readiness aggregates registered dependency probes. The spec requires readiness
to fail for pending migrations, no active policy, an empty registry, or no
required route -- none of which exist yet. Rather than assert a readiness the
skeleton cannot verify, M0 ships the registry with **zero** probes and reports
the empty set explicitly, so an operator can see what was actually checked.
Later milestones register their own probes here (M1 migrations, M3 policy and
model registry, M4 adapters).

Health endpoints are unauthenticated, so failure output names the failing
probes but never their diagnostic detail; detail goes to the logs.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from gateway.api.errors import error_response
from gateway.api.request_id import request_id_of

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """Result of one readiness probe."""

    ok: bool
    detail: str | None = None


CheckProbe = Callable[[], Awaitable[CheckOutcome]]


class ReadinessRegistry:
    """Named readiness probes, evaluated on every ``/health/ready`` call."""

    def __init__(self) -> None:
        self._probes: dict[str, CheckProbe] = {}

    def register(self, name: str, probe: CheckProbe) -> None:
        """Register ``probe`` under ``name``, replacing any existing probe."""
        self._probes[name] = probe

    def clear(self) -> None:
        """Drop all probes. Used by tests."""
        self._probes.clear()

    @property
    def names(self) -> tuple[str, ...]:
        """Registered probe names, in registration order."""
        return tuple(self._probes)

    async def evaluate(self) -> dict[str, CheckOutcome]:
        """Run every probe. A raising probe counts as a failure, not a 500."""
        results: dict[str, CheckOutcome] = {}
        for name, probe in self._probes.items():
            try:
                results[name] = await probe()
            except Exception:
                logger.exception("Readiness probe raised", extra={"check": name})
                results[name] = CheckOutcome(ok=False, detail="probe raised")
        return results


#: Process-wide registry. Milestones wire their probes in at app construction.
readiness = ReadinessRegistry()


@router.get("/health/live", summary="Process liveness")
async def health_live(request: Request) -> JSONResponse:
    """Report that the process is running and able to serve."""
    del request  # liveness consults nothing; the request ID header is middleware-owned
    return JSONResponse(status_code=200, content={"status": "live"})


@router.get("/health/ready", summary="Dependency readiness")
async def health_ready(request: Request) -> JSONResponse:
    """Report whether every registered dependency probe passes."""
    results = await readiness.evaluate()
    failed = sorted(name for name, outcome in results.items() if not outcome.ok)

    if failed:
        for name in failed:
            logger.warning(
                "Readiness probe failed",
                extra={"check": name, "event": "not_ready"},
            )
        return error_response(
            status_code=503,
            message=f"Dependencies not ready: {', '.join(failed)}.",
            error_type="gateway_availability_error",
            code="not_ready",
            request_id=request_id_of(request),
            retryable=True,
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "checks": [{"name": name, "ok": True} for name in results],
        },
    )
