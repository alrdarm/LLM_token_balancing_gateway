"""Health endpoint contract."""

from __future__ import annotations

import pytest

from gateway.api.health import CheckOutcome, readiness


def test_live_reports_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_live_does_not_consult_dependencies(client):
    """Liveness must stay green while a dependency is down, or the process
    gets killed instead of drained."""

    async def failing() -> CheckOutcome:
        return CheckOutcome(ok=False, detail="down")

    readiness.register("database", failing)
    assert client.get("/health/live").status_code == 200


def test_ready_with_no_probes_reports_the_empty_set(client):
    """M0 registers no probes; the response says so rather than implying checks ran."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": []}


def test_ready_passes_when_all_probes_pass(client):
    async def ok() -> CheckOutcome:
        return CheckOutcome(ok=True)

    readiness.register("database", ok)
    readiness.register("registry", ok)

    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == [
        {"name": "database", "ok": True},
        {"name": "registry", "ok": True},
    ]


def test_ready_fails_with_503_when_a_probe_fails(client):
    async def ok() -> CheckOutcome:
        return CheckOutcome(ok=True)

    async def failing() -> CheckOutcome:
        return CheckOutcome(ok=False, detail="pending migrations")

    readiness.register("database", ok)
    readiness.register("migrations", failing)

    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "not_ready"
    assert body["gateway"]["retryable"] is True
    assert "migrations" in body["error"]["message"]


def test_ready_failure_does_not_leak_probe_detail(client):
    """Health is unauthenticated: names are operational, detail is not disclosed."""

    async def failing() -> CheckOutcome:
        return CheckOutcome(ok=False, detail="postgres://user:pw@internal-host/db refused")

    readiness.register("database", failing)

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert "internal-host" not in response.text
    assert "pw" not in response.json()["error"]["message"]


def test_raising_probe_is_a_failure_not_a_500(client):
    async def exploding() -> CheckOutcome:
        raise RuntimeError("connection pool exhausted at /srv/gateway/db.py")

    readiness.register("database", exploding)

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert "/srv/gateway" not in response.text


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
def test_health_requires_no_authentication(client, path):
    """The spec authenticates all *non-health* endpoints."""
    assert client.get(path).status_code in (200, 503)
