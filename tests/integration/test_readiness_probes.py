"""Readiness now reflects real dependency state.

M0 shipped an empty probe registry, so ``/health/ready`` was green regardless of
whether anything worked. M1 registers the probes §11 requires: pending
migrations, an empty registry, and no active policy must each make readiness
fail.
"""

from __future__ import annotations

import pytest
from alembic import command
from fastapi.testclient import TestClient

from gateway.api.app import create_app
from gateway.api.health import readiness
from gateway.config import Settings
from gateway.persistence.migrations_config import alembic_config
from gateway.persistence.seed import seed_all
from gateway.persistence.unit_of_work import unit_of_work

pytestmark = pytest.mark.integration


@pytest.fixture
def app_settings(settings: Settings, database_url: str) -> Settings:
    return settings.model_copy(update={"database_url": database_url})


def _client(app_settings: Settings) -> TestClient:
    readiness.clear()
    return TestClient(create_app(app_settings), raise_server_exceptions=False)


def test_unmigrated_database_is_not_ready(app_settings: Settings, database_url: str):
    """Serving against a schema the code does not expect corrupts data (§11)."""
    with _client(app_settings) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "not_ready"
    assert "migrations" in body["error"]["message"]


def test_migrated_but_unseeded_database_is_not_ready(
    app_settings: Settings, migrated_engine, database_url: str
):
    """An empty registry means nothing is routable, so readiness must fail."""
    migrated_engine.dispose()
    with _client(app_settings) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    message = response.json()["error"]["message"]
    assert "registry" in message
    assert "active_policy" in message


def test_migrated_and_seeded_database_is_ready(
    app_settings: Settings, session_factory, migrated_engine
):
    with unit_of_work(session_factory) as uow:
        seed_all(uow.session)
    migrated_engine.dispose()

    with _client(app_settings) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {check["name"] for check in body["checks"]} == {
        "database",
        "migrations",
        "registry",
        "active_policy",
    }


def test_readiness_failure_does_not_leak_connection_details(
    app_settings: Settings, database_url: str
):
    """Health is unauthenticated: probe detail belongs in logs, not responses."""
    with _client(app_settings) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    # Credentials and host from the URL must not appear in the body.
    assert "gateway:gateway" not in response.text
    assert "password" not in response.text.lower()


def test_liveness_stays_green_when_dependencies_are_down(app_settings: Settings):
    """Liveness must not consult the database, or an outage kills the process
    instead of draining it."""
    with _client(app_settings) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_downgraded_database_is_not_ready(
    app_settings: Settings, migrated_engine, database_url: str
):
    """A rollback that outruns the code must be caught, not served."""
    migrated_engine.dispose()
    command.downgrade(alembic_config(database_url), "base")

    with _client(app_settings) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert "migrations" in response.json()["error"]["message"]
