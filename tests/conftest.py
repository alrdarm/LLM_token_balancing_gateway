"""Shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from gateway.api.app import create_app
from gateway.api.health import readiness
from gateway.config import Settings
from gateway.persistence.engine import create_db_engine, create_session_factory
from gateway.persistence.migrations_config import alembic_config

#: Set this to run the PostgreSQL half of the database matrix, e.g.
#: ``GATEWAY_TEST_POSTGRES_URL=postgresql+psycopg://user:pw@localhost:5432/db``.
#: The spec requires SQLite always and PostgreSQL on integration, so its absence
#: skips rather than fails.
POSTGRES_URL_ENV = "GATEWAY_TEST_POSTGRES_URL"


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore any developer ``.env``."""
    return Settings(_env_file=None, environment="local", log_level="INFO")


@pytest.fixture
def app(settings: Settings, sqlite_url: str) -> Iterator[FastAPI]:
    """A fresh app backed by a migrated SQLite database."""
    readiness.clear()
    app_settings = settings.model_copy(update={"database_url": sqlite_url})
    yield create_app(app_settings)
    readiness.clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Test client that surfaces server errors as responses, not exceptions."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Database matrix
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """A file-backed SQLite URL.

    File-backed rather than in-memory: WAL, ``BEGIN IMMEDIATE``, and the busy
    timeout only behave realistically against a real file, and those are what
    the budget algorithm depends on.
    """
    return f"sqlite+pysqlite:///{tmp_path / 'gateway-test.db'}"


def _postgres_url() -> str | None:
    return os.environ.get(POSTGRES_URL_ENV)


def _reset_postgres(url: str) -> None:
    """Drop and recreate the public schema so each test starts clean."""
    engine = create_db_engine(url)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def database_url(request: pytest.FixtureRequest, sqlite_url: str) -> str:
    """Every backend the spec requires.

    Parametrised so each persistence test runs on both, which is the M1 exit
    gate. PostgreSQL skips when no URL is configured rather than failing, so a
    default developer checkout still runs the SQLite half.
    """
    if request.param == "sqlite":
        return sqlite_url

    url = _postgres_url()
    if url is None:
        pytest.skip(f"{POSTGRES_URL_ENV} not set; skipping PostgreSQL matrix entry")
    _reset_postgres(url)
    return url


@pytest.fixture
def migrated_engine(database_url: str) -> Iterator[Engine]:
    """An engine whose schema was built by running the real migrations.

    Built via ``alembic upgrade`` rather than ``metadata.create_all`` on
    purpose: the exit gate is that *migrations* work on both backends, and
    ``create_all`` would bypass the very thing under test.
    """
    from alembic import command

    config = alembic_config(database_url)
    command.upgrade(config, "head")

    engine = create_db_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(migrated_engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(migrated_engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
