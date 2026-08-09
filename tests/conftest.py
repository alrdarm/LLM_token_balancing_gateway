"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api.app import create_app
from gateway.api.health import readiness
from gateway.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore any developer ``.env``."""
    return Settings(_env_file=None, environment="local", log_level="INFO")


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """A fresh app with an empty readiness registry."""
    readiness.clear()
    yield create_app(settings)
    readiness.clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Test client that surfaces server errors as responses, not exceptions."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
