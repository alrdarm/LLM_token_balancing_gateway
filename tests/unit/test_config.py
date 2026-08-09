"""Settings loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gateway.config import Settings, get_settings


def test_defaults_are_local_and_safe():
    settings = Settings(_env_file=None)
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.host == "127.0.0.1"
    assert not settings.is_production


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENVIRONMENT", "prod")
    monkeypatch.setenv("GATEWAY_PORT", "9001")
    monkeypatch.setenv("GATEWAY_HASH_KEY", "deployment-specific-secret")
    settings = Settings(_env_file=None)
    assert settings.environment == "prod"
    assert settings.port == 9001
    assert settings.is_production


@pytest.mark.parametrize("environment", ["staging", "prod"])
def test_deployed_environments_reject_the_default_hash_key(environment):
    """The key is what stops stored digests being brute-forced back to prompt
    text, so shipping the public default must fail loudly at startup."""
    with pytest.raises(ValidationError, match="GATEWAY_HASH_KEY"):
        Settings(_env_file=None, environment=environment)


@pytest.mark.parametrize("environment", ["local", "dev"])
def test_local_environments_allow_the_default_hash_key(environment):
    settings = Settings(_env_file=None, environment=environment)
    assert settings.hash_key
    assert not settings.is_production


def test_deployed_environment_accepts_an_explicit_key():
    settings = Settings(_env_file=None, environment="prod", hash_key="a-real-secret")
    assert settings.hash_key == "a-real-secret"


def test_database_url_defaults_to_sqlite():
    assert Settings(_env_file=None).database_url.startswith("sqlite")


def test_settings_are_frozen():
    settings = Settings(_env_file=None)
    try:
        settings.port = 1  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "immutable" in str(exc).lower()
    else:
        raise AssertionError("Settings should be immutable")


def test_get_settings_is_cached():
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
