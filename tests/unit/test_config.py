"""Settings loading."""

from __future__ import annotations

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
    settings = Settings(_env_file=None)
    assert settings.environment == "prod"
    assert settings.port == 9001
    assert settings.is_production


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
