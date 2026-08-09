"""Deployment configuration.

Settings come from environment variables prefixed ``GATEWAY_`` (or a local
``.env``). Only M0 concerns live here; routing, budget, and provider settings
arrive with the milestones that need them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Process-wide deployment settings."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "llm-token-balancing-gateway"
    environment: Environment = "local"
    log_level: LogLevel = "INFO"

    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def is_production(self) -> bool:
        """Whether this deployment must apply production disclosure rules."""
        return self.environment == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings.

    Cached so configuration is read once and stays stable for the process
    lifetime; tests clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
