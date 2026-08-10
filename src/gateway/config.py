"""Deployment configuration.

Settings come from environment variables prefixed ``GATEWAY_`` (or a local
``.env``). Only M0 concerns live here; routing, budget, and provider settings
arrive with the milestones that need them.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.domain.enums import Privacy
from gateway.persistence.types import to_money

Environment = Literal["local", "dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: Sentinel for the non-secret development key, rejected outside local/dev.
_DEFAULT_HASH_KEY = "local-development-hash-key"


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

    #: SQLite is the v0.1 default; PostgreSQL must work without domain
    #: redesign, so this is the only thing that changes between them.
    database_url: str = "sqlite+pysqlite:///./gateway.db"
    database_echo: bool = False

    #: Deployment-specific key for hashing inputs and outputs. The default is
    #: usable for local development only; see ``_validate_hash_key``.
    hash_key: str = "local-development-hash-key"

    #: Deployment control floor -- the highest-precedence layer (§2). A caller
    #: can be stricter than this but never looser.
    #: ``None`` means the deployment pins nothing and callers decide.
    deployment_privacy_floor: Privacy | None = None
    #: Applied when a caller sets no ceiling of their own. Parsed as Decimal.
    default_max_cost: Decimal = Decimal("1.000000000")
    default_max_latency_ms: int = 60_000

    @field_validator("default_max_cost", mode="before")
    @classmethod
    def _exact_default_cost(cls, value: object) -> object:
        """Reject float configuration for money, as everywhere else."""
        if value is None or isinstance(value, Decimal):
            return value
        return to_money(value)  # type: ignore[arg-type]

    @property
    def is_production(self) -> bool:
        """Whether this deployment must apply production disclosure rules."""
        return self.environment in ("staging", "prod")

    @model_validator(mode="after")
    def _validate_hash_key(self) -> Settings:
        """Refuse to run a deployed environment on the development hash key.

        The key is what stops stored digests from being brute-forced back to
        prompt text. Shipping the public default to staging or production would
        void that protection silently, so this fails at startup instead.
        """
        if self.is_production and self.hash_key == _DEFAULT_HASH_KEY:
            raise ValueError(
                "GATEWAY_HASH_KEY must be set to a deployment-specific secret "
                f"when GATEWAY_ENVIRONMENT is {self.environment!r}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings.

    Cached so configuration is read once and stays stable for the process
    lifetime; tests clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
