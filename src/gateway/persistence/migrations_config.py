"""Programmatic access to the Alembic configuration.

The readiness probe and the tests both need to know which revision the code
expects. Resolving the script location from this package rather than from the
process working directory means ``alembic upgrade`` and a running gateway agree
on the same migration history no matter where either was started from.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

#: Directory holding ``env.py`` and ``versions/``.
MIGRATIONS_PATH = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str | None = None) -> Config:
    """Build an Alembic ``Config`` pointing at this package's migrations."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url)
    return config
