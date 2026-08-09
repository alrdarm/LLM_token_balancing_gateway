"""Alembic environment.

The database URL always comes from settings rather than ``alembic.ini`` so the
application, migrations, and tests cannot disagree about which database they
are pointing at.

``render_as_batch`` is on because SQLite cannot ``ALTER`` most constraints in
place; Alembic emulates it by rebuilding the table. Without it, the first
migration that alters a column would work on PostgreSQL and fail on SQLite --
and the spec requires both.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from gateway.config import get_settings
from gateway.persistence.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the database URL.

    Order matters. A caller that constructed the ``Config`` programmatically
    (tests, the readiness probe) has already said which database it means, and
    silently falling through to settings would point migrations at the wrong
    one -- which shows up as "table already exists" against a shared default.
    """
    override = context.get_x_argument(as_dictionary=True).get("url")
    if override:
        return override

    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured

    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL without a live connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # ``begin()`` rather than ``connect()``: SQLAlchemy 2.0 has no autocommit,
    # and SQLite reports non-transactional DDL, so Alembic's own transaction
    # block is a no-op there. Without an explicit transaction the CREATE TABLEs
    # persist (pysqlite autocommits DDL) but the alembic_version INSERT is
    # rolled back on close -- leaving a schema that claims to be at no revision.
    with connectable.begin() as connection:
        if connection.dialect.name == "sqlite":
            # Alembic's own connection needs the pragma too: without it the
            # rebuilds that batch mode performs would drop foreign keys.
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
