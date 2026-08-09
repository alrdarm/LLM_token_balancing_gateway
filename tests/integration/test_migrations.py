"""Migration behaviour on every supported backend.

The M1 exit gate is "migration and round-trip tests both DBs". These tests are
parametrised through the ``database_url`` fixture, so each runs once per
backend.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from gateway.persistence.migrations_config import alembic_config
from gateway.persistence.models import Base

pytestmark = pytest.mark.integration

#: Every table §6 requires, plus Alembic's own bookkeeping.
EXPECTED_TABLES = {
    "models",
    "model_prices",
    "routing_policies",
    "requests",
    "attempts",
    "validations",
    "budgets",
    "quota_snapshots",
    "budget_reservations",
    "idempotency_records",
}


def test_upgrade_creates_every_spec_table(migrated_engine: Engine):
    tables = set(inspect(migrated_engine).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_upgrade_reaches_head(migrated_engine: Engine, database_url: str):
    script = ScriptDirectory.from_config(alembic_config(database_url))
    with migrated_engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()
    assert current == script.get_current_head()


def test_migration_schema_matches_the_orm_metadata(migrated_engine: Engine):
    """Guards against a model change landing without a migration.

    Compares the tables and columns the ORM declares against what the
    migrations actually built, so the two cannot silently drift.
    """
    inspector = inspect(migrated_engine)
    actual_tables = set(inspector.get_table_names())

    for table_name, table in Base.metadata.tables.items():
        assert table_name in actual_tables, f"{table_name} missing from migrated schema"

        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        declared_columns = {column.name for column in table.columns}
        assert declared_columns == actual_columns, (
            f"{table_name} column drift between ORM and migration: "
            f"only in ORM={declared_columns - actual_columns}, "
            f"only in DB={actual_columns - declared_columns}"
        )


def test_downgrade_removes_the_schema(migrated_engine: Engine, database_url: str):
    """A migration that cannot be undone cannot be rolled back in production."""
    migrated_engine.dispose()
    config = alembic_config(database_url)

    command.downgrade(config, "base")

    from gateway.persistence.engine import create_db_engine

    engine = create_db_engine(database_url)
    try:
        remaining = set(inspect(engine).get_table_names())
        assert not (EXPECTED_TABLES & remaining), f"tables survived downgrade: {remaining}"
    finally:
        engine.dispose()


def test_upgrade_is_repeatable_after_downgrade(database_url: str, migrated_engine: Engine):
    """Down then up again must produce the same schema."""
    migrated_engine.dispose()
    config = alembic_config(database_url)

    command.downgrade(config, "base")
    command.upgrade(config, "head")

    from gateway.persistence.engine import create_db_engine

    engine = create_db_engine(database_url)
    try:
        assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
