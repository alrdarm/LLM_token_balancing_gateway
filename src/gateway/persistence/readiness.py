"""Readiness probes for the persistence layer.

§11 requires readiness to fail for pending migrations, no active policy, or an
empty registry. Each is a separate probe so an operator can see which one is
wrong; the failing probe's *detail* stays in the logs, because health is
unauthenticated.

Probes are synchronous database calls run in a worker thread: blocking the
event loop on a hung database would make liveness fail too, and the process
would be killed rather than reported unready.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, select, text

from gateway.api.health import CheckOutcome, CheckProbe
from gateway.persistence.models import Model, RoutingPolicy

if TYPE_CHECKING:
    from alembic.config import Config


def _check_database(engine: Engine) -> CheckOutcome:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return CheckOutcome(ok=True)


def _check_migrations(engine: Engine, alembic_config: Config) -> CheckOutcome:
    """Compare the applied revision against the latest on disk."""
    script = ScriptDirectory.from_config(alembic_config)
    head = script.get_current_head()

    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()

    if current != head:
        return CheckOutcome(
            ok=False,
            detail=f"database at {current!r}, code expects {head!r}",
        )
    return CheckOutcome(ok=True)


def _check_registry(engine: Engine) -> CheckOutcome:
    """An empty registry means nothing is routable (§11)."""
    with engine.connect() as connection:
        count = connection.execute(select(Model).where(Model.enabled.is_(True)).limit(1)).first()
    if count is None:
        return CheckOutcome(ok=False, detail="no enabled models in registry")
    return CheckOutcome(ok=True)


def _check_active_policy(engine: Engine) -> CheckOutcome:
    """Routing cannot proceed without an active policy (§11)."""
    with engine.connect() as connection:
        found = connection.execute(
            select(RoutingPolicy).where(RoutingPolicy.active.is_(True)).limit(1)
        ).first()
    if found is None:
        return CheckOutcome(ok=False, detail="no active routing policy")
    return CheckOutcome(ok=True)


def register_persistence_probes(
    register: Callable[[str, CheckProbe], None],
    engine: Engine,
    alembic_config: Config,
) -> None:
    """Register the M1 readiness probes.

    Takes the registry's ``register`` callable rather than the registry itself,
    so persistence contributes probes without depending on how the API layer
    stores them.
    """

    async def database() -> CheckOutcome:
        return await asyncio.to_thread(_check_database, engine)

    async def migrations() -> CheckOutcome:
        return await asyncio.to_thread(_check_migrations, engine, alembic_config)

    async def registry_populated() -> CheckOutcome:
        return await asyncio.to_thread(_check_registry, engine)

    async def active_policy() -> CheckOutcome:
        return await asyncio.to_thread(_check_active_policy, engine)

    register("database", database)
    register("migrations", migrations)
    register("registry", registry_populated)
    register("active_policy", active_policy)
