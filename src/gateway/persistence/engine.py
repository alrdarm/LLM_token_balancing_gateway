"""Engine and session construction.

SQLite needs three pragmas set on every connection, none of them defaults:

* ``foreign_keys=ON`` -- SQLite ignores foreign keys otherwise, so cascade and
  reference constraints would silently not exist.
* ``journal_mode=WAL`` -- required by the spec, and it lets readers proceed
  while a writer holds the reservation transaction.
* ``busy_timeout`` -- §6 requires a short busy timeout to pair with
  ``BEGIN IMMEDIATE`` for atomic budget reservation.

WAL is persistent in the database file, but the other two are per-connection
and must be reapplied on every checkout.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    """Apply per-connection SQLite pragmas."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        # No-op for in-memory databases, which do not support WAL.
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def create_db_engine(url: str, *, echo: bool = False) -> Engine:
    """Build an engine with backend-appropriate settings."""
    if url.startswith("sqlite"):
        is_memory = ":memory:" in url or "mode=memory" in url
        engine = create_engine(
            url,
            echo=echo,
            future=True,
            # An in-memory database lives in its connection: a real pool would
            # hand out connections pointing at different empty databases.
            poolclass=StaticPool if is_memory else None,
            connect_args={"check_same_thread": False} if is_memory else {},
        )
        event.listen(engine, "connect", _configure_sqlite)
        return engine

    return create_engine(
        url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def immediate_transaction(session: Session) -> Iterator[Session]:
    """Open a write transaction that takes its lock immediately.

    SQLite defaults to deferred transactions, which acquire a write lock only
    at the first write. Two concurrent budget reservations could then both read
    headroom, both decide there is room, and one fail at commit -- or worse,
    both succeed against a stale read. ``BEGIN IMMEDIATE`` takes the write lock
    up front, which is what §6 specifies.

    PostgreSQL needs no equivalent here: it uses row locks taken by
    ``SELECT ... FOR UPDATE`` inside the transaction.
    """
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        session.execute(text("BEGIN IMMEDIATE"))
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
