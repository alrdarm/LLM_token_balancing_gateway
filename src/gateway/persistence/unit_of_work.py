"""Unit of work.

Groups repositories behind one transaction boundary so a caller cannot commit
half a state change -- for example an attempt row without its reservation.

The context manager rolls back on any exception rather than leaving the session
dirty, because §7 requires every terminal path to settle or release its
reservation, and a partially applied transaction makes that unprovable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker

from gateway.persistence.repositories import (
    BudgetRepository,
    IdempotencyRepository,
    ModelRepository,
    PolicyRepository,
    QuotaRepository,
    RequestRepository,
)


class UnitOfWork:
    """One transaction and the repositories that share it."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.models = ModelRepository(session)
        self.policies = PolicyRepository(session)
        self.requests = RequestRepository(session)
        self.budgets = BudgetRepository(session)
        self.idempotency = IdempotencyRepository(session)
        self.quotas = QuotaRepository(session)

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def flush(self) -> None:
        """Send pending changes without committing, to surface constraint
        violations and populate generated keys."""
        self.session.flush()

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.rollback()
        self.session.close()


@contextmanager
def unit_of_work(session_factory: sessionmaker[Session]) -> Iterator[UnitOfWork]:
    """Run a block in one transaction, committing on success.

    Nothing is committed if the block raises, so callers cannot leave a request
    recorded without the reservation that funds it.
    """
    session = session_factory()
    uow = UnitOfWork(session)
    try:
        yield uow
        uow.commit()
    except Exception:
        uow.rollback()
        raise
    finally:
        session.close()
