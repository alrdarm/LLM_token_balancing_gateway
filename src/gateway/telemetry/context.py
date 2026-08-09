"""Ambient request context.

The request ID is carried in a :class:`~contextvars.ContextVar` so log records
can be correlated without threading the ID through every call signature. The
spec traces request -> attempt -> validation by IDs, never by content, so this
context must only ever hold identifiers.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("gateway_request_id", default=None)


def get_request_id() -> str | None:
    """Return the current request ID, or ``None`` outside a request."""
    return _request_id.get()


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind ``request_id`` to the current context and return the reset token."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request ID bound before ``token`` was issued."""
    _request_id.reset(token)
