"""Authentication and scopes (§1, §11).

Every non-health endpoint requires ``Authorization: Bearer <key>``. Keys are
stored only as keyed digests, so the lookup hashes the presented secret and
compares digests -- the plaintext never touches the database or the logs.

Failures are deliberately uniform: an unknown key, a disabled key, and an
expired key all return the same 401 with the same message. Distinguishing them
would let an attacker enumerate valid keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.domain.errors import InvalidAPIKeyError, PolicyDeniedError
from gateway.persistence.models import APIKey
from gateway.telemetry.hashing import keyed_digest

logger = logging.getLogger(__name__)

BEARER_PREFIX = "bearer "

#: Scope required to see route, cost, and model detail on responses (§1, §2).
SCOPE_DEBUG = "debug"
#: Scope required to read another request's sanitized status.
SCOPE_ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class AuthenticatedClient:
    """The caller behind a request."""

    client_id: str
    scopes: frozenset[str]
    control_overrides: dict[str, Any]

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def require_scope(self, scope: str) -> None:
        """Raise unless the client holds ``scope``."""
        if not self.has_scope(scope):
            raise PolicyDeniedError(
                f"This credential lacks the required scope: {scope}.",
            )


def extract_bearer_token(authorization: str | None) -> str:
    """Pull the credential out of an ``Authorization`` header."""
    if not authorization:
        raise InvalidAPIKeyError("Missing Authorization header.")

    if not authorization.lower().startswith(BEARER_PREFIX):
        raise InvalidAPIKeyError("Authorization header must use the Bearer scheme.")

    token = authorization[len(BEARER_PREFIX) :].strip()
    if not token:
        raise InvalidAPIKeyError("Bearer token is empty.")
    return token


def key_prefix(secret: str) -> str:
    """Non-secret identifying prefix, safe to log."""
    return secret[:8]


def authenticate(
    session: Session, authorization: str | None, *, hash_key: str
) -> AuthenticatedClient:
    """Resolve an ``Authorization`` header to a client, or raise 401."""
    token = extract_bearer_token(authorization)
    digest = keyed_digest(token, key=hash_key)

    record = session.scalars(select(APIKey).where(APIKey.key_hash == digest)).first()

    # One message for every failure mode, so a caller cannot tell a wrong key
    # from a disabled or expired one.
    if record is None or not record.enabled:
        logger.warning("Authentication failed", extra={"event": "auth_failed"})
        raise InvalidAPIKeyError("Invalid API key.")

    now = datetime.now(UTC)
    if record.expires_at is not None and record.expires_at <= now:
        logger.warning("Authentication failed", extra={"event": "auth_expired"})
        raise InvalidAPIKeyError("Invalid API key.")

    record.last_used_at = now

    return AuthenticatedClient(
        client_id=record.client_id,
        scopes=frozenset(record.scopes or ()),
        control_overrides=dict(record.control_overrides or {}),
    )


def create_api_key(
    session: Session,
    *,
    client_id: str,
    secret: str,
    hash_key: str,
    label: str = "",
    scopes: list[str] | None = None,
    control_overrides: dict[str, Any] | None = None,
) -> APIKey:
    """Register a credential, storing only its digest.

    The caller keeps ``secret``; the gateway cannot recover it afterwards.
    """
    record = APIKey(
        client_id=client_id,
        label=label,
        key_hash=keyed_digest(secret, key=hash_key),
        key_prefix=key_prefix(secret),
        scopes=list(scopes or []),
        control_overrides=dict(control_overrides or {}),
        enabled=True,
    )
    session.add(record)
    return record
