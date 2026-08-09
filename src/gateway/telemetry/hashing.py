"""Keyed digests for request and output content.

The spec forbids storing raw prompts and outputs, and requires inputs and
outputs to be hashed with a *deployment-specific keyed* digest. The key matters:
an unkeyed hash of a short prompt is trivially reversible by brute force, so an
unkeyed digest would leak the very content the storage rule protects.

Keys are never logged and never appear in error messages.
"""

from __future__ import annotations

import hmac
from hashlib import sha256

#: Digest length in hex characters; sized to the ``String(64)`` hash columns.
DIGEST_LENGTH = 64


class HashKeyError(RuntimeError):
    """The deployment hash key is missing or unusable."""


def keyed_digest(payload: str | bytes, *, key: str) -> str:
    """Return the HMAC-SHA256 digest of ``payload`` under ``key``.

    Raises rather than falling back to an unkeyed hash: a silent fallback would
    downgrade the privacy guarantee exactly when the key is misconfigured.
    """
    if not key:
        raise HashKeyError("a non-empty deployment hash key is required")

    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hmac.new(key.encode("utf-8"), data, sha256).hexdigest()
