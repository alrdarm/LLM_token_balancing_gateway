"""Redaction (§11).

*"Redact authorization, provider keys, cookies, bodies, tool arguments, and
common secrets from logs, errors, and traces."*

Two complementary strategies, because either alone fails:

* **Key-based** — a field named ``authorization`` or ``api_key`` is redacted
  whatever it contains. Catches secrets we know the name of.
* **Pattern-based** — a value that *looks* like a credential is redacted
  whatever it is called. Catches secrets that arrive under a name nobody
  anticipated, which is how they usually arrive.

Redaction is applied to the value, never used to decide whether to log: a
record that cannot be redacted is dropped rather than emitted, because a
partial redaction is worse than no log line at all.

The functions here are deliberately total — they never raise. A redaction
routine that throws inside an exception handler would replace a redacted log
with an unredacted traceback, which is the exact failure it exists to prevent.
"""

from __future__ import annotations

import re
from typing import Any

#: The single placeholder, so a redacted value is unmistakable in a log.
PLACEHOLDER = "[REDACTED]"

#: Field names whose values are always secret, matched case-insensitively on
#: substrings so ``x_api_key`` and ``providerApiKey`` are both caught.
SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "authorization",
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "cookie",
    "session",
    "private_key",
    "hash_key",
    "bearer",
    "signature",
)

#: Field names that carry request or response *content* rather than secrets.
#: §6 forbids storing raw prompts and outputs; §11 extends that to logs.
CONTENT_KEY_PARTS: tuple[str, ...] = (
    "prompt",
    "messages",
    "content",
    "input",
    "output",
    "text",
    "instructions",
    "arguments",
    "tool_calls",
    "body",
    "completion",
    "delta",
)

#: Values that look like credentials regardless of their field name.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI-style and similar prefixed keys.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    # GitHub uses an underscore separator, Slack a hyphen; accept either.
    re.compile(r"\b(?:gh[pousr]|xox[baprs])[_\-][A-Za-z0-9_\-]{10,}"),
    # Bearer credentials anywhere in a string.
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.I),
    # AWS access key IDs.
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Google API keys. Length is not pinned: vendors change it, and a key
    # that is one character off must still be redacted.
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    # JWTs.
    re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}\b"),
    # Anything self-describing as a private key block.
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

#: Cap on redacted string length. A very long value is a body or a document,
#: neither of which belongs in a log even after pattern redaction.
MAX_LOGGED_STRING = 256

_MAX_DEPTH = 6


def _normalise_key(key: str) -> str:
    """Fold separator styles together.

    ``X-Api-Key``, ``x_api_key``, and ``apiKey`` are the same field wearing
    three different conventions, and a secret must not survive because a
    caller used hyphens.
    """
    return key.lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def is_sensitive_key(key: str) -> bool:
    """Whether a field name always carries a secret."""
    normalised = _normalise_key(key)
    return any(part.replace("-", "_") in normalised for part in SENSITIVE_KEY_PARTS)


def is_content_key(key: str) -> bool:
    """Whether a field name carries request or response content."""
    normalised = _normalise_key(key)
    return any(part in normalised for part in CONTENT_KEY_PARTS)


def redact_text(value: str) -> str:
    """Replace anything in ``value`` that looks like a credential."""
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(PLACEHOLDER, redacted)

    if len(redacted) > MAX_LOGGED_STRING:
        # Truncation is reported, so a reader knows they are not seeing it all.
        return f"{redacted[:MAX_LOGGED_STRING]}...[TRUNCATED]"
    return redacted


def redact(value: Any, *, key: str | None = None, _depth: int = 0) -> Any:
    """Redact a value, recursing through mappings and sequences.

    Total by construction: an unexpected type is rendered through
    :func:`redact_text` rather than raising, because this runs inside error
    paths where an exception would surface the very data it was hiding.
    """
    if key is not None and (is_sensitive_key(key) or is_content_key(key)):
        return PLACEHOLDER

    if _depth >= _MAX_DEPTH:
        # Deeply nested structures are almost always payloads, and unbounded
        # recursion on a hostile object is its own problem.
        return PLACEHOLDER

    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, dict):
        return {
            str(name): redact(item, key=str(name), _depth=_depth + 1)
            for name, item in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [redact(item, _depth=_depth + 1) for item in value]

    try:
        return redact_text(str(value))
    except Exception:  # pragma: no cover - defensive; redaction must not raise
        return PLACEHOLDER


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact an HTTP header map.

    Only the *secret* rule applies here, not the content rule: ``Content-Type``
    and ``Content-Length`` are transport metadata, and redacting them would
    strip the most useful fields from every request log while protecting
    nothing. Values are still pattern-scanned, so a credential pasted into an
    unexpected header is caught.
    """
    return {
        name: PLACEHOLDER if is_sensitive_key(name) else redact_text(str(value))
        for name, value in headers.items()
    }


def safe_repr(exc: BaseException) -> str:
    """A loggable description of an exception.

    Type and redacted message only. An exception's ``str`` frequently contains
    the offending value -- a failed SQL statement, a rejected payload, a URL
    with a key in the query string -- so it is never logged raw.
    """
    return f"{type(exc).__name__}: {redact_text(str(exc))}"
