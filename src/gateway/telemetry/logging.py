"""Structured JSON logging.

Records are emitted as single-line JSON so downstream tooling can index them
without parsing free text. The formatter serialises an explicit field
allowlist: log records must never carry prompt text, provider payloads, or
credentials, so unknown ``extra`` keys are dropped rather than passed through.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from gateway.telemetry.context import get_request_id
from gateway.telemetry.redaction import redact, redact_text

# Operational fields a call site may attach via ``logger.info(..., extra=...)``.
# Anything outside this set is dropped by the formatter.
ALLOWED_EXTRA_FIELDS = frozenset(
    {
        "attempt_id",
        "check",
        "duration_ms",
        "event",
        "method",
        "path",
        "status_code",
    }
)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON with the ambient request ID."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            # Redacted even though messages are meant to be static: a format
            # argument can carry anything, and §11 requires secrets out of logs
            # regardless of how they got there.
            "message": redact_text(record.getMessage()),
        }

        request_id = get_request_id()
        if request_id is not None:
            payload["request_id"] = request_id

        for field in ALLOWED_EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                # Belt and braces: the allowlist should already exclude
                # anything content-bearing, but a value can still contain a
                # credential pasted into an operational field.
                payload[field] = redact(value, key=field)

        if record.exc_info:
            # Type and message only. Tracebacks can embed request content and
            # internal paths, which the spec forbids exposing.
            exc_type, exc_value, _ = record.exc_info
            if exc_type is not None:
                payload["exception_type"] = exc_type.__name__
                # An exception's str routinely contains the offending value --
                # a failed statement, a rejected payload, a URL with a key in
                # the query string -- so it is redacted, never logged raw.
                payload["exception_message"] = redact_text(str(exc_value))

        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: repeated calls replace handlers rather than stacking them, so
    reloads and test fixtures do not multiply output.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def log_context(**fields: Any) -> Mapping[str, Any]:
    """Filter ``fields`` down to the allowlist for use as logging ``extra``."""
    return {key: value for key, value in fields.items() if key in ALLOWED_EXTRA_FIELDS}
