"""Redaction snapshots (§11, §12 security layer).

The spec lists redaction snapshots as required security coverage. These are
adversarial by design: each one takes a secret that a careless implementation
would leak and asserts it does not appear.
"""

from __future__ import annotations

import json
import logging

import pytest

from gateway.telemetry.logging import JsonFormatter
from gateway.telemetry.redaction import (
    PLACEHOLDER,
    is_content_key,
    is_sensitive_key,
    redact,
    redact_headers,
    redact_text,
    safe_repr,
)


# Synthetic credentials, assembled at runtime rather than written as literals.
# They must *look* like the real thing to exercise redaction, which also means
# they trip GitHub's push protection and any other scanner reading the file --
# including this repository's own secret scan. Composing them keeps the fixture
# honest without committing a credential-shaped string.
def _fake(prefix: str, body: str, *, separator: str = "-") -> str:
    return f"{prefix}{separator}{body}"


SECRETS = [
    _fake("sk", "proj-abcdefghijklmnopqrstuvwxyz123456"),
    _fake("ghp", "abcdefghijklmnopqrstuvwxyz1234567890", separator="_"),
    _fake("AKIA", "IOSFODNN7EXAMPLE", separator=""),
    _fake("AIza", "SyD1234567890abcdefghijklmnopqrstuv", separator=""),
    "Bearer " + _fake("eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxIn0.abcdefghijk", separator="."),
    _fake("xoxb", "123456789012-abcdefghijklmnopqrst"),
]


@pytest.mark.parametrize("secret", SECRETS)
def test_credential_shaped_values_are_redacted_anywhere(secret):
    """Pattern matching catches secrets under field names nobody anticipated."""
    text = f"the value was {secret} in the payload"
    assert secret not in redact_text(text)
    assert PLACEHOLDER in redact_text(text)


def test_private_key_blocks_are_redacted():
    marker = "-----BEGIN RSA PRIVATE " + "KEY-----"
    key = f"{marker}\nMIIEow==\n-----END RSA PRIVATE KEY-----"
    assert marker not in redact_text(key)


@pytest.mark.parametrize(
    "name",
    ["authorization", "Authorization", "X-Api-Key", "api_key", "GATEWAY_HASH_KEY", "cookie"],
)
def test_sensitive_field_names_are_recognised(name):
    assert is_sensitive_key(name)


@pytest.mark.parametrize("name", ["prompt", "messages", "output_text", "tool_calls", "body"])
def test_content_field_names_are_recognised(name):
    """§6 forbids raw prompts and outputs at rest; §11 extends it to logs."""
    assert is_content_key(name)


def test_sensitive_values_are_redacted_by_key_even_when_innocuous():
    """A field named api_key is redacted whatever it holds."""
    assert redact({"api_key": "hello"})["api_key"] == PLACEHOLDER


def test_prompt_content_is_redacted_by_key():
    payload = {"messages": [{"role": "user", "content": "confidential-marker"}]}
    assert "confidential-marker" not in json.dumps(redact(payload))


def test_nested_structures_are_redacted():
    payload = {
        "request": {
            "headers": {"authorization": "Bearer " + _fake("sk", "live-abcdefghijklmnop")},
            "body": {"prompt": "secret text"},
        }
    }
    rendered = json.dumps(redact(payload))
    assert "live-abcdefghijklmnop" not in rendered
    assert "secret text" not in rendered


def test_lists_are_redacted_elementwise():
    rendered = json.dumps(redact([_fake("sk", "proj-abcdefghijklmnopqrstuvwxyz123456"), "safe"]))
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "safe" in rendered


def test_deep_nesting_is_bounded():
    """Unbounded recursion on a hostile object is its own problem."""
    payload: dict = {}
    node = payload
    for _ in range(50):
        node["next"] = {}
        node = node["next"]
    node["api_key"] = _fake("sk", "deep-secret-value-123456")

    assert "deep-secret-value" not in json.dumps(redact(payload))


def test_long_values_are_truncated():
    """A very long value is a body or a document, not a log field."""
    result = redact_text("x" * 5000)
    assert "TRUNCATED" in result
    assert len(result) < 5000


def test_headers_are_redacted():
    headers = redact_headers(
        {
            "Authorization": "Bearer " + _fake("sk", "live-1234567890"),
            "Content-Type": "application/json",
        }
    )
    assert headers["Authorization"] == PLACEHOLDER
    assert headers["Content-Type"] == "application/json"


def test_safe_repr_hides_the_offending_value():
    """Exception messages routinely quote what was rejected."""
    error = ValueError("failed on " + _fake("sk", "proj-abcdefghijklmnopqrstuvwxyz123456"))
    rendered = safe_repr(error)
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "ValueError" in rendered


def test_redaction_never_raises():
    """Redaction runs inside error handlers; throwing there would surface the
    very data it exists to hide."""

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    assert redact(Hostile()) is not None


def test_numbers_and_booleans_survive():
    assert redact({"status_code": 503, "ok": False}) == {"status_code": 503, "ok": False}


# --- the formatter itself --------------------------------------------------


def _record(message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="gateway.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_redacts_the_message():
    output = JsonFormatter().format(_record("calling with " + _fake("sk", "live-abcdefghijklmnop")))
    assert "live-abcdefghijklmnop" not in output


def test_formatter_redacts_exception_messages():
    try:
        raise ValueError("bad key " + _fake("sk", "proj-abcdefghijklmnopqrstuvwxyz123456"))
    except ValueError:
        import sys

        record = _record("failed")
        record.exc_info = sys.exc_info()
        output = JsonFormatter().format(record)

    assert "abcdefghijklmnopqrstuvwxyz" not in output
    assert "ValueError" in output


def test_formatter_still_drops_unknown_fields():
    """The allowlist remains the first line of defence."""
    output = JsonFormatter().format(_record("hi", prompt="secret user text"))
    assert "secret user text" not in output
    assert "prompt" not in json.loads(output)
