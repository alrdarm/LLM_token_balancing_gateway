"""JSON log formatting and its redaction guarantees."""

from __future__ import annotations

import json
import logging

from gateway.telemetry.context import reset_request_id, set_request_id
from gateway.telemetry.logging import JsonFormatter, log_context


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="gateway.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_as_single_line_json():
    output = JsonFormatter().format(_record())
    assert "\n" not in output
    payload = json.loads(output)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "gateway.test"


def test_includes_ambient_request_id():
    token = set_request_id("req_abc")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        reset_request_id(token)
    assert payload["request_id"] == "req_abc"


def test_omits_request_id_outside_a_request():
    assert "request_id" not in json.loads(JsonFormatter().format(_record()))


def test_allowlisted_extra_is_kept():
    payload = json.loads(JsonFormatter().format(_record(status_code=503, check="database")))
    assert payload["status_code"] == 503
    assert payload["check"] == "database"


def test_non_allowlisted_extra_is_dropped():
    """Content-bearing fields must not reach logs even if a call site adds them."""
    payload = json.loads(
        JsonFormatter().format(_record(prompt="secret user text", api_key="sk-live-123"))
    )
    assert "prompt" not in payload
    assert "api_key" not in payload
    assert "secret user text" not in json.dumps(payload)


def test_exception_is_reduced_to_type_and_message():
    try:
        raise ValueError("boom at /internal/path/secrets.py")
    except ValueError:
        record = _record()
        record.exc_info = logging.sys.exc_info()  # type: ignore[attr-defined]
        payload = json.loads(JsonFormatter().format(record))

    assert payload["exception_type"] == "ValueError"
    assert "Traceback" not in json.dumps(payload)


def test_log_context_filters_to_allowlist():
    assert log_context(status_code=200, prompt="secret") == {"status_code": 200}
