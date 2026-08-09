"""Request ID resolution rules."""

from __future__ import annotations

import pytest

from gateway.api.request_id import (
    MAX_REQUEST_ID_LENGTH,
    generate_request_id,
    is_valid_request_id,
    resolve_request_id,
)


def test_generated_id_uses_req_prefix():
    request_id = generate_request_id()
    assert request_id.startswith("req_")
    assert is_valid_request_id(request_id)


def test_generated_ids_are_unique():
    assert generate_request_id() != generate_request_id()


@pytest.mark.parametrize(
    "value",
    [
        "req_2f1c8a1e-9f43-4a1b-9f6c-1e2d3c4b5a60",
        "trace-abc.123",
        "client:job_42",
        "a",
        "x" * MAX_REQUEST_ID_LENGTH,
    ],
)
def test_accepts_safe_client_ids(value):
    assert is_valid_request_id(value)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("", "empty"),
        ("x" * (MAX_REQUEST_ID_LENGTH + 1), "too long"),
        ("has space", "whitespace"),
        ("inject\r\nX-Evil: 1", "header split"),
        ("newline\n", "log forging"),
        ("null\x00byte", "control character"),
        ("emoji-\U0001f600", "non-ascii"),
        ("semi;colon", "disallowed punctuation"),
    ],
)
def test_rejects_unsafe_client_ids(value, reason):
    assert not is_valid_request_id(value), reason


def test_resolve_prefers_valid_client_id():
    assert resolve_request_id("trace-abc") == "trace-abc"


@pytest.mark.parametrize("value", [None, "", "bad value", "inject\r\nX-Evil: 1"])
def test_resolve_generates_when_absent_or_invalid(value):
    resolved = resolve_request_id(value)
    assert resolved.startswith("req_")
    assert resolved != value
