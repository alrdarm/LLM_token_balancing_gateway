"""End-to-end request ID behaviour on the HTTP surface.

The M0 exit gate names request IDs explicitly: every response carries one,
whatever the status code.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter

from gateway.api.request_id import OUTBOUND_HEADER

HEADER = OUTBOUND_HEADER


def test_response_always_carries_a_request_id(client):
    response = client.get("/health/live")
    assert response.headers[HEADER].startswith("req_")


def test_valid_client_id_is_echoed(client):
    response = client.get("/health/live", headers={"X-Request-ID": "trace-abc.123"})
    assert response.headers[HEADER] == "trace-abc.123"


def test_invalid_client_id_is_replaced_not_echoed(client):
    response = client.get("/health/live", headers={"X-Request-ID": "has space"})
    assert response.headers[HEADER] != "has space"
    assert response.headers[HEADER].startswith("req_")


def test_each_request_gets_a_distinct_generated_id(client):
    first = client.get("/health/live").headers[HEADER]
    second = client.get("/health/live").headers[HEADER]
    assert first != second


def test_404_carries_request_id_and_envelope(client):
    response = client.get("/no/such/route")
    assert response.status_code == 404
    assert response.headers[HEADER].startswith("req_")

    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["gateway"]["request_id"] == response.headers[HEADER]


def test_header_and_body_request_ids_agree_on_errors(client):
    response = client.get("/no/such/route", headers={"X-Request-ID": "trace-xyz"})
    assert response.headers[HEADER] == "trace-xyz"
    assert response.json()["gateway"]["request_id"] == "trace-xyz"


@pytest.fixture
def app_with_failing_route(app):
    """Mount a route that raises, to exercise the unhandled-error path."""
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret detail from /srv/gateway/internal.py")

    app.include_router(router)
    return app


def test_unhandled_error_returns_redacted_500_with_request_id(app_with_failing_route):
    from fastapi.testclient import TestClient

    with TestClient(app_with_failing_route, raise_server_exceptions=False) as client:
        response = client.get("/boom", headers={"X-Request-ID": "trace-boom"})

    assert response.status_code == 500
    assert response.headers[HEADER] == "trace-boom"

    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["gateway"]["request_id"] == "trace-boom"
    # No internal paths, no exception text.
    assert "/srv/gateway" not in response.text
    assert "secret detail" not in response.text
