"""The lifecycle over HTTP: idempotency, dry run, and disclosure (§1, §2, §12).

Covers T06 and T07 from §12 through the real endpoints, so idempotency is
exercised the way a client would hit it rather than at the service boundary.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from gateway.persistence.engine import create_db_engine, create_session_factory
from gateway.persistence.models import Attempt, IdempotencyRecord, Request
from tests.conftest import TEST_DEBUG_API_KEY

pytestmark = pytest.mark.e2e

CHAT_PATH = "/v1/chat/completions"
BODY = {"model": "auto", "messages": [{"role": "user", "content": "Summarise this."}]}
OTHER_BODY = {"model": "auto", "messages": [{"role": "user", "content": "Different question."}]}


def counts(api_settings) -> tuple[int, int]:
    """(requests, attempts) currently stored."""
    engine = create_db_engine(api_settings.database_url)
    factory = create_session_factory(engine)
    try:
        with factory() as session:
            return (
                session.scalar(select(func.count()).select_from(Request)) or 0,
                session.scalar(select(func.count()).select_from(Attempt)) or 0,
            )
    finally:
        engine.dispose()


# --- happy path -----------------------------------------------------------


def test_request_succeeds_end_to_end(api_client):
    response = api_client.post(CHAT_PATH, json=BODY)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"]
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_request_and_attempt_rows_are_persisted(api_client, api_settings):
    api_client.post(CHAT_PATH, json=BODY)

    requests, attempts = counts(api_settings)
    assert requests == 1
    assert attempts >= 1


def test_response_does_not_disclose_route_without_scope(api_client):
    """§1: route and cost summary only when authorized."""
    block = api_client.post(CHAT_PATH, json=BODY).json()["gateway"]

    assert set(block) == {"request_id"}
    assert "resolved_model" not in block
    assert "cost" not in block


def test_debug_scope_discloses_route_and_cost(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app, raise_server_exceptions=False) as client:
        response = client.post(
            CHAT_PATH,
            json=BODY,
            headers={"Authorization": f"Bearer {TEST_DEBUG_API_KEY}"},
        )

    block = response.json()["gateway"]
    assert block["resolved_model"]
    assert block["attempts"] >= 1
    assert block["validation"] == "PASS"
    # §2: money is a string on the wire so it cannot become a float.
    assert isinstance(block["cost"]["actual"], str)


# --- T06: same key, same body ---------------------------------------------


def test_same_key_and_body_does_not_call_the_provider_twice(api_client, api_settings):
    """T06: no second provider call for a replayed key."""
    headers = {"Idempotency-Key": "key-t06"}

    first = api_client.post(CHAT_PATH, json=BODY, headers=headers)
    assert first.status_code == 200

    _, attempts_after_first = counts(api_settings)

    second = api_client.post(CHAT_PATH, json=BODY, headers=headers)

    _, attempts_after_second = counts(api_settings)
    assert attempts_after_second == attempts_after_first, (
        "a replayed key must not produce another provider attempt"
    )
    # v0.1 stores a reference rather than the body, so the replay is reported
    # rather than reconstructed.
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "idempotency_conflict"


def test_idempotency_record_is_stored(api_client, api_settings):
    api_client.post(CHAT_PATH, json=BODY, headers={"Idempotency-Key": "key-store"})

    engine = create_db_engine(api_settings.database_url)
    factory = create_session_factory(engine)
    try:
        with factory() as session:
            record = session.scalars(
                select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == "key-store")
            ).one()
            assert record.state == "COMPLETED"
            assert record.response_ref
            assert record.input_hash
    finally:
        engine.dispose()


# --- T07: same key, different body ----------------------------------------


def test_same_key_different_body_is_409(api_client):
    """T07: key conflict, not a silent replay of the wrong answer."""
    headers = {"Idempotency-Key": "key-t07"}

    assert api_client.post(CHAT_PATH, json=BODY, headers=headers).status_code == 200

    conflict = api_client.post(CHAT_PATH, json=OTHER_BODY, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert conflict.json()["error"]["param"] == "Idempotency-Key"


def test_conflict_does_not_run_the_second_request(api_client, api_settings):
    headers = {"Idempotency-Key": "key-noleak"}
    api_client.post(CHAT_PATH, json=BODY, headers=headers)
    _, before = counts(api_settings)

    api_client.post(CHAT_PATH, json=OTHER_BODY, headers=headers)

    _, after = counts(api_settings)
    assert after == before


def test_different_keys_run_independently(api_client, api_settings):
    api_client.post(CHAT_PATH, json=BODY, headers={"Idempotency-Key": "key-a"})
    api_client.post(CHAT_PATH, json=OTHER_BODY, headers={"Idempotency-Key": "key-b"})

    requests, _ = counts(api_settings)
    assert requests == 2


def test_a_request_without_a_key_is_never_deduplicated(api_client, api_settings):
    """§1 applies idempotency only when the caller supplies a key."""
    api_client.post(CHAT_PATH, json=BODY)
    api_client.post(CHAT_PATH, json=BODY)

    requests, _ = counts(api_settings)
    assert requests == 2


# --- dry run --------------------------------------------------------------


def test_dry_run_returns_an_inspection_document(api_client):
    """§2: the generation endpoint behaves as inspection."""
    response = api_client.post(CHAT_PATH, json={**BODY, "gateway": {"dry_run": True}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "gateway.route_inspection"
    assert "candidates" in payload
    assert "planned_validation" in payload


def test_dry_run_creates_no_rows(api_client, api_settings):
    """Inspection semantics mean no request, attempt, or spend."""
    before = counts(api_settings)

    for _ in range(3):
        api_client.post(CHAT_PATH, json={**BODY, "gateway": {"dry_run": True}})

    assert counts(api_settings) == before == (0, 0)


def test_dry_run_matches_route_inspect(api_client):
    """A dry run and /route/inspect must describe the same request identically."""
    dry = api_client.post(CHAT_PATH, json={**BODY, "gateway": {"dry_run": True}}).json()
    inspected = api_client.post("/route/inspect", json=BODY).json()

    assert dry["classification"] == inspected["classification"]
    assert dry["planned_validation"] == inspected["planned_validation"]
    assert [c["rank"] for c in dry["candidates"]] == [c["rank"] for c in inspected["candidates"]]


def test_dry_run_works_on_the_responses_endpoint(api_client):
    response = api_client.post(
        "/v1/responses",
        json={"model": "auto", "input": "Summarise this.", "gateway": {"dry_run": True}},
    )
    assert response.status_code == 200
    assert response.json()["object"] == "gateway.route_inspection"


# --- privacy --------------------------------------------------------------


def test_prompt_text_is_never_stored(api_client, api_settings):
    """§6: no raw prompts or outputs at rest."""
    secret = "confidential-marker-9f3a2b"
    api_client.post(
        CHAT_PATH, json={"model": "auto", "messages": [{"role": "user", "content": secret}]}
    )

    engine = create_db_engine(api_settings.database_url)
    try:
        with engine.connect() as connection:
            for table in ("requests", "attempts", "validations", "idempotency_records"):
                rows = connection.exec_driver_sql(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                assert secret not in str(rows), f"prompt text leaked into {table}"
    finally:
        engine.dispose()
