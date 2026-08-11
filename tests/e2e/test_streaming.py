"""Streaming, failure, and disconnect (§4, §12 T05).

The first-token boundary is the thing under test throughout: before it, a
failure is an ordinary retryable error; after it, the route is fixed and the
request is terminal.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from sqlalchemy import select

from gateway.domain.enums import Risk
from gateway.domain.errors import ValidationRequiresBufferingError
from gateway.domain.requests import CanonicalRequest, GatewayControls, Message, OutputFormat
from gateway.domain.routing import RequestFeatures
from gateway.persistence.engine import create_db_engine, create_session_factory
from gateway.persistence.models import Attempt, BudgetReservation
from gateway.services.streaming import StreamMode, decide
from tests.conftest import TEST_DEBUG_API_KEY

pytestmark = pytest.mark.e2e

CHAT_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"

CHAT_BODY = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarise this."}],
    "stream": True,
}
RESPONSES_BODY = {"model": "auto", "input": "Summarise this.", "stream": True}


def sse_frames(text: str) -> list[str]:
    """Split a raw SSE body into frames."""
    return [block for block in text.split("\n\n") if block.strip()]


def data_payloads(text: str) -> list[dict]:
    """Every JSON ``data:`` payload, ignoring the ``[DONE]`` terminator."""
    payloads = []
    for block in sse_frames(text):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                continue
            payloads.append(json.loads(body))
    return payloads


def event_names(text: str) -> list[str]:
    names = []
    for block in sse_frames(text):
        for line in block.split("\n"):
            if line.startswith("event: "):
                names.append(line[7:])
    return names


# --- Chat Completions protocol --------------------------------------------


def test_chat_stream_has_the_right_content_type(api_client):
    response = api_client.post(CHAT_PATH, json=CHAT_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_chat_stream_terminates_with_done(api_client):
    """The Chat protocol's terminator; SDKs stop reading on it."""
    body = api_client.post(CHAT_PATH, json=CHAT_BODY).text
    assert body.rstrip().endswith("data: [DONE]")


def test_chat_stream_opens_with_the_assistant_role(api_client):
    payloads = data_payloads(api_client.post(CHAT_PATH, json=CHAT_BODY).text)
    first = payloads[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_chat_stream_delivers_content_and_a_finish_reason(api_client):
    payloads = data_payloads(api_client.post(CHAT_PATH, json=CHAT_BODY).text)

    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in payloads
        if chunk.get("choices")
    )
    finishes = [
        chunk["choices"][0]["finish_reason"]
        for chunk in payloads
        if chunk.get("choices") and chunk["choices"][0].get("finish_reason")
    ]

    assert content
    assert finishes == ["stop"]


def test_chat_stream_reports_usage_at_the_end(api_client):
    payloads = data_payloads(api_client.post(CHAT_PATH, json=CHAT_BODY).text)
    usage_chunks = [chunk for chunk in payloads if "usage" in chunk]

    assert usage_chunks
    assert usage_chunks[-1]["usage"]["total_tokens"] > 0


def test_chat_stream_frames_are_individually_parseable(api_client):
    """A frame containing a raw newline would corrupt everything after it."""
    for payload in data_payloads(api_client.post(CHAT_PATH, json=CHAT_BODY).text):
        assert isinstance(payload, dict)


# --- Responses protocol ----------------------------------------------------


def test_responses_stream_uses_named_events(api_client):
    """The Responses protocol is event-named, not bare data frames."""
    names = event_names(api_client.post(RESPONSES_PATH, json=RESPONSES_BODY).text)

    assert names[0] == "response.created"
    assert "response.output_text.delta" in names
    assert names[-1] == "response.completed"


def test_responses_stream_assembles_the_full_text(api_client):
    body = api_client.post(RESPONSES_PATH, json=RESPONSES_BODY).text
    payloads = data_payloads(body)

    deltas = "".join(
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    )
    completed = next(p for p in payloads if p.get("type") == "response.completed")

    assert deltas
    assert completed["response"]["output_text"] == deltas


def test_responses_stream_has_monotonic_sequence_numbers(api_client):
    payloads = data_payloads(api_client.post(RESPONSES_PATH, json=RESPONSES_BODY).text)
    sequences = [p["sequence_number"] for p in payloads if "sequence_number" in p]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_the_two_protocols_are_not_interchangeable(api_client):
    """Emitting the wrong shape breaks SDKs in ways that look like our bug."""
    chat = api_client.post(CHAT_PATH, json=CHAT_BODY).text
    responses = api_client.post(RESPONSES_PATH, json=RESPONSES_BODY).text

    assert "event: response.created" not in chat
    assert "[DONE]" not in responses


# --- buffer or reject (§4) -------------------------------------------------


def make_features(risk: Risk = Risk.LOW) -> RequestFeatures:
    from gateway.domain.enums import Privacy, TaskClass

    return RequestFeatures(
        task_class=TaskClass.SUMMARIZATION,
        complexity=2,
        risk=risk,
        privacy=Privacy.CONFIDENTIAL,
        verifiability="semi_deterministic",
        freshness_sensitive=False,
        required_capabilities=frozenset(),
        expected_output_tokens=100,
        classifier_name="test",
        classifier_version="v0",
        confidence=0.9,
    )


def make_request() -> CanonicalRequest:
    from gateway.domain.enums import Endpoint

    return CanonicalRequest(
        request_id="req_stream",
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model="auto",
        conversation=(Message(role="user", content="hi"),),
        controls=GatewayControls(),
        client_id="c",
        stream=True,
        output_format=OutputFormat(),
    )


def test_plain_validation_streams_straight_through():
    plan = decide(make_request(), make_features(), ("length_check",))
    assert plan.mode is StreamMode.PASSTHROUGH


def test_full_output_validation_buffers():
    """Cheap deterministic gates buffer rather than refuse the request."""
    plan = decide(make_request(), make_features(), ("schema_check",))
    assert plan.mode is StreamMode.BUFFERED
    assert "schema_check" in plan.buffering_validators


def test_high_risk_buffers_even_without_a_full_output_validator():
    plan = decide(make_request(), make_features(Risk.HIGH), ("length_check",))
    assert plan.mode is StreamMode.BUFFERED


def test_judge_validation_rejects_streaming():
    """§4: reject with validation_requires_buffering rather than hold the
    caller through a generation *and* a judge call."""
    with pytest.raises(ValidationRequiresBufferingError) as excinfo:
        decide(make_request(), make_features(Risk.HIGH), ("independent_review",))

    assert excinfo.value.code == "validation_requires_buffering"
    assert excinfo.value.status_code == 400


def test_streaming_a_judged_request_is_rejected_over_http(api_client):
    """A SQL review plans independent_review, so it cannot stream."""
    response = api_client.post(
        CHAT_PATH,
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Review this SQL: SELECT 1 FROM t"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_requires_buffering"
    assert response.json()["error"]["param"] == "stream"


def test_a_rejected_stream_spends_nothing(api_client, api_settings):
    """The rejection happens before any reservation, so nothing is held."""
    api_client.post(
        CHAT_PATH,
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Review this SQL: SELECT 1 FROM t"}],
            "stream": True,
        },
    )

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            assert session.scalars(select(BudgetReservation)).all() == []
            assert session.scalars(select(Attempt)).all() == []
    finally:
        engine.dispose()


def test_buffered_stream_still_looks_like_a_stream(api_client):
    """A JSON-schema request buffers, but the client still gets SSE framing."""
    response = api_client.post(
        CHAT_PATH,
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "r", "schema": {"type": "object"}},
            },
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.rstrip().endswith("data: [DONE]")


# --- failures --------------------------------------------------------------


def test_a_failure_after_output_is_reported_in_band(api_app, api_settings):
    """§4: status was committed with the headers, so failure cannot be a 5xx."""
    from fastapi.testclient import TestClient

    from gateway.providers.fake import FakeProvider, ScriptedBehaviour, partial_stream_failure

    with TestClient(api_app, raise_server_exceptions=False) as client:
        # Registered inside the context: adapters are wired during lifespan, so
        # app.state does not exist until the client has started the app.
        api_app.state.adapters.register(
            FakeProvider(
                behaviour=ScriptedBehaviour().fail_next(
                    partial_stream_failure(prompt_tokens=10, completion_tokens=4)
                )
            )
        )
        client.headers.update({"Authorization": f"Bearer {TEST_DEBUG_API_KEY}"})
        response = client.post(CHAT_PATH, json=CHAT_BODY)

    assert response.status_code == 200, "headers were already committed"
    payloads = data_payloads(response.text)
    errors = [payload for payload in payloads if "error" in payload]
    assert errors, "the client must be told the stream failed"


def test_streaming_settles_or_releases_on_every_path(api_client, api_settings):
    """§7 holds for streams too: nothing may stay reserved."""
    api_client.post(CHAT_PATH, json=CHAT_BODY)

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            active = [
                row for row in session.scalars(select(BudgetReservation)) if row.status == "ACTIVE"
            ]
            assert active == []
    finally:
        engine.dispose()


def test_streaming_records_an_attempt_marked_as_emitting(api_client, api_settings):
    """``emitted_output`` is what later forbids switching models (§4)."""
    api_client.post(CHAT_PATH, json=CHAT_BODY)

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            attempts = list(session.scalars(select(Attempt)))
            assert attempts
            assert attempts[0].emitted_output is True
            assert attempts[0].cost_actual is not None
            assert attempts[0].cost_actual > Decimal("0")
    finally:
        engine.dispose()


def test_streaming_never_produces_more_than_one_attempt(api_client, api_settings):
    """§4: no fallback or escalation once output is visible."""
    api_client.post(CHAT_PATH, json=CHAT_BODY)

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            assert len(list(session.scalars(select(Attempt)))) == 1
    finally:
        engine.dispose()


def test_streaming_requires_authentication(anonymous_client):
    assert anonymous_client.post(CHAT_PATH, json=CHAT_BODY).status_code == 401


def test_stream_carries_the_request_id_header(api_client):
    response = api_client.post(CHAT_PATH, json=CHAT_BODY)
    assert response.headers["X-LLM-Request-ID"]


def test_stream_does_not_disclose_route_without_scope(api_client):
    payloads = data_payloads(api_client.post(CHAT_PATH, json=CHAT_BODY).text)
    summaries = [payload["gateway"] for payload in payloads if "gateway" in payload]

    assert summaries
    for summary in summaries:
        assert set(summary) == {"request_id"}


def test_stream_discloses_route_with_scope(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app, raise_server_exceptions=False) as client:
        response = client.post(
            CHAT_PATH,
            json=CHAT_BODY,
            headers={"Authorization": f"Bearer {TEST_DEBUG_API_KEY}"},
        )

    summaries = [p["gateway"] for p in data_payloads(response.text) if "gateway" in p]
    assert summaries
    assert summaries[-1]["resolved_model"]
    assert isinstance(summaries[-1]["cost"]["actual"], str)
