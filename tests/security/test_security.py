"""Security coverage (§11, §12).

§12 requires: scopes, size limits, SSRF allowlist, redaction snapshots, CLI
injection, secret scan. Redaction snapshots live in ``tests/unit`` and CLI
injection in ``tests/integration``; this file covers the rest and adds a
repository-wide secret scan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from gateway.domain.errors import PolicyDeniedError
from gateway.providers.openai_http import OpenAICompatibleAdapter, OutboundNotAllowedError
from tests.conftest import TEST_API_KEY, TEST_DEBUG_API_KEY

pytestmark = pytest.mark.security

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_PATH = "/v1/chat/completions"
BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


# --- scopes ----------------------------------------------------------------


def test_debug_requires_a_scope(api_client):
    response = api_client.post(CHAT_PATH, json={**BODY, "gateway": {"debug": True}})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "policy_denied"


def test_scope_check_raises_rather_than_downgrading():
    """Silently dropping a privilege is worse than refusing it."""
    from gateway.api.auth import SCOPE_ADMIN, AuthenticatedClient

    client = AuthenticatedClient(client_id="c", scopes=frozenset(), control_overrides={})
    with pytest.raises(PolicyDeniedError):
        client.require_scope(SCOPE_ADMIN)


def test_route_disclosure_is_scope_gated(api_client):
    block = api_client.post(CHAT_PATH, json=BODY).json()["gateway"]
    assert set(block) == {"request_id"}


# --- size limits (§11: apply size limits early) ---------------------------


def test_oversized_body_is_rejected(api_client):
    from gateway.api.dependencies import MAX_BODY_BYTES

    huge = "x" * (MAX_BODY_BYTES + 1024)
    response = api_client.post(
        CHAT_PATH,
        content=f'{{"model":"auto","messages":[{{"role":"user","content":"{huge}"}}]}}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_oversized_metadata_is_rejected(api_client):
    response = api_client.post(
        CHAT_PATH,
        json={**BODY, "gateway": {"metadata": {str(i): "x" * 600 for i in range(3)}}},
    )
    assert response.status_code == 400


def test_oversized_request_id_is_replaced_not_echoed(api_client):
    """An unbounded ID would bloat every log line it appears in."""
    response = api_client.post(CHAT_PATH, json=BODY, headers={"X-Request-ID": "x" * 5000})
    assert len(response.headers["X-LLM-Request-ID"]) < 200


# --- SSRF allowlist --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/v1",
        "http://169.254.169.254/latest/meta-data",
        "https://internal.corp.local/v1",
    ],
)
def test_outbound_allowlist_blocks_unlisted_hosts(url):
    """§11: allowlist outbound destinations.

    The metadata-service address is included deliberately: it is the classic
    SSRF target, and a gateway that will POST a prompt anywhere is a proxy for
    reaching it.
    """
    with pytest.raises(OutboundNotAllowedError):
        OpenAICompatibleAdapter(
            base_url=url, api_key="k", allowed_hosts=frozenset({"api.openai.com"})
        )


def test_plaintext_is_refused_for_remote_hosts():
    """TLS verification is required; plaintext exposes prompts in transit."""
    with pytest.raises(OutboundNotAllowedError):
        OpenAICompatibleAdapter(base_url="http://api.openai.com/v1", api_key="k")


def test_non_http_schemes_are_refused():
    for url in ("file:///etc/passwd", "gopher://x/", "ftp://x/"):
        with pytest.raises(OutboundNotAllowedError):
            OpenAICompatibleAdapter(base_url=url, api_key="k")


# --- credential handling ---------------------------------------------------


def test_api_keys_are_never_returned_in_errors(anonymous_client):
    secret = "sk-super-secret-value-123456"
    response = anonymous_client.post(
        CHAT_PATH, json=BODY, headers={"Authorization": f"Bearer {secret}"}
    )
    assert secret not in response.text


def test_all_authentication_failures_look_identical(anonymous_client):
    """Distinguishable failures let an attacker enumerate valid keys."""
    bodies = set()
    for header in ("Bearer wrong", "Bearer ", "Basic abc", ""):
        response = anonymous_client.post(CHAT_PATH, json=BODY, headers={"Authorization": header})
        payload = response.json()
        bodies.add((response.status_code, payload["error"]["code"]))

    assert bodies == {(401, "invalid_api_key")}


def test_stored_keys_are_digests_not_plaintext(api_app, api_settings):
    from sqlalchemy import select

    from gateway.persistence.engine import create_db_engine, create_session_factory
    from gateway.persistence.models import APIKey

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            for row in session.scalars(select(APIKey)):
                assert row.key_hash != TEST_API_KEY
                assert row.key_hash != TEST_DEBUG_API_KEY
                assert len(row.key_hash) == 64
    finally:
        engine.dispose()


# --- repository secret scan ------------------------------------------------


#: Patterns that would indicate a committed credential.
SECRET_SCAN = (
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

#: Files that legitimately contain secret-shaped strings: the redaction tests
#: must contain examples in order to prove they are redacted.
SCAN_EXEMPT = {
    "tests/unit/test_redaction.py",
    "tests/security/test_security.py",
    "src/gateway/telemetry/redaction.py",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_no_credentials_are_committed():
    """§12 requires a secret scan. Runs over tracked files only."""
    offenders: list[str] = []

    for relative in tracked_files():
        if relative in SCAN_EXEMPT or relative.endswith((".pdf", ".png", ".jpg")):
            continue
        path = REPO_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue

        for pattern in SECRET_SCAN:
            if pattern.search(text):
                offenders.append(f"{relative}: {pattern.pattern}")

    assert not offenders, f"possible committed credentials: {offenders}"


def test_env_file_is_not_tracked():
    """§10: never commit .env."""
    assert ".env" not in tracked_files()


def test_env_example_contains_no_real_secret():
    example = (REPO_ROOT / ".env.example").read_text()
    for pattern in SECRET_SCAN:
        assert not pattern.search(example)
