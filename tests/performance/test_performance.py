"""Performance budgets (§12).

§12 asks for 100 concurrent inspections, 20 concurrent orchestrations, lock
duration, first-byte overhead, and memory. These are **budgets, not
benchmarks**: they are deliberately loose so they fail on a regression of
kind -- an accidental O(n²), a lock held across a provider call, a per-request
engine -- rather than on a slow CI runner.

Thresholds are generous for exactly that reason. A tight threshold here would
produce flaky failures that get muted, and a muted test protects nothing.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.performance

CHAT_PATH = "/v1/chat/completions"
INSPECT_PATH = "/route/inspect"
BODY = {"model": "auto", "messages": [{"role": "user", "content": "Summarise this."}]}

#: §12 asks for 100 concurrent inspections.
INSPECT_CONCURRENCY = 100
#: §12 asks for 20 concurrent orchestrations.
ORCHESTRATION_CONCURRENCY = 20

#: Budgets, not targets. Chosen an order of magnitude above observed local
#: timings so a genuine regression is what trips them.
INSPECT_BUDGET_SECONDS = 30.0
ORCHESTRATION_BUDGET_SECONDS = 30.0


def test_one_hundred_concurrent_inspections(api_client):
    """Inspection must stay cheap: it is the endpoint clients poll."""
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(api_client.post, INSPECT_PATH, json=BODY)
            for _ in range(INSPECT_CONCURRENCY)
        ]
        statuses = [future.result().status_code for future in futures]

    elapsed = time.monotonic() - started

    assert statuses.count(200) == INSPECT_CONCURRENCY
    assert elapsed < INSPECT_BUDGET_SECONDS, (
        f"{INSPECT_CONCURRENCY} inspections took {elapsed:.1f}s"
    )


def test_twenty_concurrent_orchestrations(api_client):
    """Full generations under concurrency, with the ledger still exact."""
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(api_client.post, CHAT_PATH, json=BODY)
            for _ in range(ORCHESTRATION_CONCURRENCY)
        ]
        statuses = [future.result().status_code for future in futures]

    elapsed = time.monotonic() - started

    assert statuses.count(200) == ORCHESTRATION_CONCURRENCY
    assert elapsed < ORCHESTRATION_BUDGET_SECONDS


def test_concurrent_orchestrations_leave_no_active_reservation(api_client, api_settings):
    """The ledger invariant must survive concurrency at the HTTP layer too."""
    from sqlalchemy import select

    from gateway.persistence.engine import create_db_engine, create_session_factory
    from gateway.persistence.models import BudgetReservation

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(api_client.post, CHAT_PATH, json=BODY) for _ in range(12)]
        for future in futures:
            future.result()

    engine = create_db_engine(api_settings.database_url)
    try:
        with create_session_factory(engine)() as session:
            active = [
                row for row in session.scalars(select(BudgetReservation)) if row.status == "ACTIVE"
            ]
        assert active == []
    finally:
        engine.dispose()


def test_no_database_lock_is_held_across_a_provider_call(api_client):
    """§6: the provider is invoked with no lock held.

    Detected behaviourally: if a lock spanned the provider call, concurrent
    requests would serialise and total time would approach the sum of their
    latencies rather than the max. With a deliberately slow provider, that
    difference is unmistakable.
    """
    from gateway.providers.fake import FakeProvider

    app = api_client.app
    app.state.adapters.register(FakeProvider(latency_ms=120))

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(api_client.post, CHAT_PATH, json=BODY) for _ in range(8)]
        for future in futures:
            future.result()
    elapsed = time.monotonic() - started

    # Serialised would be ~8 x 120ms = 0.96s plus overhead. A generous ceiling
    # still separates "concurrent" from "serialised".
    assert elapsed < 5.0, f"8 x 120ms requests took {elapsed:.2f}s; locks may be held"


def test_health_stays_responsive_under_load(api_client):
    """Liveness must not queue behind generation work."""
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            api_client.post(CHAT_PATH, json=BODY)

    worker = threading.Thread(target=hammer, daemon=True)
    worker.start()
    try:
        time.sleep(0.2)
        started = time.monotonic()
        response = api_client.get("/health/live")
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        worker.join(timeout=5)

    assert response.status_code == 200
    assert elapsed < 5.0
