"""Retry rules and circuit breaking (§8, §9)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import ProviderFailure
from gateway.providers.fake import (
    auth_error,
    partial_stream_failure,
    rate_limited,
    timeout_before_output,
    transient_server_error,
)
from gateway.services.resilience import CircuitBreaker, RetryPolicy, may_retry

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def permits(failure: ProviderFailure, **overrides: object) -> bool:
    defaults: dict[str, object] = {
        "attempts_used": 1,
        "max_attempts": 3,
        "remaining_deadline_ms": 10_000.0,
        "predicted_attempt_ms": 1_000.0,
        "circuit_open": False,
    }
    defaults.update(overrides)
    return may_retry(failure, **defaults)  # type: ignore[arg-type]


# --- what may be retried ---------------------------------------------------


@pytest.mark.parametrize(
    "failure", [rate_limited(), transient_server_error(), timeout_before_output()]
)
def test_transient_failures_before_output_may_retry(failure):
    assert permits(failure)


def test_auth_error_is_never_retried():
    """§8: no blind retry on a configuration fault."""
    assert not permits(auth_error())


def test_failure_after_visible_output_is_terminal():
    """§4: once output is client-visible the request cannot be retried."""
    assert not permits(partial_stream_failure(prompt_tokens=10, completion_tokens=5))


def test_attempt_cap_stops_retrying():
    assert not permits(rate_limited(), attempts_used=3, max_attempts=3)


def test_open_circuit_stops_retrying():
    assert not permits(rate_limited(), circuit_open=True)


def test_insufficient_deadline_stops_retrying():
    """§8: stop when the remaining deadline cannot cover another attempt."""
    assert not permits(rate_limited(), remaining_deadline_ms=500.0, predicted_attempt_ms=1000.0)


def test_deadline_exactly_equal_to_the_attempt_stops_retrying():
    """No margin left for validation, so the attempt would be wasted spend."""
    assert not permits(rate_limited(), remaining_deadline_ms=1000.0, predicted_attempt_ms=1000.0)


def test_a_failure_marked_unretryable_is_not_retried_even_if_transient():
    stubborn = ProviderFailure("no", outcome=AttemptOutcome.PROVIDER_ERROR, retryable=False)
    assert not permits(stubborn)


# --- backoff ---------------------------------------------------------------


def test_backoff_respects_retry_after():
    """A provider-supplied wait is information we do not otherwise have."""
    policy = RetryPolicy()
    assert policy.delay_for(1, retry_after_seconds=2.5) == 2.5


def test_negative_retry_after_is_clamped():
    assert RetryPolicy().delay_for(1, retry_after_seconds=-5) == 0.0


def test_backoff_ceiling_grows_exponentially():
    policy = RetryPolicy(base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=100.0)
    generator = random.Random(0)  # noqa: S311 - jitter distribution, not crypto

    # Full jitter samples [0, ceiling], so compare the observed maxima.
    first = max(policy.delay_for(1, rng=generator) for _ in range(200))
    third = max(policy.delay_for(3, rng=generator) for _ in range(200))
    assert third > first


def test_backoff_is_capped():
    policy = RetryPolicy(base_delay_seconds=1.0, multiplier=10.0, max_delay_seconds=5.0)
    for _ in range(100):
        assert policy.delay_for(9) <= 5.0


def test_full_jitter_spreads_retries():
    """Identical backoff would resynchronise callers into a thundering herd."""
    policy = RetryPolicy(base_delay_seconds=4.0)
    delays = {round(policy.delay_for(2), 6) for _ in range(50)}
    assert len(delays) > 10, "delays are not jittered"


def test_backoff_is_never_negative():
    policy = RetryPolicy()
    assert all(policy.delay_for(attempt) >= 0 for attempt in range(1, 8))


# --- circuit breaker -------------------------------------------------------


def test_circuit_starts_closed():
    assert not CircuitBreaker().is_open("fake", now=NOW)


def test_auth_error_opens_the_circuit_immediately():
    """§8: every retry would fail identically, so stop at once."""
    breaker = CircuitBreaker(failure_threshold=5)
    breaker.record_failure("fake", auth_error(), now=NOW)
    assert breaker.is_open("fake", now=NOW)


def test_transient_failures_open_only_at_the_threshold():
    breaker = CircuitBreaker(failure_threshold=3)

    breaker.record_failure("fake", transient_server_error(), now=NOW)
    breaker.record_failure("fake", transient_server_error(), now=NOW)
    assert not breaker.is_open("fake", now=NOW)

    breaker.record_failure("fake", transient_server_error(), now=NOW)
    assert breaker.is_open("fake", now=NOW)


def test_success_closes_the_circuit_and_clears_the_count():
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure("fake", transient_server_error(), now=NOW)
    breaker.record_success("fake")
    breaker.record_failure("fake", transient_server_error(), now=NOW)
    assert not breaker.is_open("fake", now=NOW)


def test_circuit_half_opens_after_the_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown=timedelta(seconds=30))
    breaker.record_failure("fake", transient_server_error(), now=NOW)

    assert breaker.is_open("fake", now=NOW + timedelta(seconds=29))
    assert not breaker.is_open("fake", now=NOW + timedelta(seconds=31))


def test_circuits_are_isolated_per_provider():
    """One broken provider must not disable the others."""
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure("fake", auth_error(), now=NOW)

    assert breaker.is_open("fake", now=NOW)
    assert not breaker.is_open("alien", now=NOW)


def test_open_providers_feeds_the_routing_health_gate():
    """§10 step 7 excludes unhealthy providers from ranking."""
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure("fake", auth_error(), now=NOW)
    breaker.record_failure("alien", transient_server_error(), now=NOW)

    assert breaker.open_providers(now=NOW) == frozenset({"fake", "alien"})
    assert breaker.open_providers(now=NOW + timedelta(minutes=5)) == frozenset()


def test_reset_clears_every_circuit():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure("fake", auth_error(), now=NOW)
    breaker.reset()
    assert not breaker.is_open("fake", now=NOW)
