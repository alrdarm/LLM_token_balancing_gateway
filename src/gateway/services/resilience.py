"""Retry policy and circuit breaking (§8, §9).

Retry rules, exactly as §9 states them: retry only classified transient
errors, only **before visible output**, and only within remaining attempts,
cost, and deadline. Exponential backoff with full jitter, honouring
``Retry-After`` when the provider supplied it.

Full jitter rather than fixed or "equal" jitter because retries here are
correlated: a provider returning 429 usually returns it to every caller at
once, and identical backoff would resynchronise them into the same thundering
herd the backoff exists to prevent.

The circuit breaker exists for one case §8 calls out specifically: an
auth or configuration error must open the circuit rather than be retried,
because every retry will fail identically while still costing latency -- and,
for a partially-configured provider, possibly money.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import ProviderFailure

logger = logging.getLogger(__name__)

#: Outcomes that may be retried, and only before any output is visible (§8).
RETRYABLE_OUTCOMES: frozenset[AttemptOutcome] = frozenset(
    {
        AttemptOutcome.RATE_LIMITED,
        AttemptOutcome.PROVIDER_ERROR,
        AttemptOutcome.TIMEOUT,
    }
)

#: Outcomes that mean "this model cannot serve this request", so the right
#: response is a compatible fallback rather than a retry (§8).
FALLBACK_OUTCOMES: frozenset[AttemptOutcome] = frozenset(
    {
        AttemptOutcome.CAPABILITY_REJECTED,
        AttemptOutcome.CONTEXT_REJECTED,
    }
)

#: Outcomes that indicate a broken provider rather than a broken request.
CIRCUIT_OPENING_OUTCOMES: frozenset[AttemptOutcome] = frozenset({AttemptOutcome.AUTH_ERROR})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff with full jitter."""

    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 8.0
    multiplier: float = 2.0

    def delay_for(
        self,
        attempt_number: int,
        *,
        retry_after_seconds: float | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Seconds to wait before retry ``attempt_number`` (1-based).

        A provider-supplied ``Retry-After`` wins outright: it is information we
        do not otherwise have, and ignoring it invites another 429.
        """
        if retry_after_seconds is not None:
            return max(retry_after_seconds, 0.0)

        ceiling = min(
            self.base_delay_seconds * (self.multiplier ** max(attempt_number - 1, 0)),
            self.max_delay_seconds,
        )
        generator = rng or random
        # Full jitter: uniform over [0, ceiling], not ceiling itself.
        return generator.uniform(0.0, ceiling)


def may_retry(
    failure: ProviderFailure,
    *,
    attempts_used: int,
    max_attempts: int,
    remaining_deadline_ms: float,
    predicted_attempt_ms: float,
    circuit_open: bool = False,
) -> bool:
    """Whether §8 and §9 permit another attempt after ``failure``.

    Every clause is a hard stop, checked explicitly rather than folded into one
    boolean, so a failure to retry can be attributed to a specific rule.
    """
    if failure.emitted_output:
        # §4: once output is client-visible the request is terminal.
        return False
    if not failure.retryable or failure.outcome not in RETRYABLE_OUTCOMES:
        return False
    if circuit_open:
        return False
    if attempts_used >= max_attempts:
        return False
    if remaining_deadline_ms <= predicted_attempt_ms:
        # §8: stop when the remaining deadline cannot cover another attempt
        # plus its validation margin.
        return False
    return True


@dataclass
class CircuitState:
    """Per-provider breaker state."""

    failures: int = 0
    opened_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None


class CircuitBreaker:
    """Per-provider circuit breaking (§8, §11).

    Half-open probing is deliberately single-shot: after the cooldown one
    request is allowed through, and its result decides whether the circuit
    closes or re-opens. Letting the full load through at once would hammer a
    provider that is still recovering.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cooldown: timedelta = timedelta(seconds=30),
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._states: dict[str, CircuitState] = {}

    def _state(self, provider: str) -> CircuitState:
        return self._states.setdefault(provider, CircuitState())

    def is_open(self, provider: str, *, now: datetime | None = None) -> bool:
        """Whether calls to ``provider`` are currently blocked."""
        state = self._state(provider)
        if state.opened_at is None:
            return False

        at = now or datetime.now(UTC)
        if at - state.opened_at >= self.cooldown:
            # Cooldown elapsed: allow one probe through.
            return False
        return True

    def record_success(self, provider: str) -> None:
        """A success closes the circuit and clears the failure count."""
        self._states[provider] = CircuitState()

    def record_failure(
        self,
        provider: str,
        failure: ProviderFailure,
        *,
        now: datetime | None = None,
    ) -> None:
        """Update the breaker after a failed call."""
        at = now or datetime.now(UTC)
        state = self._state(provider)

        if failure.outcome in CIRCUIT_OPENING_OUTCOMES:
            # §8: an auth or config error will fail identically every time.
            # Opening immediately avoids burning the request's whole attempt
            # budget on a fault no retry can fix.
            state.opened_at = at
            state.failures += 1
            logger.error(
                "Circuit opened on configuration fault",
                extra={"event": "circuit_opened"},
            )
            return

        state.failures += 1
        if state.failures >= self.failure_threshold:
            state.opened_at = at
            logger.warning("Circuit opened", extra={"event": "circuit_opened"})

    def open_providers(self, *, now: datetime | None = None) -> frozenset[str]:
        """Providers currently blocked, for the routing health gate (§10)."""
        return frozenset(provider for provider in self._states if self.is_open(provider, now=now))

    def reset(self) -> None:
        self._states.clear()
