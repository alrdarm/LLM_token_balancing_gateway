"""Metrics (§11).

§11 names what must be measured: terminal state, route and model, attempts,
validator result, estimated versus actual cost, reservation failures, latency
by stage, provider errors, circuit state, quota staleness, and redaction
failures.

Two rules shape the implementation:

* **No high-cardinality labels.** §11 says so explicitly, and it is the usual
  way a metrics backend is destroyed: a label carrying a request ID, a client
  ID, or a prompt fragment creates one series per request. Label values are
  restricted to closed vocabularies -- states, outcomes, model IDs -- and
  anything unrecognised is folded into ``other``.
* **In-process and dependency-free.** v0.1 is a single process, and the spec
  defers Prometheus and friends. This exposes a snapshot the health surface
  and tests can read; wiring it to a real backend is a later concern.

``estimated_vs_actual`` is tracked as a ratio rather than two totals because
the interesting signal is systematic *drift*: an estimator that is 30% low
exhausts budgets early no matter how large the absolute numbers are.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

#: Label values permitted for the ``stage`` dimension.
STAGES = (
    "normalize",
    "classify",
    "plan",
    "reserve",
    "invoke",
    "validate",
    "serialize",
    "total",
)

#: Anything outside a known vocabulary collapses here, so a mislabelled call
#: cannot create an unbounded number of series.
OTHER = "other"


@dataclass
class Snapshot:
    """A point-in-time view of every counter and histogram."""

    counters: dict[str, int] = field(default_factory=dict)
    sums: dict[str, Decimal] = field(default_factory=dict)
    latencies: dict[str, list[float]] = field(default_factory=dict)

    def counter(self, name: str, **labels: str) -> int:
        return self.counters.get(_key(name, labels), 0)


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


class Metrics:
    """Thread-safe in-process metrics.

    Locked because reservations and settlements genuinely run on multiple
    threads (the concurrency suite proves it), and a lost increment would make
    the reservation-failure counter -- one of the few numbers an operator would
    page on -- quietly wrong.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._sums: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._latencies: dict[str, list[float]] = defaultdict(list)

    # -- primitives ---------------------------------------------------------

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += amount

    def add(self, name: str, amount: Decimal, **labels: str) -> None:
        with self._lock:
            self._sums[_key(name, labels)] += amount

    def observe_latency(self, stage: str, milliseconds: float) -> None:
        safe_stage = stage if stage in STAGES else OTHER
        with self._lock:
            self._latencies[_key("latency_ms", {"stage": safe_stage})].append(milliseconds)

    # -- the §11 list -------------------------------------------------------

    def request_terminal(self, state: str, *, endpoint: str) -> None:
        self.increment("requests_total", state=state, endpoint=endpoint)

    def attempt(self, *, model_id: str, kind: str, outcome: str) -> None:
        self.increment("attempts_total", model=model_id, kind=kind, outcome=outcome)

    def validation(self, *, validator: str, result: str) -> None:
        self.increment("validations_total", validator=validator, result=result)

    def cost(self, *, estimated: Decimal, actual: Decimal, model_id: str) -> None:
        """Record spend and the estimator's drift.

        A high-severity signal when actual exceeds estimate: §6 requires the
        under-estimate to be settled accurately *and* flagged, because it means
        future reservations are systematically too small.
        """
        self.add("cost_estimated_total", estimated, model=model_id)
        self.add("cost_actual_total", actual, model=model_id)
        if actual > estimated:
            self.increment("cost_underestimated_total", model=model_id)

    def reservation_failed(self, *, reason: str) -> None:
        self.increment("reservation_failures_total", reason=reason)

    def provider_error(self, *, provider: str, outcome: str) -> None:
        self.increment("provider_errors_total", provider=provider, outcome=outcome)

    def circuit_state(self, *, provider: str, state: str) -> None:
        self.increment("circuit_transitions_total", provider=provider, state=state)

    def quota_stale(self, *, provider: str) -> None:
        self.increment("quota_stale_total", provider=provider)

    def redaction_failure(self, *, where: str) -> None:
        """§11 lists this explicitly: a redaction that fails is a security
        event, not a logging inconvenience."""
        self.increment("redaction_failures_total", where=where)

    def stream(self, *, mode: str) -> None:
        self.increment("streams_total", mode=mode)

    # -- reading ------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                counters=dict(self._counters),
                sums=dict(self._sums),
                latencies={name: list(values) for name, values in self._latencies.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._sums.clear()
            self._latencies.clear()


#: Process-wide metrics. Injected where practical; module-level for the paths
#: that would otherwise need threading through every call signature.
metrics = Metrics()
