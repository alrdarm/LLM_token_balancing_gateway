"""Streaming policy and the first-token boundary (§4).

Streaming is where the gateway's guarantees and its transport collide, because
once a byte is visible to the client it cannot be taken back. §4's invariants:

* Classification, planning, eligibility, and **reservation** all finish before
  response headers are committed. Headers are a commitment; a 200 that later
  turns out to be unaffordable cannot be retracted.
* After any provider output is client-visible, the gateway **MUST NOT** switch
  models. Failure past that point is terminal ``FAILED_PARTIAL``.
* Failure *before* the first visible token may retry or fall back normally.
* High-risk or full-output validation must **buffer** the stream or reject it
  with ``validation_requires_buffering``.
* Disconnect cancels the adapter, settles known usage, and releases the rest.

The buffer-or-reject choice is made by policy rather than fixed, because both
options are bad in different ways. Buffering costs the client the latency
streaming exists to hide; rejecting means a high-risk request cannot stream at
all. This module buffers when validation is cheap and deterministic, and
rejects only when a gate genuinely needs the whole output *and* cannot run
without a judge — where buffering would trade the stream's whole benefit for a
check that may itself take seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from gateway.domain.enums import Risk
from gateway.domain.errors import ValidationRequiresBufferingError
from gateway.domain.requests import CanonicalRequest
from gateway.domain.routing import RequestFeatures

logger = logging.getLogger(__name__)

#: Validators that need the complete output before they can judge it, but are
#: cheap enough that buffering is an acceptable trade.
FULL_OUTPUT_VALIDATORS: frozenset[str] = frozenset(
    {"schema_check", "json_parse", "sql_parser", "sql_safety", "code_compile", "citation_check"}
)

#: Validators that need the whole output *and* a provider call. Buffering here
#: would hold the client through a full generation plus a judge round-trip.
JUDGE_VALIDATORS: frozenset[str] = frozenset({"independent_review", "rubric_judge"})


class StreamMode(StrEnum):
    """How a streaming request will actually be served."""

    #: Deltas forwarded as they arrive; no gate needs the full output.
    PASSTHROUGH = "passthrough"
    #: Generated fully, validated, then emitted as a stream. The client sees
    #: SSE framing but not incremental latency.
    BUFFERED = "buffered"


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """The decision about how to serve one streaming request."""

    mode: StreamMode
    #: Validators that forced buffering, for telemetry and the debug block.
    buffering_validators: tuple[str, ...] = ()

    @property
    def buffers(self) -> bool:
        return self.mode is StreamMode.BUFFERED


def decide(
    request: CanonicalRequest,
    features: RequestFeatures,
    validation_plan: tuple[str, ...],
) -> StreamPlan:
    """Choose passthrough, buffering, or rejection for a streaming request.

    Raises :class:`ValidationRequiresBufferingError` (400,
    ``validation_requires_buffering``) when a required gate cannot be satisfied
    without holding the entire response *and* costs a provider call.
    """
    planned = set(validation_plan)

    judges = planned & JUDGE_VALIDATORS
    if judges:
        # §4 permits rejecting rather than buffering. Rejecting is the honest
        # answer here: buffering would make the caller wait for a full
        # generation *and* a judge call while pretending to stream, and the
        # caller can choose to retry without streaming.
        logger.info(
            "Rejected streaming that requires judge validation",
            extra={"event": "validation_requires_buffering"},
        )
        raise ValidationRequiresBufferingError(
            "This request requires independent validation of the complete "
            "response, which cannot be streamed. Retry with stream=false.",
            param="stream",
        )

    full_output = planned & FULL_OUTPUT_VALIDATORS
    if full_output:
        # Cheap, deterministic, and local: buffering costs milliseconds, so it
        # beats refusing the request outright.
        return StreamPlan(
            mode=StreamMode.BUFFERED,
            buffering_validators=tuple(sorted(full_output)),
        )

    if features.risk in (Risk.HIGH, Risk.CRITICAL):
        # §4 names high risk explicitly. Even with no full-output validator
        # planned, a high-risk answer should not reach the client unchecked.
        return StreamPlan(mode=StreamMode.BUFFERED, buffering_validators=("risk_floor",))

    return StreamPlan(mode=StreamMode.PASSTHROUGH)
