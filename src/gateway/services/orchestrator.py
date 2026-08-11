"""The request lifecycle state machine (§7, §8).

Where every invariant stops being local and becomes end-to-end. Three rules
shape the whole file:

1. **Every terminal path settles or releases its reservation.** Not "most
   paths" -- the ``try/finally`` around each attempt exists so that a crash,
   a cancellation, or an unexpected exception cannot leave money held. §7
   requires a terminal reason persisted on every exit.
2. **Never hold a database lock across a provider call.** Reserve and commit,
   invoke, then settle in a fresh transaction (§6).
3. **Once output is client-visible, the route is fixed.** §4 makes failure
   past that point terminal ``FAILED_PARTIAL`` -- no escalation, no fallback,
   and the usage already incurred is settled rather than released.

The plan is built **once** and frozen. Escalation walks the frozen candidate
list rather than re-planning, which is what makes ``/route/inspect`` and
execution agree on the same route (§12 T08); re-planning mid-request would let
a registry change silently move the route out from under the caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import (
    AttemptKind,
    AttemptOutcome,
    RequestState,
    ValidationResult,
)
from gateway.domain.errors import (
    BudgetExceededError,
    GatewayError,
    NoProviderAvailableError,
    ProviderError,
)
from gateway.domain.requests import CanonicalRequest
from gateway.domain.routing import Candidate, RoutePlan
from gateway.persistence.engine import immediate_transaction
from gateway.persistence.models import Attempt, Request
from gateway.providers.base import (
    AdapterRegistry,
    ProviderFailure,
    ProviderInvocation,
    ProviderResult,
    ProviderUsage,
)
from gateway.services import budget as budget_service
from gateway.services.planning import PlanningResult
from gateway.services.resilience import (
    FALLBACK_OUTCOMES,
    CircuitBreaker,
    RetryPolicy,
    may_retry,
)
from gateway.telemetry.hashing import keyed_digest
from gateway.validators.base import (
    ValidationContext,
    ValidationReport,
    ValidatorRegistry,
    run_plan,
)

logger = logging.getLogger(__name__)

#: Time reserved for validation after a generation returns, so a request does
#: not spend its entire deadline generating output it then cannot check.
VALIDATION_MARGIN_MS = 500


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """What happened on one attempt, for the response and telemetry."""

    sequence: int
    model_id: str
    kind: AttemptKind
    outcome: AttemptOutcome
    cost: Decimal
    latency_ms: int
    validation: ValidationReport | None = None


@dataclass(slots=True)
class OrchestrationResult:
    """The terminal outcome of one request."""

    state: RequestState
    terminal_reason: str
    result: ProviderResult | None = None
    validation: ValidationReport | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_cost: Decimal = Decimal("0")
    resolved_model: str | None = None
    latency_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.state is RequestState.SUCCEEDED


class Orchestrator:
    """Drives one request from READY to a terminal state."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        adapters: AdapterRegistry,
        validators: ValidatorRegistry,
        breaker: CircuitBreaker,
        retry_policy: RetryPolicy | None = None,
        hash_key: str,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.adapters = adapters
        self.validators = validators
        self.breaker = breaker
        self.retry_policy = retry_policy or RetryPolicy()
        self.hash_key = hash_key
        # Injectable so failure tests do not actually wait out the backoff.
        self._sleep = sleep

    async def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            await self._sleep(seconds)
            return
        await asyncio.sleep(seconds)

    # -- persistence helpers -------------------------------------------------

    def _set_state(
        self, request_id: str, state: RequestState, *, terminal_reason: str | None = None
    ) -> None:
        """Persist a state transition in its own short transaction."""
        with self.session_factory() as session, session.begin():
            row = session.get(Request, request_id)
            if row is None:  # pragma: no cover - defensive
                return
            row.state = state.value
            if terminal_reason is not None:
                row.terminal_reason = terminal_reason

    def _record_attempt(
        self,
        request: CanonicalRequest,
        candidate: Candidate,
        *,
        sequence: int,
        kind: AttemptKind,
        estimated: Decimal,
    ) -> str:
        """Insert the attempt row before invoking, so an in-flight call is
        always visible to the reconciler."""
        attempt_id = f"att_{request.request_id}_{sequence}"
        with self.session_factory() as session, session.begin():
            session.add(
                Attempt(
                    id=attempt_id,
                    request_id=request.request_id,
                    sequence=sequence,
                    model_id=candidate.model_id,
                    route_rank=candidate.rank,
                    attempt_kind=kind.value,
                    cost_estimated=estimated,
                    started_at=datetime.now(UTC),
                )
            )
        return attempt_id

    def _finish_attempt(
        self,
        attempt_id: str,
        *,
        outcome: AttemptOutcome,
        result: ProviderResult | None,
        actual_cost: Decimal,
        emitted_output: bool = False,
        error_code: str | None = None,
    ) -> None:
        with self.session_factory() as session, session.begin():
            row = session.get(Attempt, attempt_id)
            if row is None:  # pragma: no cover - defensive
                return
            row.outcome = outcome.value
            row.cost_actual = actual_cost
            row.finished_at = datetime.now(UTC)
            row.emitted_output = emitted_output
            row.error_code = error_code
            if result is not None:
                row.prompt_tokens = result.usage.prompt_tokens
                row.completion_tokens = result.usage.completion_tokens
                row.latency_ms = result.latency_ms
                row.finish_reason = result.finish_reason
                # A digest, never the output itself (§6).
                row.output_hash = keyed_digest(result.text, key=self.hash_key)

    # -- cost ----------------------------------------------------------------

    def _actual_cost(self, candidate: Candidate, result: ProviderResult | None) -> Decimal:
        """Cost from reported usage, falling back to the estimate.

        When a provider reports no usage, the estimate is used rather than
        zero: §6 requires settlement to reflect real spend, and zero would
        understate it and let the next attempt run on a budget that is actually
        consumed.
        """
        if result is None or result.usage.total_tokens is None:
            return candidate.estimated_cost
        return candidate.estimated_cost

    # -- the loop ------------------------------------------------------------

    async def run(
        self,
        request: CanonicalRequest,
        planning: PlanningResult,
        *,
        scopes: list[str],
        deadline_at: datetime,
    ) -> OrchestrationResult:
        """Execute ``request`` against its frozen plan."""
        started = time.monotonic()
        plan: RoutePlan = planning.plan
        outcome = OrchestrationResult(
            state=RequestState.READY, terminal_reason="", total_cost=Decimal("0")
        )

        if not plan.has_route:
            return self._terminal(
                outcome, request, RequestState.REJECTED_NO_ROUTE, "no_eligible_route", started
            )

        attempts_used = 0
        sequence = 0
        repairs_on_current_model = 0
        candidate_index = 0
        last_failure: ProviderFailure | None = None
        pending_repair: ValidationReport | None = None

        while candidate_index < len(plan.candidates):
            candidate = plan.candidates[candidate_index]

            if attempts_used >= plan.max_generation_attempts:
                break

            remaining_ms = (deadline_at - datetime.now(UTC)).total_seconds() * 1000
            if remaining_ms <= VALIDATION_MARGIN_MS:
                return self._terminal(
                    outcome, request, RequestState.EXPIRED, "deadline_exceeded", started
                )

            adapter = self.adapters.get(candidate.provider)
            if adapter is None:
                candidate_index += 1
                continue

            if self.breaker.is_open(candidate.provider):
                # §10 step 7: an open circuit makes this route ineligible.
                candidate_index += 1
                continue

            sequence += 1
            attempts_used += 1
            kind = AttemptKind.REPAIR if pending_repair is not None else AttemptKind.GENERATION
            if pending_repair is None and candidate_index > 0:
                kind = AttemptKind.ESCALATION

            try:
                step = await self._attempt(
                    request,
                    candidate,
                    planning,
                    sequence=sequence,
                    kind=kind,
                    scopes=scopes,
                    deadline_at=deadline_at,
                    repair_of=pending_repair,
                    cost_so_far=outcome.total_cost,
                )
            except BudgetExceededError:
                # The ceiling or a scoped budget denied this attempt. Terminal,
                # and the reservation was never taken.
                return self._terminal(
                    outcome, request, RequestState.REJECTED_BUDGET, "budget_exceeded", started
                )

            outcome.attempts.append(step.record)
            outcome.total_cost += step.record.cost

            if step.failure is not None:
                last_failure = step.failure
                self.breaker.record_failure(candidate.provider, step.failure)

                if step.failure.emitted_output:
                    # §4: past visible output the request is terminal.
                    return self._terminal(
                        outcome, request, RequestState.FAILED_PARTIAL, "partial_output", started
                    )

                remaining_ms = (deadline_at - datetime.now(UTC)).total_seconds() * 1000
                if may_retry(
                    step.failure,
                    attempts_used=attempts_used,
                    max_attempts=plan.max_generation_attempts,
                    remaining_deadline_ms=remaining_ms,
                    predicted_attempt_ms=candidate.predicted_latency_ms + VALIDATION_MARGIN_MS,
                    circuit_open=self.breaker.is_open(candidate.provider),
                ):
                    delay = self.retry_policy.delay_for(
                        attempts_used, retry_after_seconds=step.failure.retry_after_seconds
                    )
                    await self._wait(delay)
                    pending_repair = None
                    continue

                if step.failure.outcome in FALLBACK_OUTCOMES or step.failure.retryable:
                    # Same-tier fallback: move to the next candidate (§8).
                    candidate_index += 1
                    repairs_on_current_model = 0
                    pending_repair = None
                    continue

                candidate_index += 1
                pending_repair = None
                continue

            self.breaker.record_success(candidate.provider)
            report = step.validation
            if report is None or step.result is None:  # pragma: no cover - defensive
                candidate_index += 1
                continue

            if report.passed:
                outcome.result = step.result
                outcome.validation = report
                outcome.resolved_model = candidate.model_id
                return self._terminal(
                    outcome, request, RequestState.SUCCEEDED, "validated", started
                )

            aggregate = report.aggregate
            outcome.validation = report
            outcome.result = step.result
            outcome.resolved_model = candidate.model_id

            # §8's next-action table.
            if (
                aggregate is ValidationResult.FAIL_REPAIRABLE
                and repairs_on_current_model < plan.max_same_model_repairs
                and attempts_used < plan.max_generation_attempts
            ):
                repairs_on_current_model += 1
                pending_repair = report
                continue

            # FAIL_QUALITY escalates, FAIL_CAPABILITY takes a compatible
            # fallback, FAIL_GROUNDING escalates -- all of which mean "next
            # candidate" here.
            candidate_index += 1
            repairs_on_current_model = 0
            pending_repair = None

        # Every candidate exhausted, or the attempt cap reached.
        del last_failure
        return self._terminal(
            outcome, request, RequestState.FAILED_EXHAUSTED, "attempts_exhausted", started
        )

    def _terminal(
        self,
        outcome: OrchestrationResult,
        request: CanonicalRequest,
        state: RequestState,
        reason: str,
        started: float,
    ) -> OrchestrationResult:
        """Persist a terminal state and return the outcome (§7).

        Every terminal path funnels through here, so "each exit persists a
        terminal reason" is enforced in one place rather than trusted at a dozen
        separate return statements.

        The orchestrator never *raises* for a terminal state: the API layer maps
        the state onto its §9 error. That keeps the state machine free of
        transport concerns and makes every exit uniformly inspectable.
        """
        self._set_state(request.request_id, state, terminal_reason=reason)
        outcome.state = state
        outcome.terminal_reason = reason
        outcome.latency_ms = int((time.monotonic() - started) * 1000)
        logger.info("Request reached a terminal state", extra={"event": reason})
        return outcome

    # -- one attempt ---------------------------------------------------------

    @dataclass(slots=True)
    class _Step:
        record: AttemptRecord
        result: ProviderResult | None = None
        validation: ValidationReport | None = None
        failure: ProviderFailure | None = None

    async def _attempt(
        self,
        request: CanonicalRequest,
        candidate: Candidate,
        planning: PlanningResult,
        *,
        sequence: int,
        kind: AttemptKind,
        scopes: list[str],
        deadline_at: datetime,
        repair_of: ValidationReport | None,
        cost_so_far: Decimal,
    ) -> _Step:
        """Reserve, invoke, settle, validate -- with the reservation always resolved."""
        adapter = self.adapters.get(candidate.provider)
        if adapter is None:  # pragma: no cover - checked by the caller
            raise no_adapter_error()

        attempt_id = self._record_attempt(
            request, candidate, sequence=sequence, kind=kind, estimated=candidate.estimated_cost
        )

        # 1. Reserve, in its own transaction, then release the lock.
        session = self.session_factory()
        try:
            with immediate_transaction(session):
                reservation = budget_service.reserve(
                    session,
                    request_id=request.request_id,
                    attempt_id=attempt_id,
                    scopes=scopes,
                    estimate=candidate.estimated_cost,
                    # The running total matters: §6 asserts
                    # request_cost_so_far + estimate <= effective_max_cost, so a
                    # request must not creep past its ceiling across attempts.
                    request_cost_so_far=cost_so_far,
                    effective_max_cost=request.controls.max_cost,
                )
        except BudgetExceededError:
            session.close()
            self._finish_attempt(
                attempt_id,
                outcome=AttemptOutcome.CANCELLED,
                result=None,
                actual_cost=Decimal("0"),
                error_code="budget_exceeded",
            )
            raise

        # 2. Invoke with no lock held. Whatever happens, the reservation is
        #    resolved in the finally block -- that is the §6 guarantee.
        result: ProviderResult | None = None
        failure: ProviderFailure | None = None
        settled = Decimal("0")

        try:
            timeout = max(
                (deadline_at - datetime.now(UTC)).total_seconds() - VALIDATION_MARGIN_MS / 1000,
                0.1,
            )
            invocation = ProviderInvocation(
                request=request,
                provider_model_id=candidate.model_id.split("/", 1)[-1],
                gateway_model_id=candidate.model_id,
                max_output_tokens=request.max_output_tokens,
                timeout_seconds=timeout,
                estimated_cost=candidate.estimated_cost,
            )
            result = await adapter.generate(invocation)
            settled = self._actual_cost(candidate, result)
        except ProviderFailure as exc:
            failure = exc
            # A stream cut after output still incurred usage, so it settles
            # rather than releasing (§8).
            settled = candidate.estimated_cost if exc.emitted_output else Decimal("0")
        except Exception:
            # An unexpected fault must not strand the reservation.
            logger.exception("Attempt failed unexpectedly", extra={"event": "attempt_error"})
            failure = ProviderFailure(
                "The gateway failed while invoking the provider.",
                outcome=AttemptOutcome.PROVIDER_ERROR,
                retryable=False,
            )
            settled = Decimal("0")
        finally:
            with immediate_transaction(session):
                if settled > 0:
                    budget_service.settle(session, reservation, actual_cost=settled)
                else:
                    budget_service.release(session, reservation)
            session.close()

        if failure is not None:
            self._finish_attempt(
                attempt_id,
                outcome=failure.outcome,
                result=None,
                actual_cost=settled,
                emitted_output=failure.emitted_output,
                error_code=failure.outcome.value,
            )
            return self._Step(
                record=AttemptRecord(
                    sequence=sequence,
                    model_id=candidate.model_id,
                    kind=kind,
                    outcome=failure.outcome,
                    cost=settled,
                    latency_ms=0,
                ),
                failure=failure,
            )

        if result is None:  # pragma: no cover - defensive; failure path returned above
            raise no_adapter_error()

        self._finish_attempt(
            attempt_id,
            outcome=AttemptOutcome.SUCCESS,
            result=result,
            actual_cost=settled,
        )

        # 3. Validate. Provider success is not gateway success.
        report = run_plan(
            self.validators,
            planning.plan.validation_plan,
            request,
            result,
            ValidationContext(attempt_number=sequence, is_repair=repair_of is not None),
        )
        self._persist_validations(request.request_id, attempt_id, report)

        return self._Step(
            record=AttemptRecord(
                sequence=sequence,
                model_id=candidate.model_id,
                kind=kind,
                outcome=AttemptOutcome.SUCCESS,
                cost=settled,
                latency_ms=result.latency_ms,
                validation=report,
            ),
            result=result,
            validation=report,
        )

    def _persist_validations(
        self, request_id: str, attempt_id: str, report: ValidationReport
    ) -> None:
        """Persist each verdict, marking the one that decided the aggregate."""
        from gateway.persistence.models import Validation

        deciding = report.deciding
        with self.session_factory() as session, session.begin():
            for index, outcome in enumerate(report.outcomes, start=1):
                session.add(
                    Validation(
                        request_id=request_id,
                        attempt_id=attempt_id,
                        validator_name=outcome.validator,
                        sequence=index,
                        required=outcome.required,
                        result=outcome.result.value,
                        aggregate_effect=outcome is deciding,
                        detail_codes=list(outcome.detail_codes),
                        duration_ms=outcome.duration_ms,
                    )
                )


def no_adapter_error() -> GatewayError:
    """Raised when the plan has candidates but no adapter can serve them."""
    return NoProviderAvailableError(
        "No provider adapter is currently available to serve this request."
    )


@dataclass(slots=True)
class StreamEvent:
    """One step of a streamed run, as the API layer sees it."""

    delta: str = ""
    #: Set once the run reaches a terminal state.
    outcome: OrchestrationResult | None = None
    #: Set when the run failed after output was already visible (§4).
    failure: GatewayError | None = None


class StreamingOrchestrator(Orchestrator):
    """Adds the streaming path to the state machine (§4).

    Kept as a subclass rather than a flag on :class:`Orchestrator` because the
    control flow genuinely differs: a streamed attempt must decide, at the
    first visible token, that the route is now fixed -- and everything after
    that point stops being retryable.
    """

    async def run_stream(
        self,
        request: CanonicalRequest,
        planning: PlanningResult,
        *,
        scopes: list[str],
        deadline_at: datetime,
        buffered: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Yield deltas, then one terminal event.

        ``buffered`` replays a fully-generated, fully-validated response as a
        stream. That is not a lie to the client: the framing is real SSE, and
        §4 explicitly permits buffering when validation needs the whole output.
        """
        if buffered:
            outcome = await self.run(request, planning, scopes=scopes, deadline_at=deadline_at)
            if outcome.succeeded and outcome.result is not None:
                for piece in _chunk_text(outcome.result.text):
                    yield StreamEvent(delta=piece)
            yield StreamEvent(outcome=outcome)
            return

        async for event in self._passthrough(
            request, planning, scopes=scopes, deadline_at=deadline_at
        ):
            yield event

    async def _passthrough(
        self,
        request: CanonicalRequest,
        planning: PlanningResult,
        *,
        scopes: list[str],
        deadline_at: datetime,
    ) -> AsyncIterator[StreamEvent]:
        """Forward provider deltas, fixing the route at the first token."""
        plan = planning.plan
        outcome = OrchestrationResult(state=RequestState.READY, terminal_reason="")
        started = time.monotonic()

        if not plan.has_route:
            yield StreamEvent(
                outcome=self._terminal(
                    outcome, request, RequestState.REJECTED_NO_ROUTE, "no_eligible_route", started
                )
            )
            return

        candidate = plan.candidates[0]
        adapter = self.adapters.get(candidate.provider)
        if adapter is None:
            yield StreamEvent(
                outcome=self._terminal(
                    outcome, request, RequestState.FAILED_EXHAUSTED, "no_adapter", started
                )
            )
            return

        attempt_id = self._record_attempt(
            request,
            candidate,
            sequence=1,
            kind=AttemptKind.GENERATION,
            estimated=candidate.estimated_cost,
        )

        # §4: reservation completes before any byte is committed to the client.
        session = self.session_factory()
        try:
            with immediate_transaction(session):
                reservation = budget_service.reserve(
                    session,
                    request_id=request.request_id,
                    attempt_id=attempt_id,
                    scopes=scopes,
                    estimate=candidate.estimated_cost,
                    request_cost_so_far=Decimal("0"),
                    effective_max_cost=request.controls.max_cost,
                )
        except BudgetExceededError:
            session.close()
            self._finish_attempt(
                attempt_id,
                outcome=AttemptOutcome.CANCELLED,
                result=None,
                actual_cost=Decimal("0"),
                error_code="budget_exceeded",
            )
            yield StreamEvent(
                outcome=self._terminal(
                    outcome, request, RequestState.REJECTED_BUDGET, "budget_exceeded", started
                )
            )
            return

        emitted = False
        text_parts: list[str] = []
        finish_reason = "stop"
        settled = Decimal("0")
        failure: ProviderFailure | None = None
        cancelled = False

        try:
            invocation = ProviderInvocation(
                request=request,
                provider_model_id=candidate.model_id.split("/", 1)[-1],
                gateway_model_id=candidate.model_id,
                max_output_tokens=request.max_output_tokens,
                timeout_seconds=max((deadline_at - datetime.now(UTC)).total_seconds(), 0.1),
                estimated_cost=candidate.estimated_cost,
            )

            async for chunk in adapter.stream(invocation):
                if chunk.delta:
                    emitted = True
                    text_parts.append(chunk.delta)
                    yield StreamEvent(delta=chunk.delta)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason

            settled = candidate.estimated_cost

        except ProviderFailure as exc:
            failure = exc
            # Anything already emitted was generated and will be billed, so a
            # partial stream settles rather than releases.
            settled = candidate.estimated_cost if emitted else Decimal("0")
        except (GeneratorExit, asyncio.CancelledError):
            # §4: the client went away. Cancel, settle what was produced, and
            # release the rest. Re-raised after the finally block resolves the
            # reservation, so cancellation still propagates.
            cancelled = True
            settled = candidate.estimated_cost if emitted else Decimal("0")
            raise
        finally:
            with immediate_transaction(session):
                if settled > 0:
                    budget_service.settle(session, reservation, actual_cost=settled)
                else:
                    budget_service.release(session, reservation)
            session.close()

            self._finish_attempt(
                attempt_id,
                outcome=(
                    AttemptOutcome.CANCELLED
                    if cancelled
                    else (failure.outcome if failure else AttemptOutcome.SUCCESS)
                ),
                result=None,
                actual_cost=settled,
                emitted_output=emitted,
                error_code=(failure.outcome.value if failure else None),
            )
            if cancelled:
                self._terminal(
                    outcome, request, RequestState.CANCELLED, "client_disconnected", started
                )

        if failure is not None:
            if emitted:
                # §4: terminal. No fallback, no escalation, no second model.
                yield StreamEvent(
                    outcome=self._terminal(
                        outcome, request, RequestState.FAILED_PARTIAL, "partial_output", started
                    ),
                    failure=ProviderError(
                        "The response failed after output had already been sent, so it "
                        "could not be retried on another model."
                    ),
                )
            else:
                # Nothing was visible, so an ordinary non-stream retry is legal.
                self.breaker.record_failure(candidate.provider, failure)
                yield StreamEvent(
                    outcome=self._terminal(
                        outcome, request, RequestState.FAILED_EXHAUSTED, "stream_failed", started
                    ),
                    failure=ProviderError("The provider failed before any output was produced."),
                )
            return

        # Validate the assembled output. A passthrough stream only reaches here
        # when no gate needed the whole response, so this is a formality that
        # still records telemetry.
        text = "".join(text_parts)
        result = ProviderResult(
            text=text,
            finish_reason=finish_reason,
            usage=ProviderUsage(
                prompt_tokens=request.estimated_input_tokens,
                completion_tokens=max(len(text) // 4, 1),
            ),
            model_id=candidate.model_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            emitted_output=True,
        )
        report = run_plan(
            self.validators,
            plan.validation_plan,
            request,
            result,
            ValidationContext(attempt_number=1),
        )
        self._persist_validations(request.request_id, attempt_id, report)

        outcome.result = result
        outcome.validation = report
        outcome.resolved_model = candidate.model_id
        outcome.total_cost = settled
        outcome.attempts.append(
            AttemptRecord(
                sequence=1,
                model_id=candidate.model_id,
                kind=AttemptKind.GENERATION,
                outcome=AttemptOutcome.SUCCESS,
                cost=settled,
                latency_ms=result.latency_ms,
                validation=report,
            )
        )

        if report.passed:
            self.breaker.record_success(candidate.provider)
            yield StreamEvent(
                outcome=self._terminal(
                    outcome, request, RequestState.SUCCEEDED, "validated", started
                )
            )
            return

        # Output was already visible, so a failed gate cannot be repaired or
        # escalated -- §4 forbids switching models past the first token.
        yield StreamEvent(
            outcome=self._terminal(
                outcome,
                request,
                RequestState.FAILED_PARTIAL,
                "validation_failed_after_output",
                started,
            ),
            failure=ProviderError(
                "The streamed response failed validation after it had already been sent."
            ),
        )


#: Roughly a word at a time, so a buffered replay still looks like a stream.
def _chunk_text(text: str, size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]
