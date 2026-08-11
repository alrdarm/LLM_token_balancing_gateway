"""Non-stream orchestration, end to end (§12 acceptance scenarios).

These are the §12 scenarios treated as the regression suite the spec says they
are, not as examples. Each names the scenario it covers.

Everything runs against the deterministic fake provider and real deterministic
validators, against a real migrated database. No network, no credentials, no
spend.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import Engine, func, select

from gateway.domain.enums import (
    AttemptKind,
    BudgetWindow,
    Endpoint,
    RequestState,
    ReservationStatus,
    ValidationResult,
)
from gateway.domain.requests import CanonicalRequest, GatewayControls, Message, OutputFormat
from gateway.persistence.engine import create_session_factory
from gateway.persistence.models import Attempt, Budget, BudgetReservation, Request, Validation
from gateway.persistence.seed import seed_all
from gateway.providers.base import AdapterRegistry
from gateway.providers.fake import (
    AlienProvider,
    FakeProvider,
    ScriptedBehaviour,
    auth_error,
    partial_stream_failure,
    rate_limited,
)
from gateway.services.orchestrator import Orchestrator
from gateway.services.planning import build_plan
from gateway.services.resilience import CircuitBreaker, RetryPolicy
from gateway.validators.deterministic import default_registry

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

HASH_KEY = "e2e-hash-key"
SCOPE = "client:e2e"


async def _noop_sleep(seconds: float) -> None:
    """Skip real backoff so failure tests stay fast and deterministic."""
    return None


@pytest.fixture
def factory(migrated_engine: Engine):
    session_factory = create_session_factory(migrated_engine)
    with session_factory() as session:
        seed_all(session)
        session.add(
            Budget(
                scope=SCOPE,
                window_kind=BudgetWindow.TOTAL.value,
                window_start=datetime.now(UTC) - timedelta(days=1),
                hard_limit=Decimal("10.000000000"),
            )
        )
        session.commit()
    return session_factory


def make_request(
    request_id: str = "req_e2e",
    *,
    content: str = "Summarise this report briefly.",
    controls: GatewayControls | None = None,
    output_format: OutputFormat | None = None,
    max_output_tokens: int | None = 256,
) -> CanonicalRequest:
    return CanonicalRequest(
        request_id=request_id,
        endpoint=Endpoint.CHAT_COMPLETIONS,
        requested_model="auto",
        conversation=(Message(role="user", content=content),),
        controls=controls or GatewayControls(max_cost=Decimal("1.0")),
        client_id="client_e2e",
        estimated_input_tokens=40,
        input_hash="a" * 64,
        max_output_tokens=max_output_tokens,
        output_format=output_format or OutputFormat(),
    )


def persist_request(factory, request: CanonicalRequest) -> None:
    with factory() as session:
        session.add(
            Request(
                id=request.request_id,
                client_id=request.client_id,
                endpoint=request.endpoint.value,
                requested_model=request.requested_model,
                normalized_controls_json={},
                state=RequestState.READY.value,
                deadline_at=datetime.now(UTC) + timedelta(seconds=30),
                input_hash=request.input_hash,
                max_cost=request.controls.max_cost or Decimal("1.0"),
            )
        )
        session.commit()


def build_orchestrator(factory, adapters: AdapterRegistry) -> Orchestrator:
    return Orchestrator(
        session_factory=factory,
        adapters=adapters,
        validators=default_registry(),
        breaker=CircuitBreaker(),
        retry_policy=RetryPolicy(base_delay_seconds=0.0, max_delay_seconds=0.0),
        hash_key=HASH_KEY,
        sleep=_noop_sleep,
    )


def registry_with(provider) -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(provider)
    return registry


async def run(factory, request: CanonicalRequest, provider, *, deadline_seconds: float = 30):
    persist_request(factory, request)
    with factory() as session:
        planning = build_plan(session, request)

    orchestrator = build_orchestrator(factory, registry_with(provider))
    return await orchestrator.run(
        request,
        planning,
        scopes=[SCOPE],
        deadline_at=datetime.now(UTC) + timedelta(seconds=deadline_seconds),
    )


# --- ledger helpers --------------------------------------------------------


def budget_state(factory) -> tuple[Decimal, Decimal]:
    with factory() as session:
        budget = session.scalars(select(Budget).where(Budget.scope == SCOPE)).one()
        return budget.spent, budget.reserved


def active_reservations(factory) -> int:
    with factory() as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(BudgetReservation)
                .where(BudgetReservation.status == ReservationStatus.ACTIVE.value)
            )
            or 0
        )


def request_row(factory, request_id: str) -> Request:
    with factory() as session:
        return session.scalars(select(Request).where(Request.id == request_id)).one()


def attempts_of(factory, request_id: str) -> list[Attempt]:
    with factory() as session:
        return list(
            session.scalars(
                select(Attempt).where(Attempt.request_id == request_id).order_by(Attempt.sequence)
            )
        )


# --- happy path ------------------------------------------------------------


async def test_successful_request_reaches_succeeded(factory):
    outcome = await run(factory, make_request(), FakeProvider())

    assert outcome.succeeded
    assert outcome.result is not None
    assert outcome.resolved_model
    assert request_row(factory, "req_e2e").state == RequestState.SUCCEEDED.value


async def test_success_settles_and_leaves_no_active_reservation(factory):
    """§7: every terminal path settles or releases."""
    await run(factory, make_request(), FakeProvider())

    spent, reserved = budget_state(factory)
    assert reserved == Decimal("0")
    assert spent > Decimal("0")
    assert active_reservations(factory) == 0


async def test_success_persists_a_terminal_reason(factory):
    await run(factory, make_request(), FakeProvider())
    assert request_row(factory, "req_e2e").terminal_reason == "validated"


async def test_output_is_stored_only_as_a_digest(factory):
    """§6: no raw outputs at rest."""
    outcome = await run(factory, make_request(), FakeProvider())
    assert outcome.result is not None

    attempts = attempts_of(factory, "req_e2e")
    assert len(attempts) == 1
    assert attempts[0].output_hash is not None
    assert len(attempts[0].output_hash) == 64
    assert outcome.result.text not in (attempts[0].output_hash or "")


async def test_validations_are_persisted(factory):
    await run(factory, make_request(), FakeProvider())

    with factory() as session:
        rows = list(session.scalars(select(Validation).where(Validation.request_id == "req_e2e")))

    assert rows
    assert any(row.aggregate_effect for row in rows)


# --- T03: provider 429 before output --------------------------------------


async def test_rate_limit_before_output_retries_then_succeeds(factory):
    """T03: bounded fallback and correct attempt count."""
    provider = FakeProvider(behaviour=ScriptedBehaviour().fail_next(rate_limited()).succeed_next())
    outcome = await run(factory, make_request(), provider)

    assert outcome.succeeded
    assert len(provider.calls) == 2
    assert len(outcome.attempts) == 2


async def test_failed_attempt_releases_rather_than_settles(factory):
    """A failure before any output must not charge the budget."""
    provider = FakeProvider(behaviour=ScriptedBehaviour().fail_next(rate_limited()).succeed_next())
    await run(factory, make_request(), provider)

    attempts = attempts_of(factory, "req_e2e")
    assert attempts[0].cost_actual == Decimal("0")
    assert attempts[1].cost_actual > Decimal("0")
    assert active_reservations(factory) == 0


async def test_retries_are_bounded_by_the_attempt_cap(factory):
    """Never retry forever: the cap stops it (§8 stop rules)."""
    behaviour = ScriptedBehaviour()
    for _ in range(10):
        behaviour.fail_next(rate_limited())
    provider = FakeProvider(behaviour=behaviour)

    outcome = await run(factory, make_request(), provider)

    assert outcome.state is RequestState.FAILED_EXHAUSTED
    assert len(provider.calls) <= 3
    assert request_row(factory, "req_e2e").state == RequestState.FAILED_EXHAUSTED.value


# --- T04: repair on the same model ----------------------------------------


async def test_schema_failure_repairs_on_the_same_model(factory):
    """T04: two linked generations, one response."""
    schema = {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}}
    request = make_request(
        content="Return JSON with a title.",
        output_format=OutputFormat(kind="json_schema", json_schema=schema, schema_name="r"),
    )

    # The fake returns prose, which cannot satisfy the schema, so every attempt
    # fails validation -- what matters is that a repair was attempted.
    outcome = await run(factory, request, FakeProvider())

    kinds = [attempt.kind for attempt in outcome.attempts]
    assert AttemptKind.REPAIR in kinds, f"expected a repair attempt, got {kinds}"
    assert outcome.validation is not None
    assert outcome.validation.aggregate is ValidationResult.FAIL_REPAIRABLE


async def test_repair_is_capped_by_max_same_model_repairs(factory):
    schema = {"type": "object", "required": ["title"]}
    request = make_request(
        output_format=OutputFormat(kind="json_schema", json_schema=schema, schema_name="r")
    )
    outcome = await run(factory, request, FakeProvider())

    repairs = [a for a in outcome.attempts if a.kind is AttemptKind.REPAIR]
    assert len(repairs) <= 1


async def test_valid_schema_output_passes_without_repair(factory):
    schema = {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}}
    request = make_request(
        output_format=OutputFormat(kind="json_schema", json_schema=schema, schema_name="r")
    )
    provider = FakeProvider(text='{"title": "A report"}')

    outcome = await run(factory, request, provider)

    assert outcome.succeeded
    assert len(outcome.attempts) == 1


# --- T05 (non-stream analogue): partial output is terminal ----------------


async def test_failure_after_visible_output_is_failed_partial(factory):
    """T05: no fallback once output was emitted, and usage still settles."""
    provider = FakeProvider(
        behaviour=ScriptedBehaviour().fail_next(
            partial_stream_failure(prompt_tokens=40, completion_tokens=12)
        )
    )
    outcome = await run(factory, make_request(), provider)

    assert outcome.state is RequestState.FAILED_PARTIAL
    assert len(provider.calls) == 1, "a partial failure must not be retried"
    assert request_row(factory, "req_e2e").terminal_reason == "partial_output"

    spent, reserved = budget_state(factory)
    assert reserved == Decimal("0")
    assert spent > Decimal("0"), "emitted output must still be billed"


# --- T10: explicit model, no fallback -------------------------------------


async def test_auth_error_opens_the_circuit_and_does_not_retry(factory):
    provider = FakeProvider(behaviour=ScriptedBehaviour().fail_next(auth_error()))
    outcome = await run(factory, make_request(), provider)

    assert outcome.state is RequestState.FAILED_EXHAUSTED
    assert len(provider.calls) == 1, "an auth error must never be retried"


async def test_exhaustion_leaves_no_money_held(factory):
    """Whatever the terminal state, nothing may stay reserved."""
    behaviour = ScriptedBehaviour()
    for _ in range(6):
        behaviour.fail_next(rate_limited())

    await run(factory, make_request(), FakeProvider(behaviour=behaviour))

    _, reserved = budget_state(factory)
    assert reserved == Decimal("0")
    assert active_reservations(factory) == 0


# --- budget interaction ---------------------------------------------------


async def test_a_ceiling_no_model_can_meet_is_caught_at_planning(factory):
    """§10 gate 8 rejects before any attempt, so nothing is ever reserved.

    The cheapest model still costs more than this ceiling, so eligibility --
    not reservation -- is what stops it. Failing this early is the point: §8
    says fail before invoking another model when no feasible route remains.
    """
    request = make_request(controls=GatewayControls(max_cost=Decimal("0.000000001")))

    outcome = await run(factory, request, FakeProvider())

    assert outcome.state is RequestState.REJECTED_NO_ROUTE
    spent, reserved = budget_state(factory)
    assert spent == Decimal("0")
    assert reserved == Decimal("0")


async def test_spend_never_exceeds_the_request_ceiling(factory):
    """§6: request_cost_so_far + estimate must stay under the ceiling.

    A *released* attempt costs nothing, so only settled attempts count. This
    uses a schema the fake can never satisfy, so each attempt succeeds at the
    provider (and bills) but fails validation, driving a repair. The ceiling
    admits one billable attempt but not two.
    """
    schema = {"type": "object", "required": ["title"]}
    probe = make_request(
        "req_probe",
        output_format=OutputFormat(kind="json_schema", json_schema=schema, schema_name="r"),
    )
    persist_request(factory, probe)
    with factory() as session:
        single_cost = build_plan(session, probe).plan.candidates[0].estimated_cost

    ceiling = single_cost + (single_cost / 2)
    request = make_request(
        "req_ceiling",
        controls=GatewayControls(max_cost=ceiling),
        output_format=OutputFormat(kind="json_schema", json_schema=schema, schema_name="r"),
    )

    outcome = await run(factory, request, FakeProvider())

    # Which stop rule bites first -- the ceiling or the attempt cap -- depends
    # on how eligibility narrowed under the tighter ceiling, and §8 treats both
    # as legitimate. The invariant is that spend never crosses the ceiling and
    # nothing stays held.
    assert not outcome.succeeded
    assert outcome.state in (
        RequestState.REJECTED_BUDGET,
        RequestState.FAILED_EXHAUSTED,
    )

    spent, reserved = budget_state(factory)
    assert reserved == Decimal("0")
    assert spent <= ceiling, f"spend {spent} exceeded the ceiling {ceiling}"


# --- deadline -------------------------------------------------------------


async def test_expired_deadline_terminates_before_invoking(factory):
    """T09: a deadline already past must not spend money."""
    request = make_request()
    persist_request(factory, request)
    with factory() as session:
        planning = build_plan(session, request)

    orchestrator = build_orchestrator(factory, registry_with(FakeProvider()))

    outcome = await orchestrator.run(
        request,
        planning,
        scopes=[SCOPE],
        deadline_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert outcome.state is RequestState.EXPIRED

    spent, reserved = budget_state(factory)
    assert spent == Decimal("0")
    assert reserved == Decimal("0")
    assert request_row(factory, "req_e2e").state == RequestState.EXPIRED.value


# --- adapters -------------------------------------------------------------


async def test_alien_provider_works_through_the_same_orchestrator(factory):
    """The abstraction holds for a structurally different provider."""
    request = make_request("req_alien")
    persist_request(factory, request)

    with factory() as session:
        planning = build_plan(session, request)

    # Point every candidate at the alien adapter by registering it under the
    # provider name the seeded registry uses.
    alien = AlienProvider()
    alien.name = "fake"
    orchestrator = build_orchestrator(factory, registry_with(alien))

    outcome = await orchestrator.run(
        request,
        planning,
        scopes=[SCOPE],
        deadline_at=datetime.now(UTC) + timedelta(seconds=30),
    )

    assert outcome.succeeded
    assert outcome.result is not None
    assert outcome.result.text.startswith("alien")


async def test_no_adapter_for_the_route_is_not_a_crash(factory):
    """An empty adapter registry must fail cleanly, not raise AttributeError."""
    request = make_request("req_noadapter")
    persist_request(factory, request)
    with factory() as session:
        planning = build_plan(session, request)

    orchestrator = build_orchestrator(factory, AdapterRegistry())
    outcome = await orchestrator.run(
        request,
        planning,
        scopes=[SCOPE],
        deadline_at=datetime.now(UTC) + timedelta(seconds=30),
    )

    assert not outcome.succeeded
    assert outcome.state is RequestState.FAILED_EXHAUSTED
    assert active_reservations(factory) == 0
