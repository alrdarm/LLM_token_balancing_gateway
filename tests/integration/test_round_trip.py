"""Entity round-trips on every supported backend.

Round-trip means: written through the ORM, read back from a *fresh* session,
and identical. Reading back through the same session would pass on the identity
map alone and prove nothing about storage.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import (
    AttemptKind,
    AttemptOutcome,
    BudgetWindow,
    DataHandlingTier,
    Endpoint,
    IdempotencyState,
    Quality,
    RequestState,
    ReservationStatus,
    ValidationResult,
)
from gateway.persistence.models import (
    Attempt,
    Budget,
    BudgetReservation,
    IdempotencyRecord,
    Model,
    ModelPrice,
    QuotaSnapshot,
    Request,
    RoutingPolicy,
    Validation,
)
from gateway.persistence.unit_of_work import unit_of_work

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _make_request(request_id: str = "req_round_trip", **overrides: object) -> Request:
    defaults: dict[str, object] = {
        "id": request_id,
        "client_id": "client_a",
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
        "requested_model": "auto",
        "normalized_controls_json": {"quality": "standard", "privacy": "confidential"},
        "state": RequestState.RECEIVED.value,
        "deadline_at": NOW + timedelta(seconds=12),
        "input_hash": "a" * 64,
        "estimated_input_tokens": 81,
        "max_cost": Decimal("0.050000000"),
    }
    defaults.update(overrides)
    return Request(**defaults)


def test_model_and_price_round_trip(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        model = Model(
            id="fake/general",
            provider="fake",
            provider_model_id="general",
            display_name="Fake General",
            data_handling_tier=DataHandlingTier.STANDARD.value,
            quality_tier=Quality.STANDARD.value,
            capabilities=["tools", "json_schema"],
            supported_endpoints=[Endpoint.CHAT_COMPLETIONS.value],
            context_window_tokens=128_000,
            max_output_tokens=8_192,
        )
        model.prices.append(
            ModelPrice(
                input_per_1k_tokens=Decimal("0.000300000"),
                output_per_1k_tokens=Decimal("0.000900000"),
                request_fee=Decimal("0.000010000"),
                effective_from=NOW,
            )
        )
        uow.models.add(model)

    with unit_of_work(session_factory) as uow:
        stored = uow.models.get("fake/general")
        assert stored is not None
        assert stored.capabilities == ["tools", "json_schema"]
        assert stored.data_handling_tier == "standard"
        assert stored.context_window_tokens == 128_000

        price = uow.models.price_at("fake/general", NOW)
        assert price is not None
        assert price.input_per_1k_tokens == Decimal("0.000300000")
        assert price.request_fee == Decimal("0.000010000")


def test_money_survives_the_database_exactly(session_factory: sessionmaker[Session]):
    """The invariant this whole storage type exists for.

    A float column would return 0.1 + 0.2 as 0.30000000000000004; these values
    are chosen to expose that.
    """
    amounts = [
        Decimal("0.000000001"),
        Decimal("0.100000000"),
        Decimal("0.200000000"),
        Decimal("0.003120000"),
        Decimal("12345.678901234"),
        Decimal("0"),
    ]

    with unit_of_work(session_factory) as uow:
        for index, amount in enumerate(amounts):
            uow.budgets.add(
                Budget(
                    scope=f"test:{index}",
                    window_kind=BudgetWindow.DAILY.value,
                    window_start=NOW,
                    hard_limit=amount,
                    spent=Decimal("0"),
                    reserved=Decimal("0"),
                )
            )

    with unit_of_work(session_factory) as uow:
        for index, amount in enumerate(amounts):
            stored = uow.budgets.for_scopes([f"test:{index}"], NOW)
            assert len(stored) == 1
            assert stored[0].hard_limit == amount
            assert isinstance(stored[0].hard_limit, Decimal)


def test_headroom_arithmetic_is_exact(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.budgets.add(
            Budget(
                scope="client:a",
                window_kind=BudgetWindow.DAILY.value,
                window_start=NOW,
                hard_limit=Decimal("0.300000000"),
                spent=Decimal("0.100000000"),
                reserved=Decimal("0.200000000"),
            )
        )

    with unit_of_work(session_factory) as uow:
        budget = uow.budgets.for_scopes(["client:a"], NOW)[0]
        assert uow.budgets.headroom(budget) == Decimal("0")


def test_request_attempt_validation_round_trip(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_make_request())
        uow.flush()

        uow.requests.add_attempt(
            Attempt(
                id="att_1",
                request_id="req_round_trip",
                sequence=1,
                model_id="fake/general",
                route_rank=1,
                attempt_kind=AttemptKind.GENERATION.value,
                outcome=AttemptOutcome.SUCCESS.value,
                prompt_tokens=81,
                completion_tokens=42,
                cost_estimated=Decimal("0.004000000"),
                cost_actual=Decimal("0.003120000"),
                emitted_output=False,
                started_at=NOW,
                finished_at=NOW + timedelta(milliseconds=1840),
                latency_ms=1840,
                output_hash="b" * 64,
            )
        )
        uow.flush()

        uow.requests.add_validation(
            Validation(
                request_id="req_round_trip",
                attempt_id="att_1",
                validator_name="schema_check",
                sequence=1,
                required=True,
                result=ValidationResult.PASS.value,
                aggregate_effect=True,
                detail_codes=[],
            )
        )

    with unit_of_work(session_factory) as uow:
        stored = uow.requests.get("req_round_trip")
        assert stored is not None
        assert stored.normalized_controls_json["privacy"] == "confidential"
        assert stored.max_cost == Decimal("0.050000000")
        assert stored.deadline_at == NOW + timedelta(seconds=12)

        attempts = uow.requests.attempts_for("req_round_trip")
        assert len(attempts) == 1
        assert attempts[0].cost_actual == Decimal("0.003120000")
        assert attempts[0].emitted_output is False

        validations = uow.requests.validations_for("req_round_trip")
        assert len(validations) == 1
        assert validations[0].result == "PASS"
        assert validations[0].required is True


def test_timestamps_round_trip_as_utc_aware(session_factory: sessionmaker[Session]):
    """SQLite would otherwise return naive datetimes, breaking deadline maths."""
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_make_request("req_tz"))

    with unit_of_work(session_factory) as uow:
        stored = uow.requests.get("req_tz")
        assert stored is not None
        assert stored.deadline_at.tzinfo is not None
        assert stored.deadline_at.utcoffset() == timedelta(0)
        assert stored.created_at.tzinfo is not None


def test_naive_datetime_is_rejected(session_factory: sessionmaker[Session]):
    """SQLAlchemy wraps the bind-parameter error, so match the message."""
    with pytest.raises(StatementError, match="naive datetime rejected"):
        with unit_of_work(session_factory) as uow:
            uow.requests.add(_make_request("req_naive", deadline_at=datetime(2026, 6, 1, 12)))


def test_reservation_round_trip(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.requests.add(_make_request("req_res"))
        budget = uow.budgets.add(
            Budget(
                scope="global",
                window_kind=BudgetWindow.TOTAL.value,
                window_start=NOW,
                hard_limit=Decimal("10.000000000"),
            )
        )
        uow.flush()

        uow.budgets.add_reservation(
            BudgetReservation(
                request_id="req_res",
                attempt_id="att_res_1",
                budget_id=budget.id,
                reserved_amount=Decimal("0.004000000"),
                status=ReservationStatus.ACTIVE.value,
                expires_at=NOW + timedelta(minutes=5),
            )
        )

    with unit_of_work(session_factory) as uow:
        active = uow.budgets.active_reservations("req_res")
        assert len(active) == 1
        assert active[0].reserved_amount == Decimal("0.004000000")
        assert active[0].settled_amount is None


def test_idempotency_record_round_trip(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.idempotency.add(
            IdempotencyRecord(
                client_id="client_a",
                idempotency_key="key-1",
                input_hash="c" * 64,
                state=IdempotencyState.IN_PROGRESS.value,
                expires_at=NOW + timedelta(hours=24),
            )
        )

    with unit_of_work(session_factory) as uow:
        record = uow.idempotency.find("client_a", "key-1")
        assert record is not None
        assert record.state == "IN_PROGRESS"
        assert record.response_ref is None


def test_expired_idempotency_record_is_not_live(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.idempotency.add(
            IdempotencyRecord(
                client_id="client_a",
                idempotency_key="stale",
                input_hash="d" * 64,
                state=IdempotencyState.COMPLETED.value,
                expires_at=NOW - timedelta(seconds=1),
            )
        )

    with unit_of_work(session_factory) as uow:
        assert uow.idempotency.find("client_a", "stale") is not None
        assert uow.idempotency.find_live("client_a", "stale", NOW) is None


def test_quota_snapshots_are_append_only(session_factory: sessionmaker[Session]):
    """Staleness is a routing signal, so older observations must survive."""
    with unit_of_work(session_factory) as uow:
        for minutes, remaining in ((0, 1000), (5, 500), (10, 100)):
            uow.quotas.record(
                QuotaSnapshot(
                    provider="fake",
                    model_id="fake/general",
                    window_kind=BudgetWindow.HOURLY.value,
                    observed_at=NOW + timedelta(minutes=minutes),
                    limit_value=1000,
                    remaining=remaining,
                )
            )

    with unit_of_work(session_factory) as uow:
        latest = uow.quotas.latest("fake", "fake/general", BudgetWindow.HOURLY.value)
        assert latest is not None
        assert latest.remaining == 100
        assert latest.observed_at == NOW + timedelta(minutes=10)


def test_policy_round_trip_preserves_weights(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        uow.policies.add(
            RoutingPolicy(
                policy_id="sql-high-risk-v3",
                version=3,
                task_class="sql_review",
                active=True,
                quality_floor=Quality.CRITICAL.value,
                weights={"cost": 0.15, "latency": 0.1},
                validation_plan=["sql_parser", "sql_safety"],
            )
        )

    with unit_of_work(session_factory) as uow:
        policy = uow.policies.get_version("sql-high-risk-v3", 3)
        assert policy is not None
        assert policy.validation_plan == ["sql_parser", "sql_safety"]
        assert policy.weights["cost"] == 0.15
        assert uow.policies.active_for("sql_review") is not None
