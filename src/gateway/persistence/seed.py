"""Seed registry and routing policies.

Seeding is idempotent: it inserts what is missing and leaves existing rows
alone. Re-running it must never rewrite a published policy version, because
requests freeze ``policy_id`` and ``policy_version`` and a mutated policy would
retroactively change how a completed request was decided (§10).

The models here belong to the deterministic ``fake`` provider. The build order
in §13 requires a fake provider and fake validators to come first, so that
state and budget correctness can be established before provider variability
enters. Real adapters and their registry entries arrive in M4.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from gateway.domain.enums import (
    Capability,
    DataHandlingTier,
    Endpoint,
    Quality,
    TaskClass,
)
from gateway.persistence.models import Model, ModelPrice, RoutingPolicy
from gateway.persistence.repositories import ModelRepository, PolicyRepository

logger = logging.getLogger(__name__)

#: All seeded prices are effective from this instant, so a seeded database is
#: byte-identical regardless of when it was created.
SEED_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

_ALL_ENDPOINTS = [Endpoint.CHAT_COMPLETIONS.value, Endpoint.RESPONSES.value]


class SeedModel:
    """A registry entry plus its opening price."""

    def __init__(
        self,
        *,
        model_id: str,
        display_name: str,
        tier: DataHandlingTier,
        quality: Quality,
        capabilities: list[Capability],
        context_window_tokens: int,
        max_output_tokens: int,
        input_per_1k: str,
        output_per_1k: str,
        request_fee: str = "0",
    ) -> None:
        self.model_id = model_id
        self.display_name = display_name
        self.tier = tier
        self.quality = quality
        self.capabilities = capabilities
        self.context_window_tokens = context_window_tokens
        self.max_output_tokens = max_output_tokens
        self.input_per_1k = input_per_1k
        self.output_per_1k = output_per_1k
        self.request_fee = request_fee


#: A registry spanning every data-handling tier and quality tier, so privacy
#: and quality gates have something to exclude in tests (T01 in §12 needs a
#: cheap ``public_only`` model that a confidential request must never reach).
SEED_MODELS: tuple[SeedModel, ...] = (
    SeedModel(
        model_id="fake/flash",
        display_name="Fake Flash",
        tier=DataHandlingTier.PUBLIC_ONLY,
        quality=Quality.ECONOMY,
        capabilities=[Capability.STREAMING],
        context_window_tokens=32_000,
        max_output_tokens=4_096,
        input_per_1k="0.000050000",
        output_per_1k="0.000150000",
    ),
    SeedModel(
        model_id="fake/general",
        display_name="Fake General",
        tier=DataHandlingTier.STANDARD,
        quality=Quality.STANDARD,
        capabilities=[Capability.STREAMING, Capability.TOOLS, Capability.JSON_SCHEMA],
        context_window_tokens=128_000,
        max_output_tokens=8_192,
        input_per_1k="0.000300000",
        output_per_1k="0.000900000",
    ),
    SeedModel(
        model_id="fake/code-strong",
        display_name="Fake Code Strong",
        tier=DataHandlingTier.STANDARD,
        quality=Quality.HIGH,
        capabilities=[
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.JSON_SCHEMA,
            Capability.REASONING,
        ],
        context_window_tokens=200_000,
        max_output_tokens=16_384,
        input_per_1k="0.001200000",
        output_per_1k="0.003600000",
    ),
    SeedModel(
        model_id="fake/private-reasoning",
        display_name="Fake Private Reasoning",
        tier=DataHandlingTier.ZDR,
        quality=Quality.CRITICAL,
        capabilities=[
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.JSON_SCHEMA,
            Capability.REASONING,
            Capability.VISION,
        ],
        context_window_tokens=200_000,
        max_output_tokens=32_768,
        input_per_1k="0.003000000",
        output_per_1k="0.015000000",
        request_fee="0.000100000",
    ),
)

#: Conservative v0.1 scoring weights (§10). These encode priors, not measured
#: pass rates; ``priors_source`` records that so later empirical values can
#: replace them without an API change.
DEFAULT_WEIGHTS = {"cost": 0.4, "latency": 0.2, "quality_shortfall": 0.3, "failure_risk": 0.1}


def _seed_models(session: Session) -> int:
    """Insert missing registry entries. Returns how many were added."""
    repo = ModelRepository(session)
    added = 0

    for entry in SEED_MODELS:
        if repo.get(entry.model_id) is not None:
            continue

        model = Model(
            id=entry.model_id,
            provider="fake",
            provider_model_id=entry.model_id.split("/", 1)[1],
            display_name=entry.display_name,
            data_handling_tier=entry.tier.value,
            quality_tier=entry.quality.value,
            capabilities=[capability.value for capability in entry.capabilities],
            supported_endpoints=list(_ALL_ENDPOINTS),
            context_window_tokens=entry.context_window_tokens,
            max_output_tokens=entry.max_output_tokens,
            enabled=True,
        )
        model.prices.append(
            ModelPrice(
                currency="USD",
                input_per_1k_tokens=Decimal(entry.input_per_1k),
                output_per_1k_tokens=Decimal(entry.output_per_1k),
                request_fee=Decimal(entry.request_fee),
                effective_from=SEED_EPOCH,
                source="seed:v0.1",
            )
        )
        repo.add(model)
        added += 1

    return added


def _seed_policies(session: Session) -> int:
    """Insert missing routing policies. Returns how many were added.

    M1 seeds the default policy plus one high-risk class policy, which is what
    the persistence round-trip needs. §14 requires at least ten task classes to
    carry policy fixtures before release; M3 owns filling the matrix out.
    """
    repo = PolicyRepository(session)
    added = 0

    policies = [
        RoutingPolicy(
            policy_id="default-v1",
            version=1,
            task_class=None,
            active=True,
            quality_floor=Quality.STANDARD.value,
            max_generation_attempts=3,
            max_same_model_repairs=1,
            weights=dict(DEFAULT_WEIGHTS),
            validation_plan=["schema_check"],
            priors_source="v0.1-conservative-prior",
        ),
        RoutingPolicy(
            policy_id="sql-high-risk-v1",
            version=1,
            task_class=TaskClass.SQL_REVIEW.value,
            active=True,
            quality_floor=Quality.CRITICAL.value,
            max_generation_attempts=3,
            max_same_model_repairs=1,
            weights={"cost": 0.15, "latency": 0.1, "quality_shortfall": 0.6, "failure_risk": 0.15},
            validation_plan=["sql_parser", "sql_safety", "independent_review"],
            priors_source="v0.1-conservative-prior",
        ),
    ]

    for policy in policies:
        if repo.get_version(policy.policy_id, policy.version) is not None:
            continue
        repo.add(policy)
        added += 1

    return added


def seed_all(session: Session) -> dict[str, int]:
    """Seed registry and policies. Safe to run repeatedly."""
    counts = {"models": _seed_models(session), "policies": _seed_policies(session)}
    session.flush()
    logger.info(
        "Seeded reference data",
        extra={"event": "seed_complete"},
    )
    return counts
