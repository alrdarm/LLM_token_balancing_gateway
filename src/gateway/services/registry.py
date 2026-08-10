"""Model registry snapshots (§5).

``snapshot()`` reads the registry once and freezes it. Routing then works from
that immutable view, so a model disabled mid-request cannot change a decision
half-way through ranking, and the plan records ``registry_snapshot_at`` so the
decision stays reconstructible afterwards.

Models without an effective price are dropped at snapshot time: cost is an
input to every gate downstream, and a model that cannot be costed cannot be
budgeted or ranked.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from gateway.domain.enums import Capability, DataHandlingTier, Quality
from gateway.domain.routing import ModelSnapshot, RegistrySnapshot
from gateway.persistence.repositories import ModelRepository

logger = logging.getLogger(__name__)


def snapshot(session: Session, *, at: datetime | None = None) -> RegistrySnapshot:
    """Freeze the enabled registry as of ``at`` (default: now)."""
    taken_at = at or datetime.now(UTC)
    repository = ModelRepository(session)

    models: list[ModelSnapshot] = []
    for model in repository.list_enabled():
        price = repository.price_at(model.id, taken_at)
        if price is None:
            logger.warning(
                "Model has no effective price and cannot be routed",
                extra={"event": "model_unpriced"},
            )
            continue

        models.append(
            ModelSnapshot(
                model_id=model.id,
                provider=model.provider,
                data_handling_tier=DataHandlingTier(model.data_handling_tier),
                quality_tier=Quality(model.quality_tier),
                capabilities=frozenset(Capability(capability) for capability in model.capabilities),
                supported_endpoints=frozenset(model.supported_endpoints),
                context_window_tokens=model.context_window_tokens,
                max_output_tokens=model.max_output_tokens,
                input_per_1k=price.input_per_1k_tokens,
                output_per_1k=price.output_per_1k_tokens,
                request_fee=price.request_fee,
                latency_prior_ms=model.latency_prior_ms,
                pass_rate_prior=model.pass_rate_prior,
                failure_rate_prior=model.failure_rate_prior,
                priors_source=model.priors_source,
            )
        )

    # Sorted by ID so a snapshot of the same registry is byte-identical, which
    # is what makes ranking reproducible (§4).
    models.sort(key=lambda entry: entry.model_id)
    return RegistrySnapshot(taken_at=taken_at, models=tuple(models))
