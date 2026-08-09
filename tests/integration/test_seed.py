"""Seeding behaviour."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from gateway.domain.enums import PRIVACY_ELIGIBLE_TIERS, DataHandlingTier, Privacy
from gateway.persistence.seed import SEED_EPOCH, seed_all
from gateway.persistence.unit_of_work import unit_of_work

pytestmark = pytest.mark.integration


def test_seed_populates_registry_and_policies(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        counts = seed_all(uow.session)

    assert counts["models"] > 0
    assert counts["policies"] > 0

    with unit_of_work(session_factory) as uow:
        assert len(uow.models.list_enabled()) == counts["models"]
        assert uow.policies.active_for(None) is not None


def test_seed_is_idempotent(session_factory: sessionmaker[Session]):
    """Re-seeding must not duplicate rows or rewrite a published policy."""
    with unit_of_work(session_factory) as uow:
        first = seed_all(uow.session)

    with unit_of_work(session_factory) as uow:
        second = seed_all(uow.session)

    assert second == {"models": 0, "policies": 0}

    with unit_of_work(session_factory) as uow:
        assert len(uow.models.list_enabled()) == first["models"]


def test_seeded_registry_spans_every_data_handling_tier(
    session_factory: sessionmaker[Session],
):
    """T01 in §12 needs a cheap public_only model that confidential excludes."""
    with unit_of_work(session_factory) as uow:
        seed_all(uow.session)

    with unit_of_work(session_factory) as uow:
        tiers = {model.data_handling_tier for model in uow.models.list_enabled()}

    assert tiers == {tier.value for tier in DataHandlingTier}


def test_confidential_excludes_at_least_one_seeded_model(
    session_factory: sessionmaker[Session],
):
    with unit_of_work(session_factory) as uow:
        seed_all(uow.session)

    with unit_of_work(session_factory) as uow:
        models = uow.models.list_enabled()

    eligible = {tier.value for tier in PRIVACY_ELIGIBLE_TIERS[Privacy.CONFIDENTIAL]}
    excluded = [model for model in models if model.data_handling_tier not in eligible]
    assert excluded, "seed provides nothing for a privacy gate to exclude"


def test_every_seeded_model_has_a_price(session_factory: sessionmaker[Session]):
    """A model without a price cannot be cost-estimated, so it cannot be routed."""
    with unit_of_work(session_factory) as uow:
        seed_all(uow.session)

    with unit_of_work(session_factory) as uow:
        for model in uow.models.list_enabled():
            price = uow.models.price_at(model.id, SEED_EPOCH)
            assert price is not None, f"{model.id} has no effective price"
            assert price.input_per_1k_tokens >= Decimal("0")
            assert isinstance(price.output_per_1k_tokens, Decimal)


def test_seeded_prices_are_exact_decimals(session_factory: sessionmaker[Session]):
    with unit_of_work(session_factory) as uow:
        seed_all(uow.session)

    with unit_of_work(session_factory) as uow:
        price = uow.models.price_at("fake/general", SEED_EPOCH)
        assert price is not None
        assert price.input_per_1k_tokens == Decimal("0.000300000")
