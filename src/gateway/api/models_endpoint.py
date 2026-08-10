"""``GET /v1/models`` (§1).

Returns the selectors plus every enabled explicit model ID. Selectors are
listed as models because that is how a caller passes them to ``model``, and an
SDK's model picker should show them alongside concrete IDs.

Disabled models are omitted: advertising a model the gateway will refuse to
route to would make every eventual failure look like a gateway bug.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from gateway.api.auth import AuthenticatedClient
from gateway.api.dependencies import get_client, get_session
from gateway.api.schemas import ModelCard, ModelList
from gateway.domain.requests import SELECTORS
from gateway.persistence.repositories import ModelRepository

router = APIRouter(tags=["models"])

#: Stable creation timestamp for selectors. They are not versioned artefacts,
#: but the field is required by the OpenAI model shape.
SELECTOR_CREATED_AT = 0


@router.get("/v1/models", summary="Selectors and enabled explicit model IDs")
async def list_models(
    session: Session = Depends(get_session),
    client: AuthenticatedClient = Depends(get_client),
) -> ModelList:
    """List routable selectors and models."""
    cards = [
        ModelCard(id=selector, created=SELECTOR_CREATED_AT, owned_by="gateway")
        for selector in sorted(SELECTORS)
    ]

    for model in ModelRepository(session).list_enabled():
        cards.append(
            ModelCard(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.provider,
            )
        )

    return ModelList(data=cards)
