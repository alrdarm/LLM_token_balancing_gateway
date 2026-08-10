"""``POST /route/inspect`` (§4).

Runs authentication, normalization, classification, snapshotting, filtering,
scoring, and validation planning -- and **reserves no budget, calls no
provider, and creates no attempt rows**. That restriction is the whole point of
the endpoint, so this module deliberately shares the planning path with the
orchestrator rather than reimplementing it.

Scores and provider IDs are disclosed only with the ``debug`` scope (§4);
without it a caller sees the shape of the decision but not the internals of the
deployment's model estate.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from gateway.api.auth import SCOPE_DEBUG, AuthenticatedClient
from gateway.api.dependencies import (
    get_client,
    get_session,
    get_settings_dep,
    read_json_body,
    resolve_request_controls,
)
from gateway.api.generation import _check_model_exists, _validate_body
from gateway.api.request_id import request_id_of
from gateway.api.schemas import ChatCompletionRequest, ResponsesRequest
from gateway.config import Settings
from gateway.domain.requests import CanonicalRequest
from gateway.domain.routing import RoutePlan
from gateway.services.normalizer import (
    normalize_chat_request,
    normalize_responses_request,
)
from gateway.services.planning import PlanningResult, build_plan

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inspection"])


def _looks_like_responses(body: dict[str, Any]) -> bool:
    """Decide which wire shape was submitted.

    §4 accepts either shape. ``input`` is the Responses discriminator and
    ``messages`` the Chat one; a body carrying neither is reported against
    ``messages``, the more common case.
    """
    return "input" in body and "messages" not in body


def _canonicalize(
    *,
    request: Request,
    session: Session,
    settings: Settings,
    client: AuthenticatedClient,
    body: dict[str, Any],
) -> CanonicalRequest:
    """Normalize either accepted shape into the canonical request."""
    request_id = f"inspect_{request_id_of(request).removeprefix('req_')}"

    if _looks_like_responses(body):
        responses_payload = _validate_body(ResponsesRequest, body)
        # Checked here too, so inspect and execute agree on an unknown model
        # rather than inspect reporting a route the request could never take.
        _check_model_exists(session, responses_payload.model)
        return normalize_responses_request(
            responses_payload,
            raw_body=body,
            request_id=request_id,
            client_id=client.client_id,
            controls=resolve_request_controls(
                settings=settings,
                client=client,
                headers=dict(request.headers),
                body_controls=responses_payload.gateway,
                requested_model=responses_payload.model,
            ),
            hash_key=settings.hash_key,
        )

    chat_payload = _validate_body(ChatCompletionRequest, body)
    _check_model_exists(session, chat_payload.model)
    return normalize_chat_request(
        chat_payload,
        raw_body=body,
        request_id=request_id,
        client_id=client.client_id,
        controls=resolve_request_controls(
            settings=settings,
            client=client,
            headers=dict(request.headers),
            body_controls=chat_payload.gateway,
            requested_model=chat_payload.model,
        ),
        hash_key=settings.hash_key,
    )


def _serialize_candidate(candidate: Any, *, disclose: bool) -> dict[str, Any]:
    """Render one ranked candidate.

    Without the debug scope the model and provider identity are withheld: they
    describe the deployment's estate, not the caller's request.
    """
    entry: dict[str, Any] = {
        "rank": candidate.rank,
        "eligible": True,
        "estimated_cost": str(candidate.estimated_cost),
        "effective_cost": str(candidate.effective_cost),
        "predicted_latency_ms": candidate.predicted_latency_ms,
        "predicted_pass_probability": round(candidate.predicted_pass_probability, 4),
    }
    if disclose:
        entry["model"] = candidate.model_id
        entry["provider"] = candidate.provider
        entry["score"] = round(candidate.score, 6)
        entry["score_components"] = dict(candidate.score_components)
    return entry


def _serialize(
    result: PlanningResult,
    canonical: CanonicalRequest,
    *,
    request_id: str,
    disclose: bool,
) -> dict[str, Any]:
    """Build the ``gateway.route_inspection`` document (§4)."""
    plan: RoutePlan = result.plan
    features = result.features
    controls = canonical.controls

    warnings: list[str] = []
    if not plan.has_route:
        warnings.append("no_eligible_route")
    if features.used_fallback:
        warnings.append("classifier_fallback")
    if features.confidence < 0.5:
        warnings.append("low_classification_confidence")

    document: dict[str, Any] = {
        "object": "gateway.route_inspection",
        "request_id": request_id,
        "classification": {
            "task_class": features.task_class.value,
            "complexity": features.complexity,
            "risk": features.risk.value,
            "privacy": features.privacy.value,
            "confidence": round(features.confidence, 4),
        },
        "effective_controls": {
            "quality": result.quality_floor.value,
            "privacy": controls.privacy.value,
            "max_cost": str(controls.max_cost) if controls.max_cost is not None else None,
            "max_latency_ms": controls.max_latency_ms,
            "max_attempts": plan.max_generation_attempts,
            "allow_fallback": controls.allow_fallback,
            "validation": controls.validation,
        },
        "policy": {"id": plan.policy_id, "version": plan.policy_version},
        "candidates": [
            _serialize_candidate(candidate, disclose=disclose) for candidate in plan.candidates
        ],
        "excluded": [
            {
                **({"model": excluded.model_id} if disclose else {}),
                "reasons": list(excluded.reasons),
            }
            for excluded in plan.excluded
        ],
        "planned_validation": list(plan.validation_plan),
        "estimated_route_upper_bound": str(plan.estimated_route_upper_bound),
        "warnings": warnings,
    }

    if disclose:
        document["registry_snapshot_at"] = plan.registry_snapshot_at.isoformat()
        document["rationale_codes"] = list(plan.rationale_codes)

    return document


@router.post("/route/inspect", summary="Classify, filter, rank, and explain")
async def inspect_route(
    request: Request,
    body: dict[str, Any] = Depends(read_json_body),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    client: AuthenticatedClient = Depends(get_client),
) -> dict[str, Any]:
    """Explain how a request would be routed, without acting on it."""
    canonical = _canonicalize(
        request=request,
        session=session,
        settings=settings,
        client=client,
        body=body,
    )

    result = build_plan(session, canonical)

    logger.info(
        "Route inspected",
        extra={"event": "route_inspected"},
    )

    return _serialize(
        result,
        canonical,
        request_id=canonical.request_id,
        disclose=client.has_scope(SCOPE_DEBUG),
    )
