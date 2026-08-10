"""Gateway error taxonomy (§9).

Errors are domain objects, not HTTP concerns: the orchestrator raises them and
the API layer renders them. Each carries the status, ``type``, and ``code`` the
spec assigns, so a caller sees the same envelope regardless of which layer
failed.

Messages must never contain secrets, raw provider payloads, internal paths,
SQL, or prompt text. ``param`` names the offending field only.
"""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for every client-visible gateway failure."""

    status_code: int = 500
    error_type: str = "gateway_internal_error"
    code: str = "internal_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.retry_after = retry_after


# --- 400 -------------------------------------------------------------------


class InvalidRequestError(GatewayError):
    """Malformed or semantically unsupported request."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class InvalidGatewayControlError(GatewayError):
    """A ``gateway`` control or ``X-LLM-*`` header is invalid."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_gateway_control"


class ValidationRequiresBufferingError(GatewayError):
    """Streaming was requested but the required validation needs full output.

    §4: high-risk or full-output validation must buffer or reject streaming.
    """

    status_code = 400
    error_type = "invalid_request_error"
    code = "validation_requires_buffering"


# --- 401 / 403 -------------------------------------------------------------


class InvalidAPIKeyError(GatewayError):
    """Missing or unrecognised credential."""

    status_code = 401
    error_type = "authentication_error"
    code = "invalid_api_key"


class PolicyDeniedError(GatewayError):
    """Authenticated but not permitted."""

    status_code = 403
    error_type = "gateway_policy_error"
    code = "policy_denied"


class ModelNotAllowedError(GatewayError):
    """Authenticated but the requested model is out of scope for this client."""

    status_code = 403
    error_type = "gateway_policy_error"
    code = "model_not_allowed"


# --- 404 -------------------------------------------------------------------


class ModelNotFoundError(GatewayError):
    """Unknown explicit model ID."""

    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class RequestNotFoundError(GatewayError):
    """Unknown request ID."""

    status_code = 404
    error_type = "invalid_request_error"
    code = "request_not_found"


# --- 409 -------------------------------------------------------------------


class IdempotencyConflictError(GatewayError):
    """Same idempotency key, different canonical input (§1)."""

    status_code = 409
    error_type = "invalid_request_error"
    code = "idempotency_conflict"


class RequestInProgressError(GatewayError):
    """The idempotency key is currently owned by an in-flight request."""

    status_code = 409
    error_type = "invalid_request_error"
    code = "request_in_progress"
    retryable = True


# --- 413 -------------------------------------------------------------------


class ContextTooLargeError(GatewayError):
    """Input cannot fit any eligible model's context window."""

    status_code = 413
    error_type = "invalid_request_error"
    code = "context_too_large"


class RequestTooLargeError(GatewayError):
    """Payload exceeds the deployment's size limit."""

    status_code = 413
    error_type = "invalid_request_error"
    code = "request_too_large"


# --- 415 -------------------------------------------------------------------


class UnsupportedMediaTypeError(GatewayError):
    """Content type is not ``application/json``."""

    status_code = 415
    error_type = "invalid_request_error"
    code = "unsupported_media_type"


# --- 422 -------------------------------------------------------------------


class NoEligibleRouteError(GatewayError):
    """Valid request, but no model satisfies its constraints."""

    status_code = 422
    error_type = "gateway_policy_error"
    code = "no_eligible_route"


# --- 429 -------------------------------------------------------------------


class BudgetExceededError(GatewayError):
    """A cost ceiling or scoped budget has no headroom."""

    status_code = 429
    error_type = "gateway_budget_error"
    code = "budget_exceeded"


class QuotaExhaustedError(GatewayError):
    """Provider quota is exhausted."""

    status_code = 429
    error_type = "gateway_capacity_error"
    code = "quota_exhausted"
    retryable = True


class RateLimitedError(GatewayError):
    """The caller is being rate limited."""

    status_code = 429
    error_type = "gateway_capacity_error"
    code = "rate_limited"
    retryable = True


# --- 500 -------------------------------------------------------------------


class InternalError(GatewayError):
    """Unexpected gateway fault."""

    status_code = 500
    error_type = "gateway_internal_error"
    code = "internal_error"


class PersistenceError(GatewayError):
    """The gateway could not read or write its own state."""

    status_code = 500
    error_type = "gateway_internal_error"
    code = "persistence_error"


# --- 502 -------------------------------------------------------------------


class ProviderError(GatewayError):
    """Upstream provider failed after the gateway exhausted its options."""

    status_code = 502
    error_type = "gateway_provider_error"
    code = "provider_error"


class InvalidProviderResponseError(GatewayError):
    """Provider returned something the adapter could not interpret."""

    status_code = 502
    error_type = "gateway_provider_error"
    code = "invalid_provider_response"


# --- 503 -------------------------------------------------------------------


class NoProviderAvailableError(GatewayError):
    """No adapter is configured, healthy, or closed-circuit for this request."""

    status_code = 503
    error_type = "gateway_availability_error"
    code = "no_provider_available"
    retryable = True


class NotReadyError(GatewayError):
    """A required dependency is not ready."""

    status_code = 503
    error_type = "gateway_availability_error"
    code = "not_ready"
    retryable = True


# --- 504 -------------------------------------------------------------------


class DeadlineExceededError(GatewayError):
    """The end-to-end deadline elapsed."""

    status_code = 504
    error_type = "gateway_timeout_error"
    code = "deadline_exceeded"
