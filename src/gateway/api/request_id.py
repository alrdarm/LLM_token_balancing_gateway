"""Request identity.

The spec requires the gateway to accept a valid inbound ``X-Request-ID`` or
mint ``req_<uuid>``, and to return the resulting ID on every response via
``X-LLM-Request-ID``.

Client-supplied IDs are echoed into logs and response headers, so they are
validated against a conservative charset before being trusted. An ID
containing CR/LF could forge log lines or split response headers; an unbounded
one could bloat every record it appears in. Invalid values are replaced with a
generated ID rather than rejected, because a malformed correlation hint is not
a reason to fail an otherwise valid request.
"""

from __future__ import annotations

import re
from uuid import uuid4

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.telemetry.context import reset_request_id, set_request_id

INBOUND_HEADER = "x-request-id"
OUTBOUND_HEADER = "X-LLM-Request-ID"

MAX_REQUEST_ID_LENGTH = 128
_VALID_REQUEST_ID = re.compile(r"\A[A-Za-z0-9_.:-]+\Z")


def generate_request_id() -> str:
    """Mint a gateway-owned request ID."""
    return f"req_{uuid4()}"


def is_valid_request_id(value: str) -> bool:
    """Whether a client-supplied request ID is safe to echo."""
    return len(value) <= MAX_REQUEST_ID_LENGTH and bool(_VALID_REQUEST_ID.match(value))


def resolve_request_id(inbound: str | None) -> str:
    """Return the request ID to use, preferring a valid client-supplied one."""
    if inbound is not None and is_valid_request_id(inbound):
        return inbound
    return generate_request_id()


class RequestIDMiddleware:
    """Bind a request ID to the context and echo it on every response.

    Implemented against the raw ASGI interface rather than
    ``BaseHTTPMiddleware`` so the context variable is set in the same task that
    runs the endpoint, and so streaming responses (M6) are not buffered.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = resolve_request_id(request.headers.get(INBOUND_HEADER))
        scope["state"] = {**(scope.get("state") or {}), "request_id": request_id}
        token = set_request_id(request_id)

        header_key = OUTBOUND_HEADER.lower().encode("latin-1")
        header_value = request_id.encode("latin-1")

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                # This middleware is the single owner of the header: drop any
                # value an inner layer set, so it is never emitted twice.
                headers: list[tuple[bytes, bytes]] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != header_key
                ]
                headers.append((header_key, header_value))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)


def request_id_of(request: Request) -> str:
    """Return the request ID bound to ``request`` by the middleware."""
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    # Defensive: an endpoint reached without the middleware still needs an ID.
    return generate_request_id()
