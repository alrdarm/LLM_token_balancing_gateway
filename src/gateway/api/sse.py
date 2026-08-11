"""Server-sent event framing for both streaming protocols (§4, §13 M6).

Two protocols, deliberately kept apart:

* **Chat Completions** streams bare ``data:`` frames of ``chat.completion.chunk``
  objects and terminates with a literal ``data: [DONE]``.
* **Responses** streams *named* events (``event: response.output_text.delta``)
  each carrying a typed JSON payload, and terminates with
  ``response.completed``.

An SDK parses one or the other, so emitting the wrong shape breaks clients in
ways that look like gateway bugs. They share only the low-level frame writer.

Errors mid-stream are the awkward case: the HTTP status was committed with the
headers, so a failure after the first token cannot become a 4xx or 5xx. §4
makes that terminal ``FAILED_PARTIAL``, and the client learns about it through
an in-band error event rather than a status code.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from gateway.domain.errors import GatewayError

#: Terminator required by the Chat Completions SSE protocol.
CHAT_DONE = "[DONE]"


def frame(data: str, *, event: str | None = None) -> str:
    """Encode one SSE frame.

    Newlines inside the payload are split across ``data:`` lines, because a raw
    newline would terminate the frame early and corrupt everything after it.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Chat Completions protocol
# ---------------------------------------------------------------------------


class ChatStreamWriter:
    """Emits ``chat.completion.chunk`` frames."""

    def __init__(self, *, model: str, request_id: str) -> None:
        self.id = f"chatcmpl_{uuid4()}"
        self.model = model
        self.request_id = request_id
        self._created = _now()

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return _json(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    def role(self) -> str:
        """The opening frame, which carries the assistant role and no content."""
        return frame(self._chunk({"role": "assistant"}))

    def delta(self, text: str) -> str:
        return frame(self._chunk({"content": text}))

    def finish(self, finish_reason: str = "stop") -> str:
        return frame(self._chunk({}, finish_reason=finish_reason))

    def usage(self, usage: dict[str, int], gateway: dict[str, Any]) -> str:
        """A final chunk carrying usage and the gateway summary.

        Sent as a chunk with no choices, mirroring how ``stream_options``
        usage reporting works, so a client that ignores it is unaffected.
        """
        return frame(
            _json(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self._created,
                    "model": self.model,
                    "choices": [],
                    "usage": usage,
                    "gateway": gateway,
                }
            )
        )

    def error(self, error: GatewayError, *, request_id: str) -> str:
        """An in-band error frame.

        Status was committed with the headers, so this is the only way to tell
        a client the stream failed after it began (§4).
        """
        return frame(
            _json(
                {
                    "error": {
                        "message": error.message,
                        "type": error.error_type,
                        "param": error.param,
                        "code": error.code,
                    },
                    "gateway": {"request_id": request_id, "retryable": error.retryable},
                }
            )
        )

    def done(self) -> str:
        return frame(CHAT_DONE)


# ---------------------------------------------------------------------------
# Responses protocol
# ---------------------------------------------------------------------------


class ResponsesStreamWriter:
    """Emits named ``response.*`` events."""

    def __init__(self, *, model: str, request_id: str) -> None:
        self.id = f"resp_{uuid4()}"
        self.model = model
        self.request_id = request_id
        self._created = _now()
        self._sequence = 0

    def _next(self) -> int:
        self._sequence += 1
        return self._sequence

    def _envelope(self, status: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "response",
            "created_at": self._created,
            "model": self.model,
            "status": status,
        }

    def created(self) -> str:
        return frame(
            _json(
                {
                    "type": "response.created",
                    "sequence_number": self._next(),
                    "response": self._envelope("in_progress"),
                }
            ),
            event="response.created",
        )

    def delta(self, text: str) -> str:
        return frame(
            _json(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next(),
                    "item_id": f"msg_{self.id}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }
            ),
            event="response.output_text.delta",
        )

    def completed(self, text: str, usage: dict[str, int], gateway: dict[str, Any]) -> str:
        envelope = self._envelope("completed")
        envelope["output"] = [
            {
                "id": f"msg_{self.id}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
        envelope["output_text"] = text
        envelope["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        envelope["gateway"] = gateway
        return frame(
            _json(
                {
                    "type": "response.completed",
                    "sequence_number": self._next(),
                    "response": envelope,
                }
            ),
            event="response.completed",
        )

    def error(self, error: GatewayError, *, request_id: str) -> str:
        """An in-band failure event (§4: status is already committed)."""
        return frame(
            _json(
                {
                    "type": "response.failed",
                    "sequence_number": self._next(),
                    "response": {
                        **self._envelope("failed"),
                        "error": {
                            "message": error.message,
                            "type": error.error_type,
                            "param": error.param,
                            "code": error.code,
                        },
                        "gateway": {"request_id": request_id, "retryable": error.retryable},
                    },
                }
            ),
            event="response.failed",
        )


def _now() -> int:
    from datetime import UTC, datetime

    return int(datetime.now(UTC).timestamp())
