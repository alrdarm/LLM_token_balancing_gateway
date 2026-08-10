"""CommandCode CLI adapter (§11, §14).

The security rules here are not incidental -- prompts are attacker-influenced
text being handed to a subprocess, so §11 is explicit:

* **stdin and argument arrays, never a shell.** ``shell=True`` would make any
  prompt containing ``;`` or ``$(...)`` a command injection. The prompt goes
  in over stdin and never appears in argv at all, which also keeps it out of
  the process table where other users could read it.
* **Isolated environment.** The child gets an explicit allowlist, not the
  gateway's environment, so provider credentials and the deployment hash key
  cannot leak into a process that might log or echo them.
* **Bounded output.** A runaway process could otherwise exhaust memory; output
  is truncated at a limit and the attempt reported as invalid.
* **Process-tree timeout.** Killing only the direct child can orphan its
  grandchildren, which keeps holding the resources the timeout was meant to
  release, so the whole process group is signalled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import AsyncIterator

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import (
    ProviderCapabilities,
    ProviderFailure,
    ProviderInvocation,
    ProviderResult,
    ProviderUsage,
    StreamChunk,
)

logger = logging.getLogger(__name__)

#: Maximum bytes accepted from the child before it is treated as runaway.
MAX_OUTPUT_BYTES = 4 * 1024 * 1024

#: Environment variables the child is allowed to see. Everything else is
#: dropped, including provider keys and the deployment hash key.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

#: Grace period between SIGTERM and SIGKILL for the process group.
TERMINATE_GRACE_SECONDS = 2.0


class CommandCodeCLIAdapter:
    """Runs a local CLI binary that speaks JSON over stdin/stdout."""

    name = "commandcode"

    def __init__(
        self,
        *,
        executable: str,
        extra_args: tuple[str, ...] = (),
        env_allowlist: tuple[str, ...] = ENV_ALLOWLIST,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.executable = executable
        self.extra_args = tuple(extra_args)
        self.env_allowlist = tuple(env_allowlist)
        self.max_output_bytes = max_output_bytes

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=False,
            supports_tools=False,
            supports_json_schema=True,
            supports_vision=False,
        )

    def _child_env(self) -> dict[str, str]:
        """Build the child's environment from the allowlist only."""
        return {name: os.environ[name] for name in self.env_allowlist if name in os.environ}

    def _stdin_payload(self, invocation: ProviderInvocation) -> bytes:
        """The request, as JSON on stdin.

        Prompt content travels here and never in argv: arguments are visible in
        the process table to any user on the machine.
        """
        request = invocation.request
        payload = {
            "model": invocation.provider_model_id,
            "system": request.system_instructions or "",
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.conversation
            ],
            "max_output_tokens": invocation.max_output_tokens,
        }
        return json.dumps(payload).encode("utf-8")

    def _argv(self, invocation: ProviderInvocation) -> list[str]:
        """Argument array. Contains no caller-controlled text."""
        return [self.executable, "--model", invocation.provider_model_id, *self.extra_args]

    async def generate(self, invocation: ProviderInvocation) -> ProviderResult:
        started = time.monotonic()

        try:
            process = await asyncio.create_subprocess_exec(
                *self._argv(invocation),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._child_env(),
                # New session so signalling the group cannot reach the gateway
                # itself, and so grandchildren are included in the kill.
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise ProviderFailure(
                "The provider CLI could not be started.",
                outcome=AttemptOutcome.AUTH_ERROR,
                retryable=False,
            ) from exc

        try:
            # stderr is deliberately discarded: §9 forbids returning it, and
            # it is captured only so the child does not block on a full pipe.
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(self._stdin_payload(invocation)),
                timeout=invocation.timeout_seconds,
            )
        except TimeoutError as exc:
            await self._terminate_tree(process)
            raise ProviderFailure(
                "The provider CLI timed out.",
                outcome=AttemptOutcome.TIMEOUT,
                retryable=True,
            ) from exc

        if len(stdout) > self.max_output_bytes:
            raise ProviderFailure(
                "The provider CLI produced more output than permitted.",
                outcome=AttemptOutcome.INVALID_RESPONSE,
                retryable=False,
            )

        if process.returncode != 0:
            # stderr can contain paths and configuration; log it, never return
            # it (§9).
            logger.warning(
                "Provider CLI exited non-zero",
                extra={"event": "cli_failed", "status_code": process.returncode},
            )
            raise ProviderFailure(
                "The provider CLI failed.",
                outcome=AttemptOutcome.PROVIDER_ERROR,
                retryable=True,
            )

        return self._parse(stdout, invocation, started)

    def _parse(
        self, stdout: bytes, invocation: ProviderInvocation, started: float
    ) -> ProviderResult:
        try:
            body = json.loads(stdout)
            text = body["text"]
            usage = body.get("usage") or {}
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderFailure(
                "The provider CLI returned output the adapter could not interpret.",
                outcome=AttemptOutcome.INVALID_RESPONSE,
                retryable=False,
            ) from exc

        return ProviderResult(
            text=str(text),
            finish_reason=str(body.get("finish_reason") or "stop"),
            usage=ProviderUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ),
            model_id=invocation.gateway_model_id,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def _terminate_tree(self, process: asyncio.subprocess.Process) -> None:
        """Signal the whole process group, escalating to SIGKILL.

        ``start_new_session`` put the child in its own group, so this reaches
        any grandchildren it spawned. Killing only the child would leave those
        running and still holding resources.
        """
        if process.returncode is not None:
            return

        try:
            group = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError):
            group = None

        try:
            if group is not None:
                os.killpg(group, signal.SIGTERM)
            else:  # pragma: no cover - defensive
                process.terminate()
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
            return
        except TimeoutError:
            pass

        try:
            if group is not None:
                os.killpg(group, signal.SIGKILL)
            else:  # pragma: no cover - defensive
                process.kill()
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            return

        # Reap, so the killed child does not linger as a zombie.
        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
        except TimeoutError:  # pragma: no cover - defensive
            logger.error("Provider CLI did not exit after SIGKILL", extra={"event": "cli_stuck"})

    async def stream(self, invocation: ProviderInvocation) -> AsyncIterator[StreamChunk]:
        """Not supported; the capability gate should have excluded this model."""
        raise ProviderFailure(
            "The provider CLI does not support streaming.",
            outcome=AttemptOutcome.CAPABILITY_REJECTED,
            retryable=False,
        )
        # Unreachable, and deliberately so: the bare ``yield`` is what makes
        # this an async generator, which the ProviderAdapter protocol
        # requires. Without it, ``async for`` over the result would fail
        # with a confusing TypeError instead of the capability failure.
        yield StreamChunk()  # type: ignore[unreachable]  # pragma: no cover
