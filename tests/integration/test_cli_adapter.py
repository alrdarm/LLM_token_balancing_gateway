"""CommandCode CLI adapter, against real subprocesses (§11).

These spawn actual processes because the properties under test -- shell
injection, environment isolation, process-tree termination -- exist only at the
OS boundary. A mocked ``create_subprocess_exec`` would assert that the code
calls itself correctly and prove nothing about the guarantees.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from gateway.domain.enums import AttemptOutcome
from gateway.providers.base import ProviderFailure
from gateway.providers.commandcode_cli import CommandCodeCLIAdapter
from tests.unit.test_provider_adapters import make_invocation

pytestmark = pytest.mark.integration


def write_script(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable Python script and return its path.

    The shebang uses ``env`` rather than ``sys.executable`` because this
    checkout lives under a path containing a space ("Mobile Documents"), and a
    shebang cannot express an interpreter path with spaces. The scripts use
    only the standard library, so any python3 will do.
    """
    path = tmp_path / name
    path.write_text(f"#!/usr/bin/env python3\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


ECHO_SCRIPT = """
import json, sys
payload = json.load(sys.stdin)
text = payload["messages"][-1]["content"]
print(json.dumps({
    "text": f"cli:{text}",
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}))
"""


async def test_successful_invocation(tmp_path: Path):
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "ok.py", ECHO_SCRIPT))
    result = await adapter.generate(make_invocation())

    assert result.text == "cli:hello"
    assert result.usage.prompt_tokens == 7
    assert result.finish_reason == "stop"


async def test_prompt_travels_on_stdin_not_argv(tmp_path: Path):
    """§11: prompt text must never reach argv.

    Arguments are world-readable in the process table, so a prompt there would
    leak confidential content to any local user.
    """
    script = """
import json, sys
payload = json.load(sys.stdin)
print(json.dumps({
    "text": " ".join(sys.argv[1:]),
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}))
"""
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "argv.py", script))
    result = await adapter.generate(make_invocation())

    assert "hello" not in result.text
    assert "Be brief" not in result.text
    assert "test-model" in result.text


async def test_shell_metacharacters_in_a_prompt_are_inert(tmp_path: Path):
    """The injection case: a prompt containing shell syntax must stay data."""
    from dataclasses import replace

    from gateway.domain.requests import Message

    marker = tmp_path / "pwned.txt"
    hostile = f'hi"; touch {marker}; echo "'

    invocation = make_invocation()
    invocation = replace(
        invocation,
        request=replace(invocation.request, conversation=(Message(role="user", content=hostile),)),
    )

    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "inject.py", ECHO_SCRIPT))
    result = await adapter.generate(invocation)

    assert not marker.exists(), "shell metacharacters were executed"
    assert hostile in result.text


async def test_child_environment_is_allowlisted(tmp_path: Path):
    """§11: secrets must not reach a process that might log or echo them."""
    script = """
import json, os, sys
json.load(sys.stdin)
print(json.dumps({
    "text": ",".join(sorted(os.environ)),
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}))
"""
    os.environ["GATEWAY_HASH_KEY"] = "super-secret-deployment-key"
    os.environ["OPENAI_API_KEY"] = "sk-should-not-leak"
    try:
        adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "env.py", script))
        result = await adapter.generate(make_invocation())
    finally:
        os.environ.pop("GATEWAY_HASH_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

    seen = set(result.text.split(","))
    assert "GATEWAY_HASH_KEY" not in seen
    assert "OPENAI_API_KEY" not in seen
    assert "PATH" in seen


async def test_timeout_kills_the_process(tmp_path: Path):
    script = "import sys, time\nsys.stdin.read()\ntime.sleep(30)\n"
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "slow.py", script))

    started = time.monotonic()
    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation(timeout_seconds=0.5))
    elapsed = time.monotonic() - started

    assert excinfo.value.outcome is AttemptOutcome.TIMEOUT
    assert excinfo.value.retryable
    assert elapsed < 10, "adapter waited for the full sleep instead of killing it"


async def test_timeout_kills_grandchildren_too(tmp_path: Path):
    """Killing only the direct child would orphan its children.

    The child spawns a grandchild that would outlive it and keep writing; after
    the process-group kill, nothing should still be running to update the file.
    """
    marker = tmp_path / "grandchild.txt"
    script = f"""
import subprocess, sys, time
sys.stdin.read()
subprocess.Popen([
    {sys.executable!r}, "-c",
    "import time\\nwhile True:\\n    open({str(marker)!r}, 'a').write('x')\\n    time.sleep(0.05)",
])
time.sleep(30)
"""
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "tree.py", script))

    with pytest.raises(ProviderFailure):
        await adapter.generate(make_invocation(timeout_seconds=0.6))

    time.sleep(0.5)
    size_after_kill = marker.stat().st_size if marker.exists() else 0
    time.sleep(0.6)
    size_later = marker.stat().st_size if marker.exists() else 0

    assert size_later == size_after_kill, "a grandchild survived the process-tree kill"


async def test_nonzero_exit_is_a_retryable_provider_error(tmp_path: Path):
    script = (
        "import sys\nsys.stdin.read()\nsys.stderr.write('/srv/secret/path failed')\nsys.exit(3)\n"
    )
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "fail.py", script))

    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.PROVIDER_ERROR
    assert excinfo.value.retryable
    # §9: stderr can carry internal paths, so it must not reach the client.
    assert "/srv/secret/path" not in str(excinfo.value)


async def test_malformed_output_is_invalid_response(tmp_path: Path):
    script = "import sys\nsys.stdin.read()\nprint('not json')\n"
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "junk.py", script))

    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.INVALID_RESPONSE
    assert not excinfo.value.retryable


async def test_runaway_output_is_bounded(tmp_path: Path):
    """An unbounded child could exhaust gateway memory."""
    script = "import sys\nsys.stdin.read()\nsys.stdout.write('x' * 200000)\n"
    adapter = CommandCodeCLIAdapter(
        executable=write_script(tmp_path, "flood.py", script), max_output_bytes=1024
    )

    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.INVALID_RESPONSE


async def test_missing_executable_does_not_retry(tmp_path: Path):
    """A configuration fault will fail identically every time."""
    adapter = CommandCodeCLIAdapter(executable=str(tmp_path / "does-not-exist"))

    with pytest.raises(ProviderFailure) as excinfo:
        await adapter.generate(make_invocation())

    assert excinfo.value.outcome is AttemptOutcome.AUTH_ERROR
    assert not excinfo.value.retryable


async def test_streaming_is_rejected(tmp_path: Path):
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "ok2.py", ECHO_SCRIPT))

    with pytest.raises(ProviderFailure) as excinfo:
        async for _ in adapter.stream(make_invocation()):
            pass

    assert excinfo.value.outcome is AttemptOutcome.CAPABILITY_REJECTED


async def test_stdin_payload_carries_the_conversation(tmp_path: Path):
    adapter = CommandCodeCLIAdapter(executable=write_script(tmp_path, "ok3.py", ECHO_SCRIPT))
    payload = json.loads(adapter._stdin_payload(make_invocation()))

    assert payload["model"] == "test-model"
    assert payload["system"] == "Be brief."
    assert payload["messages"][-1]["content"] == "hello"
