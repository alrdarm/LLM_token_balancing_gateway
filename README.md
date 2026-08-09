# LLM Token-Balancing Gateway

An OpenAI-compatible gateway that selects, invokes, validates, repairs, and escalates across
heterogeneous LLM providers under explicit privacy, quality, latency, and cost constraints.

A request succeeds only when a generation attempt passes its required validation gates —
provider success alone is not gateway success.

The normative contract is
[`v0.1_specifications/llm-token-balancing-gateway-v0.1-implementation-handoff.pdf`](v0.1_specifications/).
`CLAUDE.md` summarises its architecture and invariants.

## Status

**M0 (skeleton) — provisional.** Package, configuration, CI, lint/type/test tooling, the ASGI
app, health endpoints, and request-ID propagation. No routing, persistence, providers, or
generation endpoints yet; those arrive in M1–M7.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional; defaults work as-is
```

## Run

```bash
python -m gateway                      # http://127.0.0.1:8000
curl -i http://127.0.0.1:8000/health/live
curl -s http://127.0.0.1:8000/health/ready
```

Interactive API docs: <http://127.0.0.1:8000/docs>.

## Checks

```bash
pytest -q                    # full suite
pytest tests/unit -q         # one layer
pytest -k request_id         # one topic
pytest path/to/test.py::name # one test
ruff check .                 # lint
ruff format .                # format
mypy                         # type check (strict)
```

CI runs all of the above plus a boot smoke test on every push and pull request, without provider
credentials or spend.

## Endpoints

| Method | Path            | Purpose                                        |
| ------ | --------------- | ---------------------------------------------- |
| GET    | `/health/live`  | Process liveness; consults no dependencies      |
| GET    | `/health/ready` | Runs every registered dependency probe          |

Health endpoints are unauthenticated. Every response carries `X-LLM-Request-ID`; a client may
supply `X-Request-ID` and it is echoed when it matches `[A-Za-z0-9_.:-]{1,128}`, otherwise the
gateway mints `req_<uuid>`.

`/health/ready` currently has **zero** registered probes and reports `"checks": []` — the
subsystems it must eventually gate on (migrations, active policy, model registry, adapters) do
not exist yet. Later milestones register probes with `gateway.api.health.readiness`.
