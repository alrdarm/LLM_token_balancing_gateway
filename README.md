# LLM Token-Balancing Gateway

An OpenAI-compatible gateway that selects, invokes, validates, repairs, and escalates across
heterogeneous LLM providers under explicit privacy, quality, latency, and cost constraints.

A request succeeds only when a generation attempt passes its required validation gates —
provider success alone is not gateway success.

The normative contract is
[`v0.1_specifications/llm-token-balancing-gateway-v0.1-implementation-handoff.pdf`](v0.1_specifications/).
`CLAUDE.md` summarises its architecture and invariants.

## Status

**M0 (skeleton) — merged. M1 (persistence) — provisional.** Package, configuration, CI, the ASGI
app, health endpoints, request-ID propagation, and the full persistence layer: domain values,
ORM entities, exact-decimal money, repositories, Alembic migrations on SQLite and PostgreSQL, and
a seeded registry. No API surface, routing, providers, or generation endpoints yet; those arrive
in M2–M7.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional; defaults work as-is

alembic upgrade head   # create the schema
```

Migrations are never applied automatically at startup: pending migrations are a readiness
failure, so an operator applies them deliberately rather than a booting process mutating a
schema its peers are serving.

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

Persistence tests run against **both** SQLite and PostgreSQL. PostgreSQL skips unless configured,
so a green local run does not by itself prove the matrix passed:

```bash
docker run -d --name gateway-pg -e POSTGRES_USER=gateway -e POSTGRES_PASSWORD=gateway \
  -e POSTGRES_DB=gateway_test -p 55432:5432 postgres:16-alpine

GATEWAY_TEST_POSTGRES_URL=postgresql+psycopg://gateway:gateway@127.0.0.1:55432/gateway_test pytest -q
```

CI runs lint, types, both backends, and a boot smoke test on every push and pull request, without
provider credentials or spend. It fails if the PostgreSQL entries skip.

## Endpoints

| Method | Path            | Purpose                                        |
| ------ | --------------- | ---------------------------------------------- |
| GET    | `/health/live`  | Process liveness; consults no dependencies      |
| GET    | `/health/ready` | Runs every registered dependency probe          |

Health endpoints are unauthenticated. Every response carries `X-LLM-Request-ID`; a client may
supply `X-Request-ID` and it is echoed when it matches `[A-Za-z0-9_.:-]{1,128}`, otherwise the
gateway mints `req_<uuid>`.

`/health/ready` runs four probes as of M1 — `database`, `migrations`, `registry`, and
`active_policy` — and returns 503 when any fails. Probe *names* appear in the failure message;
diagnostic detail stays in the logs, because health is unauthenticated. Provider adapter probes
join in M4.

## Storage notes

Money is stored exactly, never as floating point: `NUMERIC(20,9)` on PostgreSQL and scaled-integer
nanodollars on SQLite, because SQLAlchemy's `Numeric` round-trips through `float` on SQLite.
Prompts and outputs are never stored — only keyed HMAC digests and redacted operational metadata.
`GATEWAY_HASH_KEY` must be set to a deployment-specific secret outside local/dev; the process
refuses to start otherwise.
