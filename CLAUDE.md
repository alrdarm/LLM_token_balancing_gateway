# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

**M0 (skeleton) approved and merged.** **M1 (persistence) complete pending approval.** Domain enums, ORM entities for all §6 tables, exact-decimal money storage, repositories, unit of work, Alembic migrations on SQLite + PostgreSQL, seeded registry/policies, and real readiness probes.

Not built yet: API/canonical request (M2), routing (M3), budgets and providers (M4), orchestration (M5), streaming (M6), hardening (M7).

Development is gated milestone by milestone (M0–M7, §13 of the spec). Do not start the next milestone until the previous one is explicitly approved.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

alembic upgrade head         # apply migrations (never run automatically at boot)
python -m gateway            # serve on GATEWAY_HOST:GATEWAY_PORT (127.0.0.1:8000)

pytest -q                    # full suite (SQLite only unless the URL below is set)
pytest tests/unit -q         # one layer
pytest -k request_id         # one topic
pytest tests/x.py::test_y    # one test
ruff check . && ruff format --check .
mypy                         # strict
```

### Running the PostgreSQL half of the matrix

Persistence tests are parametrised over both backends. PostgreSQL **skips** when unconfigured, so a green local run does not by itself prove the matrix passed:

```bash
docker run -d --name gateway-pg -e POSTGRES_USER=gateway -e POSTGRES_PASSWORD=gateway \
  -e POSTGRES_DB=gateway_test -p 55432:5432 postgres:16-alpine

GATEWAY_TEST_POSTGRES_URL=postgresql+psycopg://gateway:gateway@127.0.0.1:55432/gateway_test pytest -q
```

CI runs both and **fails** if the PostgreSQL entries skip, so the gate cannot pass by silent omission.

### macOS + iCloud pitfall: `pip install -e` silently no-ops, repeatedly

`import gateway` fails with `ModuleNotFoundError` even though `pip install -e` reported success and the `.pth` file contains the right path.

Cause: the `__editable__*.pth` file carries the macOS `UF_HIDDEN` flag, and CPython ≥3.12's `site.addpackage` deliberately skips hidden `.pth` files. **This repo lives in iCloud Drive, which re-applies that flag on sync**, so clearing it once is not durable — it comes back.

```bash
stat -f '%N flags=[%Sf]' .venv/lib/python3.12/site-packages/__editable__*.pth   # diagnose
chflags nohidden .venv/lib/python3.12/site-packages/__editable__*.pth           # temporary
PYTHONPATH=src python -m gateway                                                 # reliable
```

`pytest` is immune (its `pythonpath = ["src"]` bypasses the install), which is exactly why it masks the problem — the CI app-boot smoke test is the real guard. Linux/CI is unaffected; `UF_HIDDEN` does not exist there.

The durable fix is to keep the virtualenv outside iCloud (e.g. `python3 -m venv ~/.virtualenvs/llm-gateway`).

## Reading the spec

No `pdftotext` on this machine. The Anaconda interpreter has `pdfplumber`:

```bash
/opt/anaconda3/bin/python3 -c "
import pdfplumber
with pdfplumber.open('v0.1_specifications/llm-token-balancing-gateway-v0.1-implementation-handoff.pdf') as pdf:
    print('\n'.join(p.extract_text() or '' for p in pdf.pages))
"
```

The PDF is the contract. This file is a navigation aid, not a replacement — check the PDF before implementing anything it governs. MUST / MUST NOT / SHOULD / MAY in the spec are normative.

## What is being built

A single-process **async Python gateway** presenting an OpenAI-compatible surface (`POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models`, `POST /route/inspect`, `GET /health/live|ready`) that selects, invokes, validates, repairs, and escalates across heterogeneous providers. SQLite is the v0.1 default store; PostgreSQL must work without domain redesign.

The central framing that drives the whole design: **a request succeeds only when a generation attempt passes its required validation gates. Provider success alone is not gateway success.** Routing, budgeting, and escalation all exist to serve that.

## Non-negotiable invariants

These constrain nearly every design decision; violating one is a stop-and-ask, not a judgment call.

- Privacy, provider policy, capabilities, context fit, and risk quality floors are **hard gates**. Price never overrides them.
- **Reserve estimated monetary cost atomically before every billable invocation**; settle or release on every exit path, including failures and cancellation.
- Never exceed the caller's request cost ceiling. Fail before invoking another model when no feasible route remains.
- Freeze policy version and route plan per request; persist immutable attempt and validation telemetry.
- **Do not store raw prompts or outputs by default** — keyed hashes and redacted operational metadata only.
- Parse and enforce money with `Decimal`, never binary floating point.
- For privacy, risk, and quality the effective value is the **strictest** applicable value; a caller may raise inferred risk, never lower it.
- Streaming: route is fixed before the first event. After any provider output is client-visible, **MUST NOT** switch models — failure past that point is terminal `FAILED_PARTIAL`.
- `/route/inspect` reserves no budget, calls no provider, and creates no attempt rows.

## Architecture

Both generation endpoints normalize to **one immutable `CanonicalRequest`**. Endpoint serializers are pure adapters: routing, budgets, validation, and escalation never operate on framework or provider objects. This is the main structural rule — it is what keeps two API shapes from leaking into the core.

Flow:

```
Client -> API/auth/normalizer -> classifier -> policy + registry/budget/quota/health
       -> route planner -> orchestrator state machine
       -> reserve -> provider -> settle -> validate
       -> success | repair | fallback | escalation
       -> endpoint serializer -> client
Cross-cutting: repositories | audit/metrics/tracing | redaction
```

Core interfaces (§5 of the spec):

```
Router.plan(request, features, snapshot) -> RoutePlan
Router.next_step(plan, history, validation) -> RouteDecision
PolicyEngine.evaluate(request, features, models, budgets, quotas) -> Eligibility
ModelRegistry.snapshot() -> RegistrySnapshot
BudgetManager.reserve/settle/release(...)
ProviderAdapter.generate/stream(...); capabilities()
Validator.validate(request, candidate, context) -> ValidationResult
```

Suggested layout (spec-suggested, not yet created):

```
src/gateway/
  api/{app,auth,errors,chat,responses,inspect,health}.py
  domain/{requests,models,routing,validation,budgets,errors}.py
  services/{normalizer,classifier,policy,router,orchestrator}.py
  providers/{base,litellm_adapter,commandcode_cli}.py
  validators/{base,json_schema,code,sql,llm_judge}.py
  persistence/{models,repositories,unit_of_work,migrations}.py
  telemetry/{logging,metrics,tracing,redaction}.py
tests/{unit,integration,contract,e2e,fixtures}/
```

`ProviderAdapter` is the abstraction boundary. LiteLLM may normalize HTTP transport but **is not the router** — routing decisions stay in `services/router.py`.

## Conventions established in M0

Extend these rather than reinventing them.

- **Request ID** — `RequestIDMiddleware` is the sole owner of the outbound `X-LLM-Request-ID` header and strips duplicates set by inner layers. An inbound `X-Request-ID` is echoed only if it matches `[A-Za-z0-9_.:-]{1,128}`; otherwise it is replaced (CR/LF would forge log lines and split headers). The ID lives on both `scope["state"]` and a contextvar, so log records correlate without threading it through call signatures.
- **Exception handlers run outside user middleware.** Starlette's `ServerErrorMiddleware` wraps everything, so a 500 it emits never passes through `RequestIDMiddleware`. `error_response()` therefore sets the header itself, and `handle_unexpected_error` re-binds the contextvar before logging. Any future outermost-layer response must do the same.
- **Logging is allowlist-based.** `JsonFormatter` serialises only `ALLOWED_EXTRA_FIELDS`; anything else passed as `extra` is dropped, and exceptions are reduced to type and message so tracebacks cannot leak paths or content. Widening the allowlist means checking the new field can never carry prompt text, credentials, or PII.
- **Readiness is a probe registry.** Register with `gateway.api.health.readiness.register(name, probe)`; a raising probe counts as a failure, not a 500. Failure output names the failing probes but never their detail, because health is unauthenticated. M0 registers zero probes and reports `"checks": []` rather than implying verification it has not done.
- **Error envelope** — build every error through `error_response()`. M0 covers only 404 and 500; M2 owns the full status/code taxonomy in §9.

## Conventions established in M1

- **Money never touches float.** Use `Money` (`persistence/types.py`) for every monetary column and `to_money()` at every boundary. It stores `NUMERIC(20,9)` on PostgreSQL and **scaled-integer nanodollars on SQLite**, because SQLAlchemy's `Numeric` round-trips through `float` there and silently loses exactness. `to_money()` rejects `float` outright rather than converting.
- **Timestamps are timezone-aware UTC.** `UTCDateTime` rejects naive datetimes instead of assuming UTC, since SQLite would otherwise return naive values that compare wrongly against aware ones and corrupt deadline arithmetic.
- **Portable column types, not dialect variants.** `JSONDocument` resolves to `JSONB`/`JSON` per backend. Defining it as a type (rather than inline `with_variant`) keeps autogenerated migrations free of dialect imports, so every migration runs on both backends.
- **Avoid SQL reserved words in column names.** `budgets.window_kind` is not `window` because WINDOW is reserved in PostgreSQL — SQLite accepted it and Postgres rejected the check constraint. Raw SQL in `CheckConstraint` is not auto-quoted.
- **Migrations use `connectable.begin()`.** SQLAlchemy 2.0 has no autocommit and SQLite reports non-transactional DDL, so without an explicit transaction the tables are created but the `alembic_version` stamp is rolled back — leaving a schema that claims to be at no revision.
- **Migrations never run at boot.** Pending migrations are a readiness failure (§11); an operator applies them. A booting replica must not mutate a schema its peers are serving.
- **Repositories hold no policy.** Budget mutation is deliberately absent from `BudgetRepository`: it must go through M4's atomic algorithm (§6), and a convenience helper here would invite bypassing it.
- **Readiness probes are real now.** `database`, `migrations`, `registry`, `active_policy`. Register more via `register_persistence_probes`-style contributions rather than touching the API layer.

## Control precedence

Callers send a top-level `gateway` object; `X-LLM-*` headers are the equivalent for clients that cannot extend bodies. Resolution order, strictest first:

```
deployment security policy > authenticated client policy > X-LLM-* headers
  > gateway body > model-alias defaults > system defaults
```

Selectors: `auto`, `auto-cheap`, `auto-fast`, `auto-quality`, `auto-private`, plus explicit gateway model IDs. Explicit IDs still pass governance; fallback off an explicit ID requires `allow_fallback=true`.

## Request lifecycle

```
RECEIVED -> AUTHENTICATING -> NORMALIZING -> CLASSIFYING -> PLANNING
  -> READY -> RESERVING -> INVOKING -> VALIDATING
  -> SUCCEEDED | REPAIR_PLANNING | ESCALATION_PLANNING -> RESERVING | FAILED_EXHAUSTED
Any non-terminal -> CANCELLED when safely observed.
```

Terminal states: `SUCCEEDED`, `REJECTED`, `REJECTED_NO_ROUTE`, `REJECTED_BUDGET`, `FAILED`, `FAILED_PARTIAL`, `FAILED_EXHAUSTED`, `CANCELLED`, `EXPIRED`. Every terminal path must settle or release its reservation and persist a terminal reason.

Validation severity ordering (highest wins when aggregating):

```
FAIL_GROUNDING > FAIL_CAPABILITY > FAIL_QUALITY > FAIL_REPAIRABLE > INDETERMINATE > PASS
```

`FAIL_REPAIRABLE` → repair on the same model once if useful; `FAIL_QUALITY` → escalate; `FAIL_CAPABILITY` → compatible fallback, no repair.

## Budget reservation

The atomic algorithm in §6 is the load-bearing piece. Lock budgets in stable scope order, assert headroom, increment `reserved`, insert reservation rows, commit — then invoke the provider **never holding a DB lock** — then settle in a second transaction. PostgreSQL uses row locks; SQLite uses `BEGIN IMMEDIATE` with a short busy timeout. A reconciler expires orphan reservations and must never blindly re-invoke an ambiguous billable call.

## Build order

Milestones M0–M7: skeleton → persistence → API/canonical → routing → budgets/providers → orchestration → streaming → hardening. Each has an exit gate in §13.

Start with a **deterministic fake provider and fake validators**. Complete `/route/inspect` and non-streaming orchestration against fakes before adding any real provider — this isolates state and budget correctness from provider variability. Real adapters (Gemini HTTP, CommandCode CLI, OpenAI-compatible HTTP, Anthropic HTTP) come one at a time afterward.

Per-milestone definition of done: code, migration, config example, and tests land together; public behavior has executable examples; new states/errors have redacted logs and metrics; no TODO weakens privacy, budget, idempotency, or state invariants.

Test scenarios T01–T10 in §12 are the acceptance backbone — treat them as the regression suite, not examples. Default CI must run without live provider credentials or spend; SQLite always, PostgreSQL on integration.

## Discretion boundary

The spec grants latitude over exact Python libraries, filenames, dependency-injection style, and test factories. It requires stopping for direction before changing public semantics, weakening a hard invariant, storing raw content, or adding externally billed test actions.

Explicitly out of scope for v0.1: ML/contextual-bandit routing, distributed queues, Redis, web UI, price scraping, multi-user invoicing, provider-hosted tools, background Responses mode, commercial SaaS controls. `n != 1` and audio generation are rejected.
