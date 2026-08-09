# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

Pre-implementation. The only artifact is `v0.1_specifications/llm-token-balancing-gateway-v0.1-implementation-handoff.pdf` (17 pages, dated 9 Aug 2026, "implementation-ready"). `README.md` is empty; there is no source tree, dependency manifest, lint/type/test tooling, or CI yet. Milestone M0 (below) is what establishes those, so the build/lint/test commands for this repo do not exist and should not be invented — add them here as M0 lands them.

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
