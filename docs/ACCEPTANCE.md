# v0.1 Acceptance Report

Scored against §14 of the implementation handoff. Every criterion is marked
against automated tests or a documented operational check, per §14's rule that
release requires demonstration rather than assertion.

**Overall: 4 of 22 criteria are not met.** All four trace to two pieces of
deferred feature work (two provider adapters, and real LLM judges), neither of
which is hardening. They are listed first so they are not buried.

*Evidence: 634 tests, SQLite and PostgreSQL, no live provider credentials or
spend. `ruff`, `ruff format`, `mypy --strict` clean across 65 source files.*

---

## Not met

| # | Criterion (§14) | Status | Gap |
| --- | --- | --- | --- |
| F2 | Four configurable adapter paths: Gemini HTTP, CommandCode CLI, OpenAI-compatible HTTP, Anthropic HTTP | ❌ **2 of 4** | CommandCode CLI and OpenAI-compatible HTTP are built and tested. **Gemini and Anthropic are unbuilt** (DECISION_LOG R4). |
| F3 | At least 10 task classes have policy fixtures, validation requirements, **and escalation ladders** | ⚠️ **Partial** | 14 classes carry fixtures and validation plans, and ladders execute. But `independent_review`/`rubric_judge` are **stand-ins that pass any non-empty output** (R11), so the gate at the top of each ladder is not real. |
| Q1 | Authorized debug exposes resolved model, validation, attempts, latency, and Decimal actual cost | ⚠️ **Partial** | All fields are exposed and tested. **Actual cost is derived from the estimate**, not provider-reported usage, because the only wired adapter is the fake (R5). |
| O3 | Readiness, metrics, circuit breakers, reconciliation, and quota staleness are tested | ⚠️ **Partial** | Readiness, circuit breakers, and reconciliation are tested. Metrics exist and are unit-tested. **Quota staleness is not exercised** — nothing populates `quota_snapshots` yet. |

---

## Functional

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| F1 | Both generation endpoints work with `model=auto`; `/v1/models` and `/route/inspect` behave as specified | ✅ | `tests/e2e/test_http_lifecycle.py`, `tests/contract/test_inspect.py`, real-SDK round trip in `test_sdk_smoke.py` |
| F2 | Four adapter paths | ❌ | See above |
| F3 | ≥10 task classes with fixtures, validation, ladders | ⚠️ | See above |
| F4 | Structured JSON, tools passthrough, idempotency, retry/fallback, repair/escalation, cancellation, streaming | ✅ | `test_orchestration.py`, `test_http_lifecycle.py`, `test_streaming.py` |

## Safety and correctness

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| S1 | Tests prove privacy-ineligible providers are never invoked | ✅ | T01 at the routing gate (`test_eligibility.py`) and through inspection; the gate runs before any reservation |
| S2 | Concurrent tests prove hard budgets cannot be oversubscribed on SQLite **or** PostgreSQL | ✅ | `test_budget_concurrency.py` — real threads, both backends, 10-way contention |
| S3 | No attempt begins when cost, count, deadline, privacy, capability, or quality would be violated | ✅ | `test_eligibility.py` (one test per §10 gate), plus ceiling and deadline tests in `test_orchestration.py` |
| S4 | Every terminal path settles or releases its reservation and persists a terminal reason | ✅ | Enforced structurally: all exits funnel through `Orchestrator._terminal`, reservations resolve in a `finally`. Asserted across success, failure, exhaustion, partial, budget denial, and cancellation |
| S5 | Raw content and credentials absent from DB, logs, traces, errors | ✅ | `test_redaction.py` (31 snapshots), plus an end-to-end test that posts a marker and greps every table |

## Quality and operations

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| Q1 | Successful requests expose request ID; authorized debug exposes route/cost detail | ⚠️ | Exposed and tested; cost is estimate-derived (above) |
| Q2 | Every route freezes policy version and snapshot time; tie-breaking is deterministic | ✅ | `test_determinism.py` — stability over 20 runs, order-independence, tie-break on exact `Decimal` then model ID |
| O1 | Readiness tested | ✅ | `test_readiness_probes.py` — unmigrated, unseeded, downgraded, and healthy |
| O2 | Circuit breakers tested | ✅ | `test_resilience.py` |
| O3 | Metrics, reconciliation, quota staleness tested | ⚠️ | See above |
| O4 | Unit, contract, integration, failure, security, E2E tests run in CI; SQLite always, PostgreSQL on integration | ✅ | CI runs all layers on both backends and **fails if the PostgreSQL entries skip** |
| O5 | Default CI needs no live provider credentials or spend | ✅ | HTTP adapter via `MockTransport`, CLI via generated scripts, everything else against the deterministic fake |

## Required artifacts (§15)

| Artifact | Status |
| --- | --- |
| Source and migrations | ✅ Three migrations, applied and reverted on both backends |
| Seeded registry and policies | ✅ 4 models across all data-handling tiers, 14 task-class policies |
| OpenAPI document | ✅ Served at `/openapi.json`, asserted in contract tests |
| Sample environment without secrets | ✅ `.env.example`, scanned for credential patterns by a test |
| Architecture and state docs | ✅ `CLAUDE.md`, `DECISION_LOG.md` |
| Operational runbook | ✅ `docs/RUNBOOK.md` |
| CI suite | ✅ Three jobs: static, tests on both backends, boot smoke |
| Acceptance report | ✅ This document |

---

## Recommendation

**Do not release as v0.1 without closing F2 and F3.** The gateway's central
claim is that a request succeeds only when it passes its validation gates;
with stand-in judges, high-risk and subjective classes have no real quality
gate, so that claim is weaker than it reads. And §14 names four adapter paths
explicitly.

Everything else is demonstrated. The remaining work is bounded and known:

1. **Gemini HTTP and Anthropic HTTP adapters** (R4). The `ProviderAdapter`
   boundary is already validated against a deliberately alien fake, so these
   should be translation work rather than redesign.
2. **A real LLM judge** (R11). This introduces the first genuinely billable
   code path in the codebase and needs its own budget accounting — §6 already
   links judge cost through `validations.validator_attempt_id`.
3. Optionally **R5** (real tokenizer) and **quota snapshot population**, both
   of which would upgrade the partial criteria above to met.
