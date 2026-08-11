# Operational Runbook

Operating the LLM Token-Balancing Gateway v0.1. The specification PDF is the
contract; this describes running what was built.

## Deploying

The gateway is a single async process. It needs a database and a
deployment-specific hash key, and nothing else.

```bash
export GATEWAY_ENVIRONMENT=prod
export GATEWAY_DATABASE_URL='postgresql+psycopg://user:pw@host:5432/gateway'
export GATEWAY_HASH_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

alembic upgrade head        # apply migrations FIRST, as a separate step
python -m gateway
```

**Migrations never run at startup.** Pending migrations are a readiness
failure, so a booting replica reports itself unready rather than mutating a
schema its peers are already serving. Apply them deliberately, before rolling
the fleet.

**`GATEWAY_HASH_KEY` is not optional outside local/dev.** The process refuses
to start on the development default. The key is what stops stored digests from
being brute-forced back to prompt text — rotating it makes existing digests
unmatchable, which breaks idempotency replay but leaks nothing.

## Health and readiness

| Endpoint | Meaning | Use for |
| --- | --- | --- |
| `GET /health/live` | Process is running. Consults nothing. | Liveness probe |
| `GET /health/ready` | Every dependency probe passes. | Readiness probe, load-balancer gate |

Readiness fails (503) when: migrations are pending, the model registry is
empty, or no routing policy is active. The response names *which* probes
failed but never their detail, because health is unauthenticated — detail is
in the logs.

**Never point a liveness probe at `/health/ready`.** Liveness deliberately
ignores dependencies; wiring it to readiness means a database blip gets the
process killed instead of drained.

## Reading the logs

Single-line JSON. Every record carries `request_id`, which correlates a
request across its attempts and validations.

Logs are allowlist-based *and* redacted: unknown fields are dropped, and
values that look like credentials are replaced with `[REDACTED]` whatever they
are called. Exception messages are redacted too, because they routinely quote
the value that was rejected.

**If you see prompt text in a log, treat it as a security incident**, not a
formatting bug. It means something bypassed both the allowlist and the
redaction pass.

## Common situations

### Requests failing with 503 `no_provider_available`

No adapter is registered for any eligible model. Check that the deployment
registers the adapters it expects at startup. v0.1 ships the deterministic
fake; real adapters are wired per deployment.

### Requests failing with 422 `no_eligible_route`

The request is valid but nothing can serve it. Use `POST /route/inspect` with
a `debug`-scoped key — it returns *every* reason each model was excluded, and
reserves no budget and calls no provider while doing so.

Common causes, in order of likelihood: a confidential request with only
`public_only` models enabled; a cost ceiling below the cheapest candidate; a
required capability no enabled model has.

### Requests failing with 429 `budget_exceeded`

A scoped budget or the caller's own ceiling has no headroom. Check
`budgets.hard_limit - spent - reserved`. If `reserved` is large but nothing is
in flight, reservations have leaked — see below.

### Reservations appear stuck

Run the reconciler. It resolves reservations past their expiry:

* An orphan whose attempt never started is **released** — nothing could have
  been billed.
* An orphan whose attempt started but has no outcome is **settled at its
  estimate and marked `EXPIRED`**. Whether the provider was actually called is
  unknowable, and §11 forbids re-invoking to find out. Over-counting spend
  merely denies a request; under-counting lets the next one overspend a budget
  that is really depleted.

```python
from gateway.services.reconciler import reconcile_reservations, expire_stale_requests

reconcile_reservations(session_factory)
expire_stale_requests(session_factory)
```

**The reconciler never calls a provider.** It imports none.

### Streaming rejected with 400 `validation_requires_buffering`

The request's task class plans a judge validator, which needs the complete
response. Streaming would hold the caller through a full generation plus a
judge round-trip while pretending to stream, so the gateway refuses instead.
Retry with `stream=false`.

### A provider is failing repeatedly

Its circuit opens and routing excludes it. Auth and configuration errors open
the circuit **immediately** rather than after a threshold, because every retry
would fail identically. Circuit state is in-process and does not coordinate
across replicas.

## Rotating credentials

API keys are stored as keyed digests; the plaintext cannot be recovered.

1. Issue a new key with `gateway.api.auth.create_api_key`.
2. Migrate clients.
3. Set `enabled = false` on the old row, or set `expires_at`.

Do not edit a key in place — issuing and expiring keeps the audit trail intact.
All authentication failures return an identical 401 so keys cannot be
enumerated by comparing responses.

## Retention

| Data | Retention | Notes |
| --- | --- | --- |
| Telemetry (requests, attempts, validations) | 90 days | Operational metadata only |
| Idempotency response references | 24 hours | Purge with `purge_expired_idempotency` |
| Raw prompts and outputs | **Never stored** | Keyed HMAC digests only |

## What this deployment does *not* do

Stated plainly so nobody discovers it during an incident:

* **Only two of four adapter paths exist.** Gemini and Anthropic are unbuilt.
* **The LLM judges are stand-ins** — `independent_review` and `rubric_judge`
  pass any non-empty output. Escalation ladders run, but the quality gate at
  the top of them is not real.
* **Token estimation is a character-count heuristic**, so reservations are
  approximate. It over-estimates by design; the excess is released.
* **Circuit breaker state is per process.** Multiple replicas do not share it.
* **`last_used_at` on API keys is not populated** — writing it per request
  made every call contend on one row.
