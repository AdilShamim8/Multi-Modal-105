# API Design

> Status: current · Last audited: 2026-10-18 · Live spec: `/docs` (OpenAPI)

## Principles

1. **Async-first.** Generation is throughput-bound; the default path is
   submit → 202 + poll. Synchronous responses exist only under a hard
   budget with queue fallback — the API never blocks indefinitely.
2. **Stable error envelope.** Every failure — validation, auth, rate
   limit, safety, internal — returns `{"code", "message", "details",
   "correlation_id"}`. Clients branch on `code`, never on message text.
3. **Idempotency where it pays.** Submission is not idempotent (jobs are
   cheap and client-scoped); artifact keys are content-addressed, so
   retries never duplicate media.
4. **Bounded inputs at the schema, not in handlers.** Geometry, steps,
   frames, and prompt length are constrained in
   `serving/schemas.py` — 422 before any code runs.

## Surface

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| POST | `/v1/generations` | Submit image/video job | 202, 200 (sync), 400, 401, 422, 429, 503 |
| GET | `/v1/generations/{id}` | Status + artifact URLs | 200, 401, 404 |
| DELETE | `/v1/generations/{id}` | Cancel queued job | 200, 401, 404, 409 |
| GET | `/v1/queue/stats` | Queue telemetry (internal) | 200, 401 |
| GET | `/healthz` | Liveness (LB) | 200 |
| GET | `/readyz` | Readiness (queue/store/engine) | 200, 503 |
| GET | `/metrics` | Prometheus exposition | 200 |

## Versioning policy

* Breaking changes → new path prefix (`/v2/...`); `/v1` maintained ≥ 2
  minor releases with deprecation headers (`Sunset`).
* Additive changes (new optional fields) land in-place.
* The OpenAPI schema is generated from the same pydantic models that
  validate requests — spec drift is structurally impossible.

## Client integration sketch

```bash
# Submit (async)
JOB=$(curl -s -X POST https://api.example.com/v1/generations \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a golden retriever in snow","num_steps":30}' | jq -r .job_id)

# Poll until terminal
until [ "$(curl -s -H "X-API-Key: $KEY" .../v1/generations/$JOB | jq -r .status)" != queued ]; do
  sleep 2
done

# Fetch artifact (presigned, CDN-fronted)
curl -O "$(curl -s -H "X-API-Key: $KEY" .../v1/generations/$JOB | jq -r .artifacts[0].url)"
```

Rate limits: 20-token bucket, 5/s refill per key (headers on 429 include
`Retry-After`). Raise limits via the platform team, not by retry storms —
the edge limiter is stricter than the app limiter by design.
