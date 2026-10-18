# API Reference

> Live OpenAPI: `/docs` (Swagger) · `/redoc` · Generated from the same pydantic
> models that validate requests — spec drift is structurally impossible.
> Last audited: 2026-10-18

## Authentication

All `/v1/*` routes require `X-API-Key: <key>`. Health and metrics stay open.
Rate limit per key: 20-token bucket, 5/s refill; `429` carries `Retry-After`.

## POST /v1/generations

Submit a generation job.

**Request** (`GenerationRequest`):

| Field | Type | Default | Constraints |
|---|---|---|---|
| prompt | string | — | 1–2000 chars, required |
| modality | enum | `image` | `image` \| `video` |
| negative_prompt | string? | null | ≤ 2000 chars |
| width / height | int | 512 | 64–1024, multiple of 8 |
| num_frames | int | 16 | power of two ≥ 4 (video) |
| num_steps | int | 30 | 1–100 |
| guidance_scale | float | 7.5 | 0–30 |
| sampler | enum | `dpm++` | `ddpm` \| `ddim` \| `dpm++` |
| num_outputs | int | 1 | 1–4 |
| seed | int? | null | 0–2³¹−1 |
| sync | bool | false | bounded 30 s, queue fallback |

**Responses**

| Code | Body | Meaning |
|---|---|---|
| 202 | `JobEnvelope` | queued; poll `links.status` |
| 200 | `SyncGenerationOut` | sync path completed in budget |
| 400 | `ErrorOut` | moderation rejected (`content_rejected`) |
| 401 | `ErrorOut` | missing/invalid key |
| 422 | validation | schema violation (fast, before code runs) |
| 429 | `ErrorOut` + `Retry-After` | rate limited |
| 503 | `ErrorOut` | queue at capacity (`queue_full`) |

**Example**

```bash
curl -X POST https://api.example.com/v1/generations \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a golden retriever in snow","num_steps":30}' | jq
# → {"job_id":"17922...","status":"queued","links":{"status":".../v1/generations/17922..."}}
```

## GET /v1/generations/{job_id}

**200** → `JobStatusOut`: status (`queued|running|succeeded|failed|cancelled`),
timestamps, `queue_wait_s`, `duration_s`, `artifacts[]` (presigned URLs),
`error`, `attempts`. **404** `job_not_found` on unknown/expired ids.

## DELETE /v1/generations/{job_id}

Cancel a **queued** job. **200** cancelled · **404** unknown ·
**409** `not_cancellable` (running/terminal).

## GET /v1/queue/stats *(internal)*

Queue depth, processing count, rate-limiter snapshot.

## System endpoints

| Path | Purpose | Auth |
|---|---|---|
| `GET /healthz` | liveness — LB probes | none |
| `GET /readyz` | readiness — queue/store/engine checks; 503 pulls from rotation | none |
| `GET /metrics` | Prometheus exposition | none |

## Error envelope (every failure, one shape)

```json
{
  "code": "queue_full",
  "message": "generation queue at capacity — shedding load",
  "details": {"depth": 10000, "max": 10000},
  "correlation_id": "8f3a..."
}
```

Branch on `code`, never on message text. Include `correlation_id` in every
support request — it traces the request end-to-end through the platform.
