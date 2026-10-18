# Serving Guide

> Last audited: 2026-10-18 · Code: `src/multimodal/serving/` · Ops: [runbook](../operations/runbook.md)

## Start small, ship the same shape

```bash
mma serve --port 8000                       # single process, dev
docker compose -f deployments/docker/docker-compose.yml up -d   # full local stack
kubectl apply -f deployments/k8s/           # production topology
```

All three run the SAME application factory (`serving/app.py`); only the
process wrapper differs. What is true locally is true in prod — that is
the point.

## The three knobs operators actually touch

| Knob | Default | Effect |
|---|---|---|
| `MM_QUEUE_MAX_DEPTH` | 10,000 | when to shed (503) instead of queue |
| KEDA `listLength` target | 8/worker | latency ↔ fleet size dial |
| `MM_SYNC_TIMEOUT_S` | 30 | interactive budget before queue fallback |

Everything else is code-review territory; these three are the ones a
weekday on-call legitimately tunes.

## Verifying a deployment

```bash
curl -s localhost:8000/healthz | jq         # liveness: process + version
curl -s localhost:8000/readyz  | jq         # readiness: queue + store + engine
curl -s localhost:8000/metrics  | grep mm_  # metric contract present?
```

Submit a canary job (small steps, sync) — it exercises auth →
moderation → queue → worker → artifact → status end-to-end. If the
canary is green, the platform is green; the e2e test suite runs exactly
this flow.

## Scaling behaviour (what happens without you)

* API: CPU HPA 3→40 replicas; scale-down waits 5 min (in-flight jobs drain).
* Workers: KEDA on queue depth 2→60 pods; cooldown 5 min; fallback floor
  of 2 if Redis metrics are unreachable.
- Workers drain gracefully on SIGTERM: in-flight jobs finish (≤120 s
  grace), queued jobs stay in Redis, replacement pods recover strays.

## What to watch

The Grafana overview dashboard (`deployments/monitoring/`) leads with
the four numbers that matter: throughput by status, queue depth vs.
running, generation p50/p95/p99, and API status-class rates. Alert rules
(`alerts.yml`) encode the thresholds; every alert's `runbook` annotation
is a deep link to the recovery procedure.

## Safety and auth in serving

* Auth: `X-API-Key` (constant-time compare); health/metrics stay open.
* Rate limit: 20-token bucket, 5/s refill per key; `Retry-After` on 429.
* Moderation: blocklist + optional classifier hook, pre-queue — a
  rejected prompt never reaches a GPU.
* Secrets: env-only (`MM_API_KEYS`), fingerprints in logs — never raw keys.
