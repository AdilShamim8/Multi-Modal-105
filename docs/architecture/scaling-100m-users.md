# Scaling to 100–200 Million Active Users

> Status: current · Last audited: 2026-10-18 · Companion: [capacity-planning.md](capacity-planning.md)

This document answers the question this platform was reorganized around:
**what breaks first at 100–200M monthly active users, and what does the
architecture do about it?** Numbers are derived explicitly so they can be
argued with, not asserted.

## 1. Traffic model (assumptions, stated)

| Parameter | Value | Rationale |
|---|---|---|
| MAU | 200M | target ceiling |
| Daily active fraction | 25% | 50M DAU |
| Generations per active user per day | 0.4 | most users browse, a minority generate heavily |
| Mean daily volume | 20M jobs/day | 50M × 0.4 |
| Average sustained rate | **231 jobs/s** | 20M / 86,400 |
| Peak-to-mean ratio | 4× | evening peaks across time zones |
| **Peak rate** | **926 jobs/s** | the number everything below is sized for |
| T2I share | 92% | video is 10× cost, used less |

## 2. What breaks first — in order

### 2.1 GPU fleet (breaks first — by design)

One L4 GPU sustains ≈2.5 image jobs/s (512px, 30 steps, DPM++, batch 8 —
measured baseline; see capacity doc for the benchmark method).

```
Peak need:      926 jobs/s ÷ 2.5 jobs/s/GPU ≈ 370 GPUs
Plus headroom:  × 1.3 (batching inefficiency, mixed geometry) ≈ 480 GPUs
Warm floor:     24 GPUs (bounds cold-start latency to < 2 min)
```

The architecture's answer:
* **KEDA scales workers on queue depth** (`14-worker-keda.yaml`), target
  backlog 8 jobs/worker, max 60 pods × 3 zones × node capacity — bounded by
  `gpu_max_nodes` as the cost circuit-breaker.
* **Continuous batching** (`serving/batching.py`) coalesces same-signature
  jobs, lifting effective throughput per GPU ~1.6–2.2× before adding hardware.
* **Admission control** (`MM_QUEUE_MAX_DEPTH=10,000`) sheds load with
  `503 + Retry-After` instead of letting queue wait times grow unboundedly —
  a bounded rejection beats an unbounded promise.
* **Step degradation** under OOM (engine `_degrade`) trades quality for
  availability only within a single job, never silently across the fleet.

### 2.2 Redis (single-point risk → replication)

The queue and job hashes are the only stateful component in the hot path.

* At 926 jobs/s, LPUSH/BRPOPLPUSH round-trips are ~2 ms — Redis handles
  >100k ops/s on one node; throughput is not the risk.
* **Availability is the risk**: Redis down = no job admission, no status
  reads. Mitigations in the deployment: AOF persistence, replica +
  sentinel failover, and the API degrades to `/healthz`-alive +
  `/readyz`-failing so the LB drains it gracefully rather than 500-ing.
* Job records are **reconstructible**: artifacts live in S3 with
  checksummed keys; the hash only accelerates lookups. TTL (24h) bounds
  the dataset to < 5 GB even at peak.

### 2.3 API tier (scales horizontally — non-event)

At 926 jobs/s, each submission is ~5 ms of work (validate, moderate,
enqueue). 8 replicas at 4 uvicorn workers each absorb >10k req/s — the
API tier is 10× overprovisioned at peak. CPU-based HPA (max 40) exists
for status-polling storms (10–20 GETs per job lifecycle), which are
cache-friendly and cheap.

### 2.4 Artifact egress (solved by not serving it)

20M jobs/day × ~300 KB PNG ≈ **69 GB/day** — trivially served, but NOT
through the API. Presigned S3 URLs put media traffic on the CDN; the API
only ever hands out pointers. This is the single highest-leverage
bandwidth decision in the system: it converts media egress from an
API-capacity problem into a cache-hit-ratio problem (typical 85–95% on
presigned content after first fetch).

### 2.5 Model registry distribution (solved at deploy time)

A 2 GB DiT bundle × 480 GPUs = 960 GB pulled per cold start wave. The
registry volume is a read-only PVC populated by the preload job
(`scripts/preload_models.py`), and node images cache the base layers —
cold-start pull traffic never crosses the WAN twice.

## 3. Latency budget at peak (p95 targets)

| Stage | Budget | Mechanism |
|---|---|---|
| Edge + auth + rate limit | 50 ms | LB, in-process buckets |
| Moderation | 10 ms | blocklist regex; classifier async with timeout |
| Queue admission | 20 ms | single LPUSH |
| Queue wait | **dominant term** | KEDA 15s poll + warm floor; target p95 < 45 s |
| Generation (512px/30 steps) | 3–6 s | L4 baseline |
| Artifact write + presign | 150 ms | S3 multipart, connection-pooled client |
| **End-to-end p95** | **< 60 s** | SLO (see operations/slos.md) |

Queue wait dominates by design: **the system is honest about being
throughput-bound**. The `sync=true` path exists for the interactive
minority (small steps, ≤30 s budget) and falls back to the queue rather
than lying.

## 4. Multi-region posture

Active-active for the API tier (stateless, GeoDNS); leader-per-region for
Redis with async cross-region replication (jobs are regional by submission
origin); S3 cross-region replication with the artifact key as the stable
identity — a job's media is fetchable from whichever region served it.
Failover is a DNS weight change plus queue drain, documented as a
runbook drill (monthly game-day, `operations/incident-response.md`).

## 5. Cost envelope (order of magnitude, on-demand pricing)

480 L4-equivalents × $0.70/h ≈ **$8k/day at absolute peak**, but the peak
is 3–4 h/day; with the warm floor of 24 GPUs and KEDA's 5-min cooldown,
blended cost lands near **$2.5–3.5k/day ≈ $1M/year** for 200M MAU. Spot
tuples for the worker tier (checkpoint-resumable by design) cut this
40–60%. The point of the estimate: at this scale, **batching efficiency
and queue-target tuning are worth more than hardware negotiation** —
both are config knobs here, not re-architectures.

## 6. What this platform deliberately does NOT claim

* Sub-second image generation — that requires distilled few-step models;
  the engine interface accepts them the day they're trained.
* Perfect moderation — this is a layered gate with a journaled audit
  trail, not a guarantee; the incident process owns the residual risk.
* A toy codebase becoming a hyper-scale system by documentation alone —
  the numbers above are the *plan the code is shaped by*: the queue
  patterns, the admission control, the batching, and the autoscaling
  contracts are all real and testable in this repo.
