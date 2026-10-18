# Failure Modes and Mitigations (FMEA)

> Status: current · Last audited: 2026-10-18 · Drill cadence: monthly game-day

Severity: **S1** user-visible outage or safety incident · **S2** degraded
capacity/latency · **S3** operational toil without user impact.

## Failure mode register

| # | Component | Failure | Sev | Detection | Mitigation (in code/infra) | Recovery |
|---|---|---|---|---|---|---|
| 1 | Worker | CUDA OOM mid-job | S2 | worker logs `out_of_memory` | `_execute_with_oom_retry`: halve batch, cut steps, retry ≤2; `OutOfMemoryError` → job requeue | self-healing; alert if rate > 5%/10m (`MmGenerationFailuresSpike`) |
| 2 | Worker | Crash mid-job | S2 | processing-list scan | `recover_stuck()` at worker startup requeues RUNNING jobs older than 10 min | automatic on replacement pod |
| 3 | Queue | Redis unavailable | S1 | `/readyz` false → LB drains | API keeps serving `/healthz` (liveness ≠ readiness); KEDA `fallback` holds worker floor | runbook: restart/ failover; queued jobs survive via AOF |
| 4 | Queue | Backlog runaway | S2 | `MmQueueBacklogGrowing` page | admission control sheds at 10k depth with 503+Retry-After; KEDA scales out | runbook: raise KEDA target vs. shed harder |
| 5 | API | Replica OOM (many prompts) | S3 | restarts metric | request schema caps prompt 2k chars; body-size ingress limit 256k | horizontal HPA |
| 6 | API | Bad deploy | S1 | 5xx rate alert | rolling update `maxUnavailable: 0`; `/readyz` gates traffic | `kubectl rollout undo` (runbook §2) |
| 7 | Artifacts | S3 write failure | S2 | `ArtifactError` counter | job retried once; store is behind interface — swap to standby bucket via config | re-run failed jobs from queue replay |
| 8 | Artifacts | Corrupt/missing media | S2 | checksum mismatch at presign | `bundle.json` sha256 verified at load; atomic writes (.tmp → rename) | regenerate from job replay |
| 9 | Safety | Unsafe output shipped | **S1** | moderation-reject metrics + report pipeline | pre-queue gate (blocklist+classifier); audit journal on every job; kill-switch env `MM_MODERATION_ENABLED` | incident process; job purge tooling |
| 10 | Safety | Prompt-injection to queue metadata | S2 | schema validation 422 | prompts stored as opaque strings; no template interpolation anywhere in the job path | n/a (by construction) |
| 11 | Models | Bad checkpoint promoted | S2 | eval gates (FID/IS/CLIP thresholds in `configs/eval`) | release workflow requires eval report; bundles carry manifests | rollback to previous bundle (registry keeps N) |
| 12 | Infra | Node/pool exhaustion | S2 | KEDA maxed + queue rising | `gpu_max_nodes` cost cap; alert `MmWorkerStarved` | capacity runbook; raise cap consciously |
| 13 | Infra | Zone outage | S1 | node probes | regional cluster, topology spread, PDBs keep ≥2 API replicas | drain + reschedule automatic |
| 14 | Secrets | Key leakage via logs | **S1** | gitleaks + CI scan | raw keys never logged — fingerprint (sha256[:8]) only; env-only loading | rotate keys (quarterly or on alert) |
| 15 | Depend. | Upstream CVE in torch | S2 | pip-audit + Dependabot | pinned base image; grouped update PRs | rebuild images, canary rollout |
| 16 | Load | Coordinated retry storm | S2 | 202-rate spike after incident | Retry-After headers; edge rate limits; exponential client backoff documented in API guide | self-healing |

## The two S1 families, expanded

### Safety incidents (#9)
The moderation gate is *pre-queue*, which means a policy miss never burns
GPU capacity, and every decision is journaled per job. The residual risk
is a classifier gap. Response: pull the job artifact (S3 versioning),
purge by correlation id, tighten the policy (config, not code), and file
the audit trail. The kill-switch env var exists for full-stop scenarios —
availability is deliberately sacrificed for safety without a code deploy.

### Secret incidents (#14)
Assume any key may leak; the design makes leaks low-impact: keys are
read-only gates (not credentials to data), fingerprints not keys ride the
logs, and rotation is a secret-store update + rollout, not a schema
migration.

## Known accepted risks (honesty section)

1. **Single-region Redis in the base deployment** — multi-region posture
   is documented in the scaling doc but the shipped manifests deploy one
   region. Accepted for a reference architecture; flagged for operators.
2. **Queue exactly-once semantics** — the reliable-queue pattern gives
   at-least-once with bounded retry (2). Duplicate *artifacts* are
   checksummed (idempotent keys), duplicate *cost* is bounded; true
   exactly-once was rejected as complexity without user-visible benefit.
3. **Moderation policy is illustrative** — the shipped blocklist is a
   starter set; production operators must own the policy content.
