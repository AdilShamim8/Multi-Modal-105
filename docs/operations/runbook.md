# Operations Runbook

> Status: current · Last audited: 2026-10-18 · Alerts reference sections here by anchor.

## 0. Golden rules

1. **Liveness ≠ readiness.** A pod can be alive but not serving — check
   `/readyz` before declaring an incident over.
2. **The queue is the truth.** Metrics tell you what happened; queue depth
   tells you what is *about to* happen.
3. **Roll forward for code, roll back for deploys.** Bad code → fix;
   bad rollout → `rollout undo` (60 s), then diagnose.

## 1. Deploy

```bash
# CI builds + pushes images on tag; manual gate for prod:
kubectl -n multimodal set image deploy/multimodal-api \
    api=ghcr.io/adilshamim8/multimodal-api:NEW_TAG
kubectl -n multimodal rollout status deploy/multimodal-api --timeout=5m

# Watch for readiness gate failures (model load / queue connect):
kubectl -n multimodal get events --sort-by=.lastTimestamp | tail -20
```

Workers drain automatically (SIGTERM → finish in-flight → exit); their
rolling update is bounded by `terminationGracePeriodSeconds: 120`.

## 2. Rollback

```bash
kubectl -n multimodal rollout undo deploy/multimodal-api
kubectl -n multimodal rollout undo deploy/multimodal-worker
```

Model-level rollback (no deploy): repoint the registry key —
`MM_T2I_MODEL` in the ConfigMap, then restart workers. Bundles are
immutable and checksummed; the previous one is still there.

## 3. Queue backlog (#queue-backlog)

Symptoms: `MmQueueBacklogGrowing` page; users see slow job completion.
1. `kubectl -n multimodal get pods -l component=worker` — count healthy.
2. If healthy but backlog grows: raise KEDA scale ceiling deliberately:
   `kubectl patch scaledobject multimodal-worker -n multimodal --type merge \
     -p '{"spec":{"maxReplicaCount":80}}'` (cost cap decision — escalate).
3. If unhealthy: check DCGM/GPU node health, restart workers (safe —
   `recover_stuck()` requeues in-flight jobs on the replacement).
4. If admission control is shedding (503s): announce status, raise
   `MM_QUEUE_MAX_DEPTH` only if the backlog is *draining*.

## 4. Redis unavailability

API automatically drains from LB (readyz fails). Jobs enqueued before the
outage survive AOF. Recovery: sentinel failover (or pod restart), then
verify with `redis-cli llen mm:generation:jobs`. No manual job surgery —
the recovery scan handles strays.

## 5. Generation failures spike (#generation-failures)

```bash
kubectl -n multimodal logs -l component=worker --since=10m \
  | grep -E '"level":"(error|warning)"' | jq -c '{job_id, msg, details}'
```

Common shapes: OOM loops (one bad geometry → engine degrades twice →
fails; look at `details.free_gb`), artifact store auth (check secret
rotation), model bundle checksum mismatch (registry corruption — re-export).

## 6. Moderation policy update

Blocklist/classifier changes are config (ConfigMap `MM_MODERATION_MODE`)
+ policy file — no image rebuild. Roll API pods to apply. Rejections are
journaled per job; `mm_api_requests_total{status="400"}` tracks rate.

## 7. Key rotation (quarterly or on alert)

Update secret store → `kubectl -n multimodal create secret ... --dry-run` →
rolling restart of API. Old keys stay valid for one overlap window
(comma-separated list), then are removed in a second pass — zero-downtime.

## 8. Drills

Monthly game-day: zone failure (drain a zone), Redis failover, bad-model
rollback, moderation kill-switch. Every drill ends with a runbook PR —
the runbook is versioned with the code for exactly this reason.
