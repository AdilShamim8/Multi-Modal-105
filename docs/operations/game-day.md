# Game-Day Drill Catalog

> Last updated: 2026-10-18 · Cadence: monthly · Every drill ends with a runbook PR.

## Why drills exist

The runbook describes recovery under calm conditions. Incidents happen
under adrenaline. The difference is rehearsed muscle memory — and the
drills are where runbook steps are discovered to be wrong while the
stakes are zero.

## The catalog

| # | Drill | Inject | Expected behaviour | Pass criteria |
|---|---|---|---|---|
| 1 | Zone failure | `kubectl drain` a zone's nodes | PDBs hold ≥ 2 API replicas; pods reschedule | no SLO burn; < 5 min rebalance |
| 2 | Redis failover | restart redis pod mid-load | API drains from LB; jobs survive AOF; workers hold floor | zero job loss; recovery < 5 min |
| 3 | Worker crash mid-job | `kill -9` a worker running a batch | replacement pod `recover_stuck()` requeues | orphan requeued; user sees success or retry |
| 4 | Bad model rollout | deploy bundle with regressed FID | eval gate blocks promotion; manual rollback drill | gate fired; rollback < 2 min via registry repoint |
| 5 | Moderation kill-switch | policy incident simulation | block-all mode engages without deploy | no unsafe generation during drill |
| 6 | Queue flood | locust at 10× capacity | admission sheds with 429/503; alerts fire in order | shedding works; `MmQueueNearCapacity` pages |
| 7 | Key rotation | rotate live keys | overlap window; zero 401s for valid clients | no client disruption |

## Rules of engagement

1. **Production-like, not production** — staging cluster, synthetic
   traffic, real manifests.
2. **The scribe records timestamps** — drill reports use the same
   template as incident reviews.
3. **Every friction point becomes a runbook PR the same day** — a step
   that needed guessing is a bug in the runbook.
4. **Rotate the incident-commander role** — the on-call bench must be
   at least two people deep, always.
