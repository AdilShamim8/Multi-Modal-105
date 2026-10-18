# Incident Response

> Status: current · Last audited: 2026-10-18 · Pager: alerts.yml severity ladder (page / ticket / info)

## Severity definitions

| Level | Definition | Response |
|---|---|---|
| **SEV-1** | Safety incident (unsafe media shipped, secret leak) or full outage | Page immediately; incident commander named within 5 min |
| **SEV-2** | SLO breach (queue p95, error rate), capacity loss > 30% | Ticket + on-call within 15 min |
| **SEV-3** | Single-replica failures, alert noise, toil | Next business day |

## SEV-1 safety flow (#moderation-spike, unsafe output)

1. **Stop the bleeding** — kill-switch, not diagnosis:
   `MM_MODERATION_ENABLED` → block-all is NOT implemented as a separate
   mode; set `MM_MODERATION_MODE=classifier` with an always-reject hook,
   or scale API to refuse via ingress (last resort).
2. **Evidence preservation** — export the job's audit trail:
   `redis-cli LRANGE mm:job:{id}:events 0 -1` + artifact key from S3
   (versioned — do not delete, quarantine bucket).
3. **Scope** — same correlation id / API key fingerprint across the last
   24h; the journaled events make this a query, not a forensics project.
4. **Policy fix** — blocklist/ConfigMap change (no deploy) → API roll.
5. **Post-incident** — blameless review within 72h; every SEV-1 produces
   a runbook PR and (if applicable) a test that would have caught it.

## SEV-1 outage flow

1. Stabilize over diagnose: rollback last deploy (runbook §2), then
   check queue → workers → redis → API in that order (dependency order).
2. Comms cadence: status page update within 10 min, every 30 min after.
3. If queue must be drained to recover: `redis-cli DEL mm:generation:jobs`
   is the documented last resort — failed jobs were already acked to
   users as 202; a follow-up notification batch is part of the drill.

## Roles

* **Incident commander** — runs the timeline, decides mitigation vs.
  diagnosis, owns comms. Never also the person typing.
* **Operator** — executes runbook steps, reports results verbatim.
* **Scribe** — timestamped decisions in the incident channel.

## Post-incident review template

```
Summary:        one paragraph, user-visible impact
Timeline:       detection → mitigation → resolution (UTC timestamps)
Contributing:   technical + process causes (plural by construction)
What worked:    alerts that fired early, runbook steps that worked
What didn't:    silent gaps, manual steps that should be automated
Actions:        owner + deadline per action; runbook PRs same day
```

## Links

Alert → runbook mapping lives in `deployments/monitoring/alerts.yml`
annotations. If an alert fires with no runbook section, the FIRST action
of that incident is writing one.
