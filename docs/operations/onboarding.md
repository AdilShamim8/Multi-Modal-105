# On-Call Onboarding

> First week on the bench. Last updated: 2026-10-18 · Prerequisite: [runbook](runbook.md) read end-to-end.

## Day 1–2: understand what you're guarding

1. Run the platform locally: `make docker-up`, then `scripts/smoke_test.sh`.
   You should be able to explain every line of that script's output.
2. Read [system architecture](../architecture/system-architecture.md) and
   draw the request lifecycle from memory. If you can't, re-read — the
   diagram is the mental model for every incident.
3. Open the Grafana dashboard and identify the four numbers that matter:
   throughput by status, queue depth vs running, p95 latency, error rates.

## Day 3–4: rehearse recovery

1. Execute drills #2, #3, #6 from the [game-day catalog](game-day.md)
   against your local stack — Redis restart, worker kill, queue flood.
2. Break something on purpose (delete a bundle, corrupt a manifest) and
   recover using ONLY the runbook. Every gap you hit → runbook PR.
3. Trace one job end-to-end through the logs using its correlation id:
   `docker compose logs api | grep <cid>` then `docker compose logs worker
   | grep <cid>`. This skill is the difference between 5-minute and
   5-hour incidents.

## Day 5: own the loop

1. Shadow the nightly CI runs; triage one failure end-to-end.
2. Write one improvement PR: a test, an alert threshold, a runbook step.
   The bench improves the system every week — that's the job.
3. Confirm your SEV-1 contact path works (pager test with the IC on duty).

## The five questions you must answer before going on the roster

1. What is the difference between `/healthz` and `/readyz`, and which one
   does the load balancer use to drain a bad replica?
2. Where does a job live while a worker executes it, and what recovers it
   if the worker dies?
3. What is the first action in a moderation SEV-1, before any diagnosis?
4. Which knob sheds load, and which alert fires before it does?
5. How do you roll back a bad model without a code deploy?

(Answers: runbook §0, ADR-003, incident-response SEV-1, capacity doc,
registry guide — but know them cold.)
