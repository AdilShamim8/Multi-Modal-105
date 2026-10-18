# Service Level Objectives

> Status: current · Last audited: 2026-10-18 · Window: 30 days rolling

## SLOs (user-facing, the contract)

| # | Objective | Target | Measured by |
|---|---|---|---|
| SLO-1 | Job completion success (of admitted jobs) | ≥ 99.5% | `mm_generations_total{status}` |
| SLO-2 | End-to-end p95 latency (image, balanced profile) | ≤ 60 s | `mm_generation_latency_seconds` |
| SLO-3 | API availability (non-5xx on /v1) | ≥ 99.9% | `mm_api_requests_total` |
| SLO-4 | Status read p95 | ≤ 200 ms | request latency histogram |
| SLO-5 | Artifact URL validity (fetch succeeds) | ≥ 99.9% | CDN + S3 logs |

## Error budgets (the enforcement mechanism)

* SLO-3's 99.9% = **43 min/month** of tolerable API unavailability.
* Budget burn alert at 50% and 90% consumption (measured as 5xx rate
  integrated over the window).
* **Budget exhausted → feature freeze for reliability work.** This is the
  cultural mechanism that makes the numbers real instead of decorative.

## What is deliberately NOT an SLO

* **Queue wait at peak surge** — bounded by admission control, not
  promised; the honest failure mode is a fast 503, not a slow lie.
* **Video latency parity with images** — video is 5–8× the cost; its
  budget is 5 min p95 and stated separately in capacity planning.
* **Moderation perfection** — a process control with audit trails, not a
  guarantee; residual risk is owned by the incident process, not hidden
  in an SLO.

## SLI implementation notes

* Latencies are measured **inside the platform** (submission → artifact
  write), excluding client network — that's what we control.
* Success excludes jobs cancelled by users and safety rejections (they
  are correct behaviour, not failures).
* Every SLI is derivable from the metric contract in
  `src/multimodal/serving/metrics.py` — dashboards and alerts reference
  those names verbatim; renaming a metric is a breaking change and
  treated like an API change.
