# ADR-011: The metric contract is an API

- **Status:** accepted (2026-10-18)
- **Context:** dashboards, alert rules, and the Grafana layouts reference
  metric names as strings (`mm_generations_total`, ...). A rename in
  `serving/metrics.py` silently breaks alerting — the exact failure mode
  the "docs are the contract" rule exists to prevent.
- **Decision:**
  1. Metric names and label sets in `serving/metrics.py` are a **public
     contract**, versioned like the HTTP API.
  2. Renaming requires: deprecation window with dual export, alert-rule
     update in the same PR, changelog entry under "breaking".
  3. `tests/unit/test_platform.py` freezes the names — CI fails on
     accidental drift.
  4. `deployments/monitoring/alerts.yml` annotations must reference
     runbook anchors; an alert without a runbook link fails review.
- **Consequences:**
  - (+) Alerting survives refactors by construction.
  - (+) "Which dashboard shows queue depth?" has one answer forever.
  - (−) New metrics carry process overhead (test + alert + doc in the
    same change) — deliberately, because unbudgeted metrics rot.
