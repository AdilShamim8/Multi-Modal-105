---
name: Bug report
about: Something is broken — tell us precisely how
title: "[bug] <one-line symptom>"
labels: bug, triage
assignees: []
---

## Environment

- Version / commit: <!-- e.g. v1.0.0 or short SHA -->
- Deployment: <!-- local · docker-compose · k8s (which cluster) -->
- Hardware: <!-- CPU-only / GPU model + VRAM -->
- Python & torch versions: <!-- python -V; python -c "import torch; print(torch.__version__)" -->

## What happened?

<!-- Symptom as an operator observed it. Include the job/request id if serving-related. -->

## What did you expect?

## Minimal reproducer

```bash
# Commands or config that reproduce the fault. Trim until it stops reproducing.
```

## Logs / metrics

```
<!-- Paste structured log lines (JSON) with the correlation_id, or metric jumps. -->
<!-- NEVER paste API keys, bucket credentials, or user prompts containing PII. -->
```

## Severity

- [ ] Blocks me locally
- [ ] Breaks a deployment / rollout
- [ ] Production impact (SLO burn) — also page the on-call per docs/operations/incident-response.md
