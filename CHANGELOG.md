# Changelog

All notable changes to the Multi-Modal Generation Platform are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing queued. The next minor will carry: webhooks for job completion,
Redis-backed distributed rate limiting (multi-replica parity), and the
spot-instance worker profile. See docs/adr/ for the decision log.

## [1.0.1] — 2026-10-18

**Pipeline repair release.** The 1.0.0 tag was cut before the CI/CD
surfaces had ever run to completion: four of its checks failed and no
container images or distribution artifacts were published. This release
makes every gate green and ships the first working release pipeline.

**Verified at release (2026-10-18):** `ci` and `security` workflows fully
green on main (lint, mypy, unit tests on 3.10/3.11/3.12, package builds,
gitleaks, pip-audit, CodeQL, trivy) · 213 passing tests · 578 lint
findings eliminated.

### Fixed — CI/CD
- `release`: image tags embedded the mixed-case owner `AdilShamim8`,
  which GHCR rejects ("repository name must be lowercase"); the owner is
  now lowercased once per job via bash parameter expansion.
- `release`: api/worker images built `FROM multimodal/base:1.0.0`, an
  image no pipeline ever produced. A `containers-base` job now builds
  and pushes the base image to GHCR first; derived images consume it
  via the new `BASE_IMAGE` build-arg. `workflow_dispatch` releases check
  out the requested tag.
- `docker-build` (PR gate): same missing-base defect, plus buildx images
  were never `load`ed, so the smoke test could not see them. Base now
  builds locally per PR; all images load into the daemon and smoke-test.
- `security/trivy`: pinned `aquasecurity/trivy-action@0.28.0` did not
  exist (tags are `v`-prefixed) — the job died in 2 s. Now pinned to
  `v0.36.0`.
- `security/pip-audit`: the invocation passed `pyproject.toml` to
  `--desc` and errored before auditing anything. The gate now resolves
  the complete runtime tree (CPU torch + `.[t2i,t2v,serving]`) in a
  dedicated venv, freezes it to exact pins, and audits those pins with
  `--disable-pip --no-deps --strict`.

### Fixed — Code (found by the newly-enforced lint gate)
- `agent`: `imageio` was referenced but never imported — every gif/mp4
  export raised `NameError` after minutes of sampling. Lazy import with
  an actionable install hint.
- `models`/`training`: missing `Union`/`Tuple` imports (F821).
- `training/t2i`: `_load_vae` logged "Loaded VAE from HuggingFace" while
  discarding the loaded object — the log now states the transfer is not
  implemented.
- `serving`: `zip(artifacts, extensions)` pairs now enforce
  `strict=True` — silent truncation would drop artifact files on a 200 OK.
- 13 dead assignments removed; typing modernized (`X | None`, builtin
  generics); import order canonicalized; `verify_repo.sh` now runs the
  same ruff gates as CI so local and remote cannot drift again.

### Fixed — Packaging & Tests
- `test` extra now installs the serving plane (`multimodal-agent[serving]`)
  — CI unit jobs died on `ModuleNotFoundError: redis`. Previously
  undeclared imports are declared: `starlette` (serving), `scipy` (t2i),
  and a new `train` extra (`accelerate`, `wandb`, `webdataset`).
- `tests/` is a real package and `tests/conftest.py` prepends the repo
  root, so the suite imports identically under bare `pytest` (CI) and
  `python -m pytest` (local).

## [1.0.0] — 2026-10-18

The productionization release: the research codebase is reorganized into a
deployable platform with serving, CI/CD, infrastructure, and operations
documentation. **Curator: @AdilShamim8.**

**Verified at release (2026-10-18):** 219 tracked files · 110 Python
modules · **213 passing tests** (unit + e2e + integration) · full local
gate reproducible via `scripts/verify_repo.sh`.

### Added — Platform & Architecture
- `src/` layout under the `multimodal` package; all legacy root packages
  migrated with import rewrites (history preserved via `git mv`).
- Core runtime: exception hierarchy, structured JSON logging, component
  registry, deterministic seeding, device management, timing utilities.
- Configuration system: dataclass experiment configs + YAML loader with
  environment-variable interpolation and schema validation.
- CLI (`python -m multimodal.cli`): `train`, `generate`, `evaluate`,
  `serve`, `worker` — replaces the previously broken console scripts.

### Added — Serving Layer
- FastAPI application factory with health/readiness, generation, and job
  status routes; OpenAPI schemas for every endpoint.
- Redis-backed asynchronous job queue with admission control, TTL, and a
  state machine (queued → running → succeeded/failed/cancelled).
- GPU worker with graceful shutdown, warmup, and artifact upload
  (local filesystem or S3-compatible stores, presigned URLs).
- Safety: prompt moderation gate (blocklist + pluggable classifier hooks).
- Observability: Prometheus metrics, request correlation IDs, optional
  OpenTelemetry tracing.

### Added — Infrastructure & CI
- Dockerfiles for API, worker, and training planes; docker-compose dev stack.
- Kubernetes manifests: API deployment/service/HPA/ingress, GPU worker
  deployment, KEDA queue-driven autoscaling, ConfigMap, Secret template,
  PodDisruptionBudget, NetworkPolicies.
- Terraform: regional GKE cluster with GPU node pool and staging variant.
- GitHub Actions: lint, typecheck, unit tests (PR gate), nightly integration,
  security scanning (CodeQL, pip-audit, gitleaks, Trivy), release workflow.
- Monitoring stack: Prometheus scrape config, alert rules, Grafana dashboard.

### Added — Documentation
- Architecture docs: system overview, data flow, API design, scaling to
  100M+ active users, failure-mode analysis (FMEA), capacity planning.
- Operations docs: runbook, incident response, SLO/SLI definitions.
- Model deep-dives (VAE, GANs, autoregressive, diffusion) and pipeline
  guides (T2I, T2V), plus eight architecture decision records (ADRs).
- Full tutorial relocated to `docs/tutorial/Tutorial-105.md`.

### Added — Tests
- Unit suite (config, schedulers, VAE/VQ/DiT/U-Net shapes, samplers,
  metrics, queue, auth/rate-limit), integration smoke tests, API e2e tests,
  and a Locust load-test scenario.

### Fixed
- `setup.py` console scripts referenced non-existent modules
  (`training.train`, `sampling.sample`, `evaluation.evaluate`); packaging
  now ships working entry points via `pyproject.toml`.
- Empty placeholder modules implemented: `gan_config.py`, `t2i_config.py`,
  `t2v_config.py`, `stylegan.py`, `gan_trainer.py`.
- Import paths, line endings, and repository hygiene (gitattributes,
  editorconfig, pre-commit gates).

### Compatibility
- Python ≥ 3.10; PyTorch ≥ 2.1 (CPU or CUDA).
- No pretrained weights ship with the repository — fetch via the model
  registry (see `docs/guides/getting-started.md`).

[Unreleased]: https://github.com/AdilShamim8/Multi-Modal-105/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AdilShamim8/Multi-Modal-105/releases/tag/v1.0.0
