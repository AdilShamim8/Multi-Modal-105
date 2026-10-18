# Changelog

All notable changes to the Multi-Modal Generation Platform are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — latent production defects surfaced by the mypy sweep (92 → 0 errors)
- **`T2IPipeline.from_pretrained` did not exist.** The serving engine calls it
  for every image-model load; production died with `AttributeError` (tests
  injected fakes, so 215 green tests never saw it). It now loads registry
  bundles — checksum-verified manifest, `config.json` architecture dispatch
  (`dit` | `unet`), `weights.safetensors`, optional `vae.safetensors`, and
  injectable VAE / text encoder / scheduler for shared-component callers.
- **`T2VPipeline` did not exist anywhere.** `serving/engine.py` imported it
  from `agent.pipeline`, which never defined it — every production video job
  died with `ImportError` before the first denoising step. New module
  `agent/t2v_pipeline.py`: causal video VAE + VideoDiT + T5/CLIP conditioning
  behind a deterministic DDIM reverse pass, `generate(...)` → MP4 bytes (or
  raw frame tensors for tests), `from_pretrained()` video-bundle loading.
- **`InferenceEngine._run` only read the `images` result key** — any pipeline
  returning pre-encoded video bytes under `videos` raised
  `GenerationError('no outputs')`. Accepts `images` / `videos` / `artifacts`.
- **`T2IEvaluator.from_folder` did not exist** and `mma evaluate` called it
  plus a `metrics=` kwarg `evaluate_all` never accepted — two `TypeError`s
  before a single metric. The classmethod builds the evaluator from reference
  images (FID degrades to `nan` without them), and the CLI maps `--metrics`
  onto the compute toggles.
- **`GANTrainer` called APIs that never existed**: `CheckpointManager(
  checkpoint_dir=, save_every=, keep_last=)`, `save(generator=, discriminator=,
  ...)`, `MetricLogger(experiment_name=)` — every GAN run and checkpoint save
  raised `TypeError`. Now uses the real APIs (including the new
  `CheckpointManager.save(extra_states=...)` for multi-module checkpoints).
- **`GANConfig.max_grad_norm` and `T2VConfig.max_token_length` were missing** —
  `AttributeError` on the first GAN training step and at T2V trainer init.
- **GAN checkpoint metadata crashed `json.dump`**: `vars(config)` leaves the
  nested `OptimizerConfig` a dataclass; serialized with `dataclasses.asdict`.
- **`DDPMSampler.sample` rejected the `latents` kwarg** — `img2img` with the
  ddpm sampler raised `TypeError`. All three samplers now accept `latents`,
  callable CFG-wrapper models, and `str | torch.device` devices.
- **Dead code removed from the T2V trainer**: a `DDIMSampler` instance with a
  `set_timesteps` lambda monkey-patch that never executed.
- **`verify_repo.sh` could pass without running its steps** (`cmd && ok`
  AND-lists are exempt from `set -e`): steps run bare with an ERR trap;
  fault-injection regression test added.

### Changed — type-safety gate
- `ci.yml`: the mypy step's `continue-on-error` is removed. The codebase is
  mypy-clean (0 errors, 86 files, mypy 1.13.0) — type errors fail the PR
  gate from here on (ADR-002's tightening condition is met).
- `RedisLike` protocol matches redis-py's real stub signatures (names,
  arity, keywords checkable for both `redis.Redis` and the test fakes).
- Register-buffer attributes (`w_avg`, `ema_cluster_size`, `pos_embed`, ...)
  are declared for the type checker; nested `ModuleList` iteration uses
  zero-cost casts; evaluation `evaluate` signatures gained `Literal`
  overloads; `MM_ENV` is runtime-validated before its `Literal` cast.

### Tests
- 10 new regression tests (`tests/unit/test_production_paths.py`) exercise
  the real bundle-loading paths — every one fails on the pre-repair code.
- Suite total: 201 unit + 24 e2e/integration.

The next minor will carry: webhooks for job completion, Redis-backed
distributed rate limiting (multi-replica parity), and the spot-instance
worker profile. See docs/adr/ for the decision log.

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
