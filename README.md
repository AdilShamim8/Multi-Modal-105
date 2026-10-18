# Multi-Modal Generation Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-served-009688?style=flat-square&logo=fastapi&logoColor=white)
![Kubernetes](https://img.shields.io/badge/K8s-ready-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![Tests](https://img.shields.io/badge/tests-213%20passing-2EA44F?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Last audit](https://img.shields.io/badge/last%20audit-2026--10--18-blue?style=flat-square)

**A production-engineered platform for text-to-image and text-to-video
generation — VAE, GAN, autoregressive, and latent-diffusion (U-Net / DiT)
implementations wrapped in a serving system designed for internet scale.**

[Overview](#what-this-is) ·
[Architecture](#architecture) ·
[Quickstart](#quickstart) ·
[Repo Map](#repository-map) ·
[Scaling to 100M Users](docs/architecture/scaling-100m-users.md) ·
[Tutorial](docs/tutorial/Tutorial-105.md)

</div>

---

## What this is

Two planes in one codebase, sharing nothing but types:

1. **The research plane** — from-first-principles implementations of every
   modern generative family: KL-regularized VAE, StyleGAN2 (with lazy R1,
   truncation, EMA), VQ-VAE tokenizers, and latent diffusion in both
   U-Net and DiT form, with DDPM/DDIM/DPM-Solver++ samplers and a full
   evaluation suite (IS, FID, CLIP, diversity).
2. **The serving plane** — the production system around them: async
   job API with admission control, Redis reliable-queue, GPU workers
   with continuous batching and OOM degradation, S3 artifact store,
   moderation gates, Prometheus/OpenTelemetry observability, and
   Kubernetes + KEDA autoscaling.

**Honest scope note:** this repository ships the *system*, not weights.
Model binaries are excluded by policy; you train them with the included
trainers and promote them through the eval-gated export pipeline. What
you get today is a coherent, tested platform architecture — 186 tests,
CI, and operations documentation included.

## Architecture

```
                      ┌────────────── Edge (TLS, rate limits) ──────────────┐
                      ▼
        ┌───────────────────────── API tier (stateless, HPA 3→40) ─────────┐
        │  auth → rate limit → moderation → validation → admission control │
        └───────┬──────────────────────────────────────────┬───────────────┘
        LPUSH   │                                          │ presign
                ▼                                          ▼
        ┌─ Redis (reliable queue) ─┐              ┌─ S3 artifacts + CDN ─┐
        └───────┬──────────────────┘              └──────────▲───────────┘
        BRPOPLPUSH│ recover_stuck()                           │ write
                ▼                                            │
        ┌─ GPU workers (KEDA 2→60 on queue depth) ────────────┴───┐
        │  continuous batching → engine (OOM-degrade → retry ≤2)   │
        │  model registry volume (checksummed bundles)             │
        └───────────────────────────────────────────────────────────┘
```

Deep dives: [system architecture](docs/architecture/system-architecture.md) ·
[data flow](docs/architecture/data-flow.md) ·
[failure modes (FMEA)](docs/architecture/failure-modes.md) ·
[scaling to 100M+ users](docs/architecture/scaling-100m-users.md) ·
[capacity planning](docs/architecture/capacity-planning.md)

## Quickstart

```bash
git clone https://github.com/AdilShamim8/Multi-Modal-105.git
cd Multi-Modal-105
make install-dev          # CPU torch + editable install + pre-commit
make test-unit            # 186-test suite, green in seconds

# Local serving stack (api + worker + redis + prometheus + grafana):
make docker-up
curl -s -X POST localhost:8000/v1/generations \
     -H "X-API-Key: dev-local-key" -H "Content-Type: application/json" \
     -d '{"prompt":"a lighthouse in a storm, cinematic lighting"}' | jq
```

API docs at `http://localhost:8000/docs`. Train a plane:

```bash
mma train --pipeline t2i --config configs/t2i/dit_base.yaml
mma evaluate --config configs/eval/default.yaml --samples-dir outputs/samples
mma export --model-dir outputs/t2i_dit_base/final --format torchscript --out …
```

More: [getting started](docs/guides/getting-started.md) ·
[training guide](docs/guides/training.md) ·
[serving guide](docs/guides/serving.md).

## Repository map

```
src/multimodal/           the package (src layout, PEP 561 typed)
├── core                  version, exceptions, registry, logging, utils
├── config/               dataclass configs + strict YAML loader
├── models/               vae · gans · autoregressive · compression · diffusion
├── data/                 datasets, transforms, latent caching
├── sampling/             DDPM · DDIM · DPM-Solver++
├── training/             trainers for every plane (incl. GAN with lazy R1)
├── evaluation/           IS · FID · CLIP · diversity · evaluator
├── agent/                T2I pipeline + deployment export (bundles)
├── serving/              API · queue · workers · safety · observability
└── cli.py                `mma` control plane
configs/                  YAML experiments (t2i · t2v · vae · gan · eval · serving)
deployments/              docker · k8s (KEDA, PDB, netpol) · terraform · monitoring
tests/                    unit · integration · e2e · load (186 tests)
scripts/                  preload · healthcheck · benchmark · smoke · backup
docs/                     architecture · operations · models · guides · ADRs · tutorial
```

## Quality gates

| Gate | Command | Status |
|---|---|---|
| Full suite (CPU, <90 s) | `make test-unit` + e2e | 213 passing |
| One-command gate | `scripts/verify_repo.sh` | compile · imports · manifests · tests |
| Lint / format | `make lint` | ruff-clean |
| Type check | `make typecheck` | mypy (lenient, tightening per ADR-002) |
| Security | CI `security.yml` | gitleaks · pip-audit · CodeQL · Trivy |
| Coverage | `make coverage` | enforced on new code |

Defects fixed during the 2026-10-18 productionization (each with a
regression test): broken console entry points, five empty stub modules,
DiT missing-import (never instantiable), U-Net skip accounting (never
constructible), StyleGAN modulated-conv view crash, minibatch-stddev
shape crash, DiT/pipeline interface mismatch.

## Documentation

- **Architecture**: [system](docs/architecture/system-architecture.md) · [data flow](docs/architecture/data-flow.md) · [API design](docs/architecture/api-design.md) · [scaling](docs/architecture/scaling-100m-users.md) · [FMEA](docs/architecture/failure-modes.md) · [capacity](docs/architecture/capacity-planning.md)
- **Operations**: [runbook](docs/operations/runbook.md) · [incident response](docs/operations/incident-response.md) · [SLOs](docs/operations/slos.md) · [game-day drills](docs/operations/game-day.md)
- **Models**: [VAE](docs/models/vae.md) · [GANs](docs/models/gans.md) · [autoregressive](docs/models/autoregressive.md) · [diffusion](docs/models/diffusion.md)
- **Pipelines**: [T2I](docs/pipelines/t2i.md) · [T2V](docs/pipelines/t2v.md) · [data prep (image)](docs/guides/data-preparation-t2i.md) · [data prep (video)](docs/guides/data-preparation-t2v.md)
- **Decisions**: 11 ADRs in [docs/adr/](docs/adr/) — src layout, configs, queue, async-first, bundles, DiT-primary, KEDA, limits, GKE, testing, metrics
- **Learning**: the full [Tutorial-105](docs/tutorial/Tutorial-105.md) (beginner → practitioner, 2 parts)
- [FAQ](docs/faq.md) · [glossary](docs/glossary.md) · [CHANGELOG](CHANGELOG.md)

## Contributing

`main` stays green: small atomic commits, conventional messages, tests
with every fix. See [CONTRIBUTING.md](CONTRIBUTING.md) ·
[SECURITY.md](SECURITY.md) · [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT © 2026 Adil Shamim — see [LICENSE](LICENSE).
