# Documentation Index

> The documentation is the contract: what it says the system does, the
> system does, and the tests check. Last audited: 2026-10-18.

## Architecture (how it is built)
- [System Architecture](architecture/system-architecture.md) — topology, request lifecycle, module invariants
- [Data Flow](architecture/data-flow.md) — training & serving data paths, job states, data ownership
- [API Design](architecture/api-design.md) — surface, versioning, error envelope
- [Scaling to 100M+ Users](architecture/scaling-100m-users.md) — the traffic model and what breaks first
- [Failure Modes (FMEA)](architecture/failure-modes.md) — 16 failure modes with mitigations and accepted risks
- [Capacity Planning](architecture/capacity-planning.md) — per-job GPU cost, queueing math, cost levers

## Operations (how it runs)
- [Runbook](operations/runbook.md) — deploy, rollback, queue backlog, drills
- [Incident Response](operations/incident-response.md) — SEV ladder, safety flows, post-incident template
- [SLOs & SLIs](operations/slos.md) — targets, error budgets, what is deliberately not an SLO

## Models (how the science works)
- [VAE](models/vae.md) · [GANs](models/gans.md) · [Autoregressive](models/autoregressive.md) · [Diffusion](models/diffusion.md)

## Pipelines (how products are built)
- [Text-to-Image](pipelines/t2i.md) · [Text-to-Video](pipelines/t2v.md)

## Guides (how you use it)
- [Getting Started](guides/getting-started.md) · [Training](guides/training.md) · [Serving](guides/serving.md) · [Evaluation](guides/evaluation.md)

## Decisions (why it is this way)
- ADR index: [001 src layout](adr/001-src-layout.md) · [002 configs](adr/002-config-system.md) · [003 Redis queue](adr/003-redis-queue.md) · [004 async-first](adr/004-async-first-api.md) · [005 bundles](adr/005-safetensors-bundles.md) · [006 DiT primary](adr/006-dit-primary.md) · [007 KEDA](adr/007-keda-autoscaling.md) · [008 limits](adr/008-defence-in-depth.md) · [009 GKE](adr/009-terraform-gke.md)

## Learning
- [Tutorial-105](tutorial/Tutorial-105.md) — the full beginner-to-practitioner tutorial (2 parts)
