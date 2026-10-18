# Training Guide

> Last audited: 2026-10-18 · Configs: `configs/` · Code: `src/multimodal/training/`

## The training ladder

Train the planes **in dependency order** — each stage freezes and feeds
the next:

```
1. VAE          configs/vae/vae_base.yaml       → compression network
2. (optional) GAN discriminator head joins VAE training at step 50k
3. T2I diffusion configs/t2i/dit_base.yaml      → image denoiser
4. T2V diffusion configs/t2v/dit_video_base.yaml → video denoiser
```

## Launching a run

```bash
mma train --pipeline t2i --config configs/t2i/dit_base.yaml
```

Overrides without editing files (env interpolation wins):

```bash
MM_T2I_BATCH=32 MM_OUTPUT_DIR=/mnt/runs/exp-17 mma train --pipeline t2i --config configs/t2i/dit_base.yaml
```

## What every trainer gives you for free

* Deterministic seeding (`seed_everything`) — reruns are comparable.
* EMA weights — what evaluation and export consume.
* Atomic checkpointing (write-tmp-rename, N-last retention) — a crashed
  run never corrupts the resume point.
* `is_main_process()`-guarded logging — no 8× duplicated metrics under DDP.
* Resume: pass `--resume <checkpoint>`; step + optimizer state restored.

## Reading a run

```bash
make stats                       # repo hygiene
tensorboard --logdir outputs/t2i_dit_base/tb   # if tensorboard enabled
```

Early-loss sanity checks (the 15-minute investment that saves days):
1. VAE: recon falling, KL rising smoothly — if KL spikes, lower `kl_weight`.
2. Diffusion: loss should be ~1.0 at start (v-prediction is normalized);
   a 0.01 loss means the model predicts t=0 trivially — check the
   timestep sampler.
3. GAN: d_loss oscillating around g_loss is HEALTHY; a monotonically
   falling d_loss means the critic won — raise `r1_gamma`.

## Before promoting a checkpoint

```bash
mma evaluate --config configs/eval/default.yaml --samples-dir outputs/samples
```

Release gates: FID ≤ 25, IS ≥ 8, CLIP ≥ 0.25 (`configs/eval/default.yaml`
thresholds). Then export a bundle:

```bash
mma export --model-dir outputs/t2i_dit_base/final --format torchscript --out .../model.torchscript
```

The bundle manifest (commit, config, sha256) is what the registry and
rollbacks key off — never hand-copy raw checkpoints into serving.
