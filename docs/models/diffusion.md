# Diffusion Models — the generation core

> Code: `src/multimodal/models/diffusion/` · Samplers: `src/multimodal/sampling/` · Last audited: 2026-10-18

## The two-process mental model

```
Forward (training):  x₀ ──── q(xₜ|x₀) = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε ───▶ xₜ (noise)
Backward (sampling): x_T ─── p(xₜ₋₁|xₜ) ────────────────────▶ x₀ (image)
```

Everything in `diffusion/` is one of: schedule math (when/how much
noise), architecture (what predicts ε or v), or sampling (how to walk
backwards cheaply).

## Implementation map

| Component | File | Key decisions |
|---|---|---|
| `NoiseScheduler` | `noise_scheduler.py` | scaled-linear/cosine betas; **v-prediction default** (stable at high resolution); precomputed ᾱ buffers; DDPM posterior terms |
| Forward process | `forward_backward.py` | `q_sample`, posterior `q(xₜ₋₁|xₜ,x₀)` — the exact closed forms the samplers consume |
| `UNet` | `unet.py` | SD-style: cross-attention text conditioning, GroupNorm+SiLU resblocks, attention resolutions (4,2,1) |
| `DiT` | `dit.py` | Transformer denoiser: patchify latents, **adaLN-Zero** conditioning, learn_sigma variant |
| `VideoDiT` | `dit_video.py` | Factorized space-time attention + causal 3D convs (see T2V pipeline doc) |
| `EMAModel` | `dit.py` | Weight EMA reused by all trainers |

## v-prediction (why the default is not ε)

ε-prediction's loss weighting collapses at high ᾱₜ (the model is asked
to predict noise it cannot see); v-prediction — v = √ᾱₜ·ε − √(1−ᾱₜ)·x₀ —
has a flat signal-to-noise profile across the schedule and trains
stably at 512px+. The scheduler exposes all three prediction types;
switching is a config value, not a refactor.

## min-SNR weighting

`snr_gamma=5.0` clips the per-timestep loss weight at SNR γ — the
effective-epoch finding from Hang et al. 2023: uniform weighting spends
most gradient signal on near-pure-noise timesteps that barely matter
for sample quality. One line of config, measurably faster convergence.

## Sampler contract

All three samplers (`ddpm`, `ddim`, `dpm_solver`) present one interface:
`sample(model, shape, ...)` where `model(x, t) → prediction`. CFG is
applied by wrapping the model (doubling batch, guidance in the wrapper —
see `agent/pipeline.py`), so samplers stay guidance-free and testable in
isolation.

* **DDPM** — ancestral, 1000 steps: the reference implementation.
* **DDIM** — deterministic (η=0): reproducibility golden path (same seed
  ⇒ byte-identical output; the regression tests depend on this).
* **DPM-Solver++** — 2nd-order multistep: the production default, 30
  steps at 50-step DDIM quality.

## Verification

`tests/unit/`: schedule monotonicity (ᾱ strictly decreasing, in (0,1)),
posterior variance positivity, v↔ε↔x₀ round-trip identities, DDIM
determinism under fixed seed, DPM++ consistency vs DDIM at high steps.
