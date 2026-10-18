# GANs — Generative Adversarial Networks

> Code: `src/multimodal/models/gans/` · Trainer: `training/gan_trainer.py` (StyleGAN2 system) · Config: `configs/gan/stylegan2_base.yaml` · Last audited: 2026-10-18

## Role in the platform

In a diffusion-centric stack, GANs earn their place two ways: as
**discriminator heads** inside VAE/compression training (sharp
high-frequency detail) and as a **fast sampler family** for the
adversarial track (StyleGAN2 at 256px). They are kept first-class
citizens — the platform's registry treats a GAN exactly like any other
generative component.

## Implementation map

| Component | File | Notes |
|---|---|---|
| `StyleGAN2Generator` | `generator.py` | Mapping net (8 layers, lr_mul 0.01), modulated convs w/ demodulation, learned constant input, ToRGB skip pyramid |
| `StyleGAN2Discriminator` | `discriminator.py` | Residual blocks, minibatch stddev, `r1_penalty()` built in |
| `StyleGAN2` (system) | `stylegan.py` | Softplus/hinge/WGAN-GP losses, **lazy R1** (every 16 steps, amortized), truncation trick, generator EMA |
| `GANTrainer` | `gan_trainer.py` | Alternating D/G steps, grad clipping, checkpointing, sample grids |

## The stabilization stack (why training doesn't diverge)

1. **R1 lazy regularization** — γ=10 applied every 16 critic steps ÷16:
   equal regularization pressure at 1/16 the compute (StyleGAN2 App. B).
2. **Equalized learning rate** (`EqualLinear`/`EqualConv2d`) — weights are
   N(0,1)-parameterized and scaled at runtime; lr stops depending on fan-in.
3. **Generator EMA** (`ema_beta=0.999`) — evaluation and exports use the
   averaged weights; mode-collapse wobbles average out.
4. **β=(0, 0.99) Adam** — the StyleGAN2 finding for adversarial dynamics.
5. **Truncation trick** (ψ=0.7 default at sample time) — trades fidelity
   for diversity control via the tracked `w_avg`.

## Production invariants

* Discriminator LR ≠ generator LR is a **bug** in this codebase (they
  default equal); if you need asymmetry, that's a config change with an
  ADR, not a silent edit.
* EMA weights — never raw weights — are what enter model bundles
  (`agent/export.py`).
* WGAN-GP mode disables R1 (they conflict); the trainer enforces this by
  the loss_type switch, tested in `tests/unit/test_gan_system.py`.

## Verification

Unit tests cover: loss symmetry, lazy-R1 cadence (only steps ≡ 0 mod 16),
truncation pulls w toward the running mean, EMA update math.
