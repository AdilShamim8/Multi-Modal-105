# VAE — Variational Autoencoder (compression stage)

> Code: `src/multimodal/models/vae/` · Trainer: `training/vae_trainer.py` · Config: `configs/vae/vae_base.yaml` · Last audited: 2026-10-18

## Role in the platform

The VAE is the *compression network* that makes latent diffusion
affordable: 512×512×3 images become 64×64×4 latents (48× fewer tokens for
the denoiser to attend over). Generation quality is bounded by VAE
reconstruction quality — which is why it is trained first and frozen.

## Implementation map

| Component | File | Notes |
|---|---|---|
| `DiagonalGaussian` | `vae.py` | Reparameterization trick; logvar clamped [-30, 20] |
| `AutoencoderKL` | `vae.py` | 8× spatial downsample, 4-ch latent, `scaling_factor=0.18215` |
| `Encoder` / `Decoder` | `encoder.py` / `decoder.py` | ResBlocks + attention at 16px, channel multipliers (1,2,4,8) |
| Losses | trainer | recon (MSE/perceptual) + KL (`kl_weight=1e-6`) + adversarial after step 50k |

## Why these numbers

* **KL weight 1e-6** — the LDM finding: a nearly-unregularized latent
  reconstructs better; the KL term only keeps the latent space
  *sampleable* for the diffusion model that follows.
* **Adversarial loss from step 50k** (`disc_start_step`) — early
  adversarial training destabilizes reconstruction; late, it sharpens
  high-frequency detail the MSE term leaves blurry.
* **scaling_factor 0.18215** — normalizes latent unit-variance so the
  diffusion schedule's noise assumptions hold. All encode/decode helpers
  (`encode_to_latent` / `decode_from_latent`) apply it symmetrically.

## Production invariants

1. VAE weights are **frozen** during diffusion training — never load a
   trainable copy by accident (checkpoint manifests make this auditable).
2. Decode memory dominates serving VRAM at 1024px — the engine degrades
   batch size before degrading resolution (quality-floor policy).
3. Any change to `latent_channels` or downsample factor is a breaking
   change for every downstream bundle — version-bump required.

## Verification

`tests/unit/test_vae.py` checks: encoder/decoder shape contracts, KL
non-negativity, logvar clamping, scaling-factor round-trip symmetry.
