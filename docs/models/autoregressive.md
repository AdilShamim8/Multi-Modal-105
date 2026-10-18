# Autoregressive Models — VQ-VAE token generation

> Code: `src/multimodal/models/autoregressive/vqvae.py` · Last audited: 2026-10-18

## Role in the platform

The third generative family in the overview: compress images into a
**discrete token grid** (VQ-VAE), then model token sequences
autoregressively. In this codebase the tokenizer half is fully
implemented (codebook, EMA updates, straight-through gradients); it is
the component that connects this repo's GAN/VAE work to the modern
"tokens → transformer" paradigm used by DALL·E-class systems.

## Implementation map

| Component | Location | Notes |
|---|---|---|
| Codebook | `vqvae.py` | Vector quantization with **EMA codebook updates** (commit β-free; EMA is more stable than codebook loss) |
| Straight-through estimator | `vqvae.py` | Gradients bypass the non-differentiable argmin |
| Commitment loss | `vqvae.py` | β·‖z − sg(e)‖² pulls encoder outputs to codebook entries |

## Why EMA codebooks

A learned codebook tensor fights the encoder for gradients and dies
silently (codebook collapse: all inputs map to one entry). EMA updates
track cluster means directly — no codebook optimizer, no collapse mode
by construction. Dead codes are re-seeded from random encoder outputs
(the standard revival trick), keeping effective vocabulary utilization
measurable (exposed as a metric for the trainer).

## Position in the bigger picture

* **vs VAE**: discrete latents give exact reconstruction at high rates +
  a natural interface for transformers; VAE latents are continuous and
  pair with diffusion instead.
* **vs diffusion**: AR generation is one-token-at-a-time (slow at
  512px even with 16×16 token grids), but *compositional* — the same
  transformer prior that powers text models works unchanged on image
  tokens.
* **In this repo**: the VQ tokenizer can plug into the compression
  stage as an alternative to `AutoencoderKL`; the registry + config
  surface exists for exactly that kind of experiment.

## Verification

Unit tests (`tests/unit/test_vqvae.py`): quantization round-trip shapes,
straight-through gradient flow, commitment-loss direction, codebook EMA
step moves entries toward batch means.
