# Text-to-Image Pipeline

> Code: `src/multimodal/agent/pipeline.py` (T2IPipeline), `training/t2i_trainer.py` · Config: `configs/t2i/` · Last audited: 2026-10-18

## End-to-end composition

```
prompt ──▶ CLIP text encoder ──▶ context (B, 77, 768)
                                      │ cross-attention
noise z_T ──▶ U-Net or DiT ◀──────────┘ + timestep (adaLN)
    ▲                     │
    └── sampler loop ──────┘  (DDPM | DDIM | DPM-Solver++)
                                      │ latents (B, 4, H/8, W/8)
                          VAE decode ─┴─▶ PIL images
```

`T2IPipeline` bundles the four components behind one `__call__`:
prompt encode → CFG-wrapped denoise → VAE decode → postprocess. `img2img`
reuses the same machinery after DDIM-inverting the init image to a noisy
latent at the requested `strength`.

## Data preparation (the unglamorous 80%)

Filters applied at curation time (`configs/t2i/dit_base.yaml`):
resolution floors, caption length ≥ 5 chars, CLIP aesthetic ≥ 4.5,
perceptual-hash dedup. Each filter is journaled — dataset versions are
reproducible, which is what makes a FID comparison between two training
runs *mean something*.

## Training details that matter

* **Caption dropout 10%** — trains the unconditional branch CFG needs;
  without it, guidance has nothing to steer away from.
* **min-SNR-γ=5 loss weighting** — see the diffusion model doc.
* **EMA 0.9999** — sampled grids and eval use EMA weights.
* **Latents precomputed once** (frozen VAE) — 40%+ step-time saving.

## Classifier-free guidance at inference

```
ε_guided = ε_uncond + s·(ε_cond − ε_uncond),  s default 7.5
```

Implemented as a batch-doubled model wrapper (one forward, both branches
— the memory-efficient formulation). `s` is per-request; the API schema
bounds it at 30 (above ~20, saturation and color artifacts dominate).

## Serving integration

The engine consumes the pipeline through `generate(**params)`:
`multimodal/serving/engine.py` handles seeding (per-job deterministic),
OOM degradation (halve outputs → cut steps), and PNG encoding. The
pipeline never imports serving code — dependency direction is one-way.

## Verification

Integration test (`tests/integration/test_t2i_pipeline.py`): tiny VAE +
DiT on CPU, 8×8 latents, 4 steps — asserts output shapes, seed
reproducibility, and CFG-scale sensitivity (s=0 vs s=7.5 differ).
