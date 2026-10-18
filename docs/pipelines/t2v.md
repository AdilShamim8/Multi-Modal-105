# Text-to-Video Pipeline

> Code: `src/multimodal/models/compression/video_vae.py`, `models/diffusion/dit_video.py`, `training/t2v_trainer.py` · Config: `configs/t2v/dit_video_base.yaml` · Last audited: 2026-10-18

## The T2V system (overall)

```
text ─▶ encoder ─▶ context
                          │
noise (C·T/4, H/8, W/8) ─▶ VideoDiT ── sampler ──▶ video latents
                          │                              │
        3D RoPE (t,h,w) ──┘                 causal 3D VAE decode
                                                         │
                                                    MP4 frames
```

## Stage 1 — compression network (LDM for video)

`VideoAutoencoderKL` is a **causal 3D VAE**: 4× temporal, 8× spatial
compression. Causality matters for two production reasons:
* **Streaming decode** — frames emerge as available; chunked decode
  keeps peak VRAM flat regardless of clip length.
* **Frame-extension** — extend a clip by decoding new temporal chunks
  without re-encoding the past.

## Stage 2 — DiT for video (factorized attention)

Full 3D self-attention over (T/4)·(H/8/2)·(W/8/2) tokens is quadratic
suicide. `VideoDiT` factorizes: **spatial attention within each frame,
temporal attention across frames at each spatial position** — O(T·S² +
S·T²) instead of O((T·S)²). 3D RoPE (separate frequencies for t/h/w)
gives the model position awareness without learned embeddings that
hard-code one resolution.

## Stage 3 — large-scale training challenges (and the code's answers)

| Challenge | Mitigation in this repo |
|---|---|
| Activation memory explodes with T | `gradient_checkpointing: true` + `activation_offload` (CPU offload) |
| Effective batch too small per GPU | `gradient_accumulation_steps: 8` |
| VAE re-encoding waste | **latent caching** — encode once (`data/latent_cache.py`), reuse every epoch |
| Variable geometry batches | resolution/length bucketing at data prep |
| Checkpoint size (DiT-B ≈ 1.3 GB) | sharded safetensors + atomic writes (`utils/checkpoint.py`) |
| Node loss mid-run | resumable step + optimizer state in every checkpoint |

## Data preparation

Standardization before anything: FPS normalization to 8, resolution
bucketing 256–1024, aspect crop policy. Then quality gates: motion score
≥ 0.15 (drops slideshows and static crops), CLIP-text alignment on
captions, dedup. **Latent caching runs after** standardization so cache
keys are deterministic per dataset version.

## Serving integration

Video jobs take the async path exclusively (`sync` applies the timeout
to image profiles; video's cost profile makes sync a lie). Workers
chunk-decode and stream encoded frames to the artifact store; the job's
`num_frames` must be a power of two ≥ 4 (temporal VAE constraint,
enforced at the API schema).
