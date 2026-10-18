# Data Preparation — Text-to-Video

> Last updated: 2026-10-18 · Code: `src/multimodal/data/video_dataset.py`, `data/latent_cache.py`

## Pipeline

```
raw clips → standardization → quality gates → dedup → latent cache
```

**Standardization comes FIRST** (unlike T2I) because every downstream
stage depends on normalized geometry:

| Stage | Operation | Why here |
|---|---|---|
| FPS normalization | resample to 8 fps | motion magnitude becomes comparable across sources |
| Resolution bucketing | 256–1024, aspect-preserving | batches of mixed geometry waste GPU memory |
| Length bucketing | 16/32/64 frames (power of two) | temporal VAE constraint; enables batch packing |

## Quality gates

* **Motion score ≥ 0.15** (`min_motion_score`) — drops slideshows and
  static camera crops masquerading as video. Cheap optical-flow proxy;
  the single most valuable video filter.
* **CLIP text-video alignment** — captions must describe the clip, not
  the video they came from.
* **Scene-cut splitting** — hard cuts inside a "clip" produce latents the
  causal VAE cannot compress coherently.
* Perceptual dedup across (first, middle, last) frame hashes.

## Latent caching (the large-scale unlock)

Video VAE encoding is ~40% of naive step time. Caching regime
(`configs/t2v/dit_video_base.yaml`):

1. Encode each standardized clip once → `(C, T/4, H/8, W/8)` latent.
2. Store sharded safetensors, keyed by (dataset version, VAE bundle sha).
3. Training batches read latents directly — the compression network is
   never in the training loop again.
4. Integrity: every shard carries a checksum; corrupt shards are re-encoded
   lazily rather than failing the epoch.

## Storage reality check

16-frame 256px clips at fp16 latents ≈ 40 KB each — a 10M-clip dataset
caches in ~400 GB. That's an object-store line item, not a engineering
problem; budget it before promising a dataset size.

## What NOT to do

1. **Don't cache before standardization** — FPS or bucketing changes
   invalidate every cached latent (the cache keys catch this; plan the
   re-encode cost).
2. **Don't train mixed-fps.** The model learns shutter-speed artifacts
   indistinguishable from style.
3. **Don't skip motion filtering.** Static clips are free negatives that
   teach the model "video = one frame repeated".
