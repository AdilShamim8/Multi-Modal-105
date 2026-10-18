# Data Preparation — Text-to-Image

> Last updated: 2026-10-18 · Code: `src/multimodal/data/` · Filters: `configs/t2i/dit_base.yaml`

## Pipeline (in order — order matters)

```
raw pairs → resolution filter → caption filter → aesthetic filter
        → dedup → bucket → (freeze) → latent cache → training shards
```

Each stage writes a manifest (row counts + parameters) — dataset versions
are reproducible, which is what makes FID comparisons between runs *mean
something*.

## Stage details

| Stage | Threshold (config) | Rationale |
|---|---|---|
| Resolution floor | shortest side ≥ 512 | upsampling garbage trains the model to generate blur |
| Caption length | ≥ 5 chars | drops empty/junk alt-text |
| CLIP aesthetic | ≥ 4.5 | quality prior; the single highest-leverage filter for FID |
| Perceptual dedup | pHash hamming ≤ 4 | near-duplicates inflate epochs and collapse diversity |
| Aspect bucketing | 1:1, 3:4, 4:3, 16:9 | crops stay honest; SDXL-style mixed-bucket training |

## Latent caching

After curation (and VAE freeze), encode once into sharded safetensors:

```bash
mma train --pipeline t2i --config configs/t2i/dit_base.yaml   # cache builds on first epoch
```

Cache keys are deterministic per (dataset version, VAE bundle sha) — a
changed filter or VAE invalidates the cache automatically instead of
silently training on stale latents.

## What NOT to do (learned the expensive way)

1. **No unfiltered web pulls.** The filters exist because their absence
   produces models that reproduce watermarks, borders, and text artifacts.
2. **No mid-training curation changes.** A dataset that shifts under the
   model looks like a training instability and wastes days of diagnosis.
3. **No dedup by URL only.** The same image under ten URLs is still one
   image; perceptual hashing catches what URL sets miss.
4. **Don't skip the manifest.** "Which data trained this checkpoint?" is
   the first question of every post-incident review.
