# Evaluation Guide

> Last audited: 2026-10-18 · Code: `src/multimodal/evaluation/` · Config: `configs/eval/default.yaml`

## The metric suite (and what each one actually measures)

| Metric | File | Measures | Fools you when |
|---|---|---|---|
| **IS** (Inception Score) | `inception_score.py` | class-coverage × confidence of a classifier | mode-dropped but confident samples; sensitive to classifier priors |
| **FID** | `fid.py` | distributional distance to reference (Frechet on inception feats) | small samples (<5k), different preprocessing than reference |
| **CLIP score** | `clip_score.py` | image-text alignment | adversarial/patterned images that CLIP "likes" |
| Diversity (LPIPS) | `evaluator.py` | pairwise perceptual spread | computed on a curated subset |

**The rule: no single number decides anything.** FID falls while CLIP
score rises and diversity collapses = classic mode collapse wearing a
promotion costume. The evaluator computes the suite together and the
release gate reads them together.

## Running an evaluation

```bash
mma evaluate --config configs/eval/default.yaml \
    --samples-dir outputs/t2i_dit_base/samples \
    --reference-dir ./data/eval_reference \
    --metrics is fid clip
```

Defaults: 10k samples (FID ±0.3 noise floor), fixed seed 42, batch 16.
**Sample count and seed are part of the comparison** — a FID from 2k
samples is not comparable to one from 10k, and the report records both.

## Release gates

`configs/eval/default.yaml` thresholds — FID ≤ 25, IS ≥ 8, CLIP ≥ 0.25 —
are the promotion contract wired into the release workflow. A regression
cannot reach the model registry as a "training tweak"; it fails the gate
with the report attached.

## Benchmarking (performance, not quality)

```bash
python scripts/benchmark_serving.py --steps 30 --batch 8 --runs 20
```

Reports jobs/s/GPU and wall-time percentiles — the numbers the capacity
model (`docs/architecture/capacity-planning.md`) is built from. Re-run
after any engine/batching change; the doc's fleet math is only as good
as its last benchmark.

## Evaluation hygiene

1. Reference sets are **versioned artifacts**, not directories that
   drift — FID against a moving reference is a random number.
2. Every eval report carries: model bundle sha, sample count, seed,
   sampler profile. Comparisons without these fields are gossip.
3. Evaluate EMA weights — the same weights serving will load.
