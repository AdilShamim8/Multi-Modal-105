# Experiment Configuration Guide

Configs are plain YAML with three platform conventions:

## 1. Environment interpolation

```yaml
data_root: ${MM_T2I_DATA_ROOT:./data/laion}   # ${VAR} or ${VAR:default}
```

`${VAR}` fails loudly when unset; `${VAR:default}` falls back. The same
config file therefore works across dev/staging/prod without forking.

## 2. Inheritance

```yaml
__base__: ../base.yaml    # deep-merged under this file's values
```

Chains resolve recursively (max depth 8). Overrides win at every level.

## 3. Strict coercion

`mma train --config <file>` coerces the merged payload into the pipeline's
dataclass (`T2IConfig`, `T2VConfig`, `VAEConfig`, `GANConfig`). Unknown
keys are rejected — a typo'd `batch_size:` becomes a startup error, not
three silent epochs at the wrong batch size.

## Layout

| Path | Purpose |
|---|---|
| `base.yaml` | Shared defaults (seeds, precision, logging, optimizer skeleton) |
| `t2i/dit_base.yaml` | Text-to-image, DiT backbone |
| `t2i/unet_base.yaml` | Text-to-image, SD-style U-Net backbone |
| `t2i/sampling.yaml` | Inference-time sampling profiles |
| `t2v/dit_video_base.yaml` | Text-to-video, factorized space-time DiT |
| `vae/vae_base.yaml` | VAE compression-stage pretraining |
| `gan/stylegan2_base.yaml` | StyleGAN2 adversarial track |
| `eval/default.yaml` | Evaluation suite + release gates |
| `serving/production.yaml` | Serving-plane deployment profile |

## Validation loop

```bash
python - <<'PY'
from multimodal.config.loader import load_config, to_dataclass
from multimodal.config.t2i_config import T2IConfig
cfg = to_dataclass(load_config("configs/t2i/dit_base.yaml"), T2IConfig)
print(cfg.batch_size, cfg.dit_depth)
PY
```

Last updated: 2026-10-18.
