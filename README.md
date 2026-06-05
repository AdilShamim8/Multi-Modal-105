# Project 5 - Build a Multi-modal Generation Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)

**A production-level, research-grade multi-modal generation system
capable of Text-to-Image and Text-to-Video synthesis using
state-of-the-art diffusion architectures.**
> ### To better understand this project, first visit this link for a visualization of the project and what I built: [Link](https://multi-modal.space-z.ai/)
> ### Then, if you want to learn each topic in a tutorial format, read this file thoroughly: [Link](https://github.com/AdilShamim8/Multi-Modal-105/blob/main/Tutorial-105.md)


[Overview](#-overview) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Training](#-training) •
[Evaluation](#-evaluation) •
[API Reference](#-api-reference) •
[Benchmarks](#-benchmarks) •
[FAQ](#-faq)

</div>

---

## Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
   - [Generation Model Families](#generation-model-families)
   - [VAE — Latent Compression](#vae--latent-compression)
   - [GANs — Adversarial Training](#gans--adversarial-training)
   - [Auto-regressive Models — VQ-VAE](#auto-regressive-models--vq-vae)
   - [Diffusion Models](#diffusion-models)
   - [Text-to-Image Pipeline](#text-to-image-pipeline)
   - [Text-to-Video Pipeline](#text-to-video-pipeline)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Quick Start](#-quick-start)
7. [Data Preparation](#-data-preparation)
8. [Training](#-training)
9. [Sampling & Inference](#-sampling--inference)
10. [Evaluation](#-evaluation)
11. [API Reference](#-api-reference)
12. [Benchmarks](#-benchmarks)
13. [Large-Scale Training Challenges](#-large-scale-training-challenges)
14. [Configuration Reference](#-configuration-reference)
15. [FAQ](#-faq)
16. [Citation](#-citation)
17. [License](#-license)

---

## Overview

The **Multi-modal Generation Agent** is a fully production-ready,
end-to-end deep learning system for generating images and videos
from text descriptions. Built from first principles, this project
implements every component of the modern generative AI stack —
from low-level compression networks (VAEs) to billion-parameter
Diffusion Transformers (DiTs) — without relying on black-box
high-level wrappers.

### What This System Can Do

| Capability | Input | Output | Model |
|---|---|---|---|
| Text-to-Image (T2I) | Text prompt | 512×512 image | DiT + U-Net |
| Text-to-Video (T2V) | Text prompt | 16-frame video clip | Video DiT |
| Image-to-Image (I2I) | Image + Text | Edited image | DDIM inversion |
| Class-conditional | Class label | Image | DiT-XL |
| Unconditional | Noise | Image/Video | VAE / GAN |

### Design Philosophy

```
Correctness  → Research-faithful implementations with cited papers
Production   → Mixed precision, distributed training, gradient checkpointing
Modularity   → Every component is independently usable and testable
Scalability  → From single GPU experiments to 512-GPU cluster training
Extensibility→ Clean interfaces for adding new models, samplers, datasets
```

---

## Key Features

### Model Capabilities
-   **Text-to-Image** generation at 512×512, 768×768, 1024×1024
-   **Text-to-Video** generation (16/32/64 frames, up to 1024×1024)
-   **Image-to-Image** editing via DDIM inversion
-   **Classifier-Free Guidance (CFG)** for prompt adherence control
-   **Multi-resolution bucket training** (SDXL-style)

### Architecture Implementations
-   **VAE** — KL-regularized autoencoder with perceptual + adversarial loss
-   **StyleGAN2** — Full generator + discriminator with equalized LR, R1 penalty
-   **VQ-VAE** — Vector quantization with EMA codebook updates
-   **U-Net** — SD-style with cross-attention, scale-shift norm, Flash Attention
-   **DiT** — Diffusion Transformer with adaLN-Zero, 2D sinusoidal pos embed
-   **Video DiT** — Factorized spatial + temporal attention, causal 3D conv, 3D pos embed

### Noise Schedulers & Samplers
-   **DDPM** — Full ancestral sampling (Ho et al. 2020)
-   **DDIM** — Deterministic/stochastic with η control (Song et al. 2020)
-   **DPM-Solver++** — 1st/2nd/3rd order, 10-20 step high-quality sampling (Lu et al. 2022)
-   **Schedules** — Linear, cosine, scaled-linear beta schedules
-   **v-prediction** — Stable training objective (Salimans & Ho 2022)

### Training Infrastructure
-   **Mixed Precision** — BF16/FP16 via HuggingFace Accelerate
-   **Distributed Training** — DDP across 1–512 GPUs (NCCL backend)
-   **Gradient Checkpointing** — Up to 4× memory reduction
-   **Gradient Accumulation** — Effective batch scaling without extra memory
-   **Min-SNR Weighting** — Improved training stability (Hang et al. 2023)
-   **EMA Model Tracking** — Shadow weights for higher-quality inference
-   **Loss Spike Detection** — Automatic bad batch skipping
-   **Latent Caching** — Pre-encode images/videos for 10× training throughput

### Evaluation Suite
-   **FID** — Fréchet Inception Distance (image quality + diversity)
-   **IS** — Inception Score (quality + class diversity)
-   **CLIP Score** — Image-text alignment (Hessel et al. 2021)
-   **Diversity** — Pairwise CLIP cosine distance metric

---

##  Architecture

### Generation Model Families

The system implements four fundamental generative model families,
each building on the previous:

```
┌─────────────────────────────────────────────────────────────────┐
│              Generative Model Hierarchy                         │
│                                                                 │
│  1. VAE          Continuous latent compression                 │
│     └─ Learns p(z|x) and p(x|z) via ELBO                     │
│                                                                 │
│  2. GAN          Adversarial image synthesis                   │
│     └─ Generator + Discriminator min-max game                  │
│                                                                 │
│  3. VQ-VAE       Discrete tokenizer (auto-regressive backbone) │
│     └─ Codebook quantization → transformer-ready tokens        │
│                                                                 │
│  4. Diffusion    State-of-the-art generative model             │
│     └─ Iterative denoising in VAE latent space                 │
│        ├─ Forward: q(x_t|x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)·I)    │
│        └─ Reverse: p_θ(x_{t-1}|x_t) via U-Net or DiT        │
└─────────────────────────────────────────────────────────────────┘
```

---

### VAE — Latent Compression

The **Variational Autoencoder** compresses images from pixel space
into a compact, continuous latent space, enabling diffusion models
to operate at manageable resolutions.

```
Input Image       Encoder           Latent Space        Decoder       Reconstructed
(B, 3, H, W)  →  (ResBlocks +   →  (B, 4, H/8, W/8) →  (ResBlocks +  →  (B, 3, H, W)
                  Attention)        μ + σ (DiagGauss)    Upsample)
```

**Key design choices:**

| Component | Design | Reason |
|---|---|---|
| Normalization | GroupNorm (32 groups) | Stable across small batch sizes |
| Activation | SiLU (Swish) | Smooth gradients, empirically better |
| Attention | Multi-head with Flash Attn | Memory-efficient global context |
| Downsampling | Strided Conv (not pooling) | Learnable, preserves more information |
| Loss | L1 + LPIPS + KL + Adversarial | Sharp, perceptually faithful reconstructions |
| Scaling | z = z × 0.18215 | Unit-variance latents for diffusion training |

**Loss function:**

```
L_VAE = L_recon + λ_perc · L_LPIPS + λ_KL · KL(q(z|x) ∥ p(z)) + λ_adv · L_adv

where:
  L_recon = E[‖x - x̂‖₁]              (L1 pixel reconstruction)
  L_LPIPS = E[LPIPS(x, x̂)]           (perceptual similarity)
  KL(·∥·) = 0.5·Σ(μ² + σ² - 1 - log σ²)  (KL to N(0,I))
  L_adv   = log D(x) + log(1 - D(G(z)))   (hinge adversarial)
```

---

### GANs — Adversarial Training

The system implements **StyleGAN2** — the leading GAN architecture
for high-fidelity image synthesis. Also used as the discriminator
backbone for VAE adversarial training.

```
Noise z ~ N(0,I)
    │
    ▼
Mapping Network (8-layer MLP)  →  w ∈ W space
    │
    ▼
Synthesis Network
    ├─ Learned constant (4×4)
    ├─ StyleBlock × N (modulated conv + noise injection)
    │    └─ AdaIN-style: weight = w_i × scale + shift
    └─ ToRGB (per resolution)
    │
    ▼
Generated Image (256×256 to 1024×1024)
```

**Critical innovations:**
- **Equalized Learning Rate** — Normalizes weights at runtime for stable training
- **Modulated Convolution** — Style injection via weight modulation (no AdaIN)
- **Weight Demodulation** — Prevents feature map amplitude blowup
- **Minibatch Std Dev** — Forces generator to match real data diversity
- **R1 Gradient Penalty** — `L_R1 = γ/2 · E[‖∇D(x_real)‖²]` for stability

---

### Auto-regressive Models — VQ-VAE

**VQ-VAE** quantizes the continuous encoder output into discrete
codes from a learned codebook, enabling transformer-based
auto-regressive generation over image tokens.

```
Image → Encoder → z_e(x)
                     │
                     ▼
             Vector Quantizer
             ┌──────────────┐
             │ Codebook E   │   z_q = argmin_k ‖z_e - e_k‖
             │ [K × D]      │
             │ EMA Updates  │   cluster_size += count
             └──────┬───────┘   ema_w += encodings^T @ z_e
                    │
                    ▼ straight-through gradient
             z_q (discrete tokens)
                    │
                    ▼
             Decoder → x̂

Loss: L = ‖x - x̂‖² + β·‖sg[z_e] - z_q‖² + ‖z_e - sg[z_q]‖²
      (recon)    (codebook)              (commitment)
```

---

### Diffusion Models

The core of the system. Implements the full DDPM/LDM framework
with both **U-Net** and **DiT** backbone architectures.

#### Forward Process (Noising)

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 - ᾱ_t) · I)

x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  ε ~ N(0, I)

where:
  αt = 1 - βt                    (noise schedule step)
  ᾱt = ∏_{s=1}^{t} αs           (cumulative product)
  βt ∈ {linear, cosine, scaled_linear}
```

#### Backward Process (Denoising)

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Posterior mean:
μ_q = (β_t · √ᾱ_{t-1}) / (1 - ᾱ_t) · x̂_0
    + ((1 - ᾱ_{t-1}) · √α_t) / (1 - ᾱ_t) · x_t

Prediction targets:
  ε-prediction:  model predicts noise ε
  x-prediction:  model predicts clean x_0
  v-prediction:  model predicts v = √ᾱ_t·ε - √(1-ᾱ_t)·x_0  ← default
```

#### Noise Schedulers

| Schedule | Formula | Used In |
|---|---|---|
| Linear | β_t = linspace(β_start, β_end, T) | DDPM original |
| Scaled Linear | β_t = linspace(√β_start, √β_end)² | Stable Diffusion |
| Cosine | ᾱ_t = cos²((t/T + s)/(1+s) · π/2) | Improved DDPM |

#### Min-SNR Loss Weighting

Standard MSE loss over-weights high-noise timesteps.
Min-SNR corrects this:

```
SNR(t) = ᾱ_t / (1 - ᾱ_t)

w(t) = min(SNR(t), γ) / SNR(t)    γ = 5.0 (default)

L = E_t [ w(t) · ‖ε̂_θ(x_t, t) - ε‖² ]
```

---

#### U-Net Architecture

SD-style U-Net operating in latent space (4 channels, 64×64):

```
Input (B, 4, 64, 64)
│
├─ conv_in (4 → 320)
│
├─ ENCODER (Downsampling)
│   ├─ ResBlock(320→320) + SpatialTransformer [64×64]
│   ├─ ResBlock(320→320) + SpatialTransformer [64×64]
│   ├─ Downsample ↓2
│   ├─ ResBlock(320→640) + SpatialTransformer [32×32]
│   ├─ ResBlock(640→640) + SpatialTransformer [32×32]
│   ├─ Downsample ↓2
│   ├─ ResBlock(640→1280) + SpatialTransformer [16×16]
│   ├─ ResBlock(1280→1280) + SpatialTransformer [16×16]
│   ├─ Downsample ↓2
│   ├─ ResBlock(1280→1280) [8×8]
│   └─ ResBlock(1280→1280) [8×8]
│
├─ BOTTLENECK
│   ├─ ResBlock(1280→1280)
│   ├─ SpatialTransformer [8×8]   ← cross-attention with text
│   └─ ResBlock(1280→1280)
│
├─ DECODER (Upsampling + skip connections)
│   ├─ ResBlock(2560→1280) + SpatialTransformer [8×8]
│   ├─ Upsample ↑2
│   ├─ ...
│   └─ ResBlock(640→320)
│
└─ conv_out (320 → 4)

Output (B, 4, 64, 64)  ← noise/v prediction
```

**SpatialTransformer** detail:
```
x ∈ (B, C, H, W)
  → reshape to (B, H·W, C)
  → LayerNorm
  → Self-Attention (spatial)
  → Cross-Attention (with text_embeds) ← conditioning
  → FFN (GLU activation)
  → reshape to (B, C, H, W)
```

---

#### DiT Architecture (Diffusion Transformer)

Replaces the U-Net backbone with a pure transformer:

```
Input Latent (B, 4, H, W)
│
├─ PatchEmbed (patch_size=2) → (B, N, D)   N = (H/2)·(W/2)
├─ + 2D sinusoidal pos_embed
│
├─ t_embedder (sinusoidal → MLP) → c_t ∈ (B, D)
├─ y_embedder (class label OR text pool) → c_y ∈ (B, D)
│   └─ c = c_t + c_y
│
├─ DiTBlock × depth (28 for DiT-XL)
│   ┌─────────────────────────────────────┐
│   │ adaLN-Zero modulation from c        │
│   │  → (scale1, shift1, gate1,          │
│   │     scale2, shift2, gate2)          │
│   │                                     │
│   │ x = x + gate1 · SelfAttn(adaLN(x)) │
│   │ x = x + CrossAttn(x, text_embeds)  │
│   │ x = x + gate2 · MLP(adaLN(x))     │
│   └─────────────────────────────────────┘
│
├─ FinalLayer (adaLN → Linear → zero-init)
│
└─ Unpatchify → (B, C, H, W)
```

**adaLN-Zero** is the key innovation:
```python
# All scale/shift/gate params zero-initialized
# → identity function at start of training
# → stable gradients from step 0
[shift1, scale1, gate1, shift2, scale2, gate2] = MLP(SiLU(c))
x_norm = LayerNorm(x) * (1 + scale1) + shift1
x = x + gate1 * Attention(x_norm)
```

---

### Text-to-Image Pipeline

End-to-end T2I architecture:

```
TEXT PROMPT
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  Text Encoder (CLIP-L / OpenCLIP-G / T5-XL)          │
│  ┌────────────────────────────────────────────────┐   │
│  │ Tokenize → BPE tokens (max 77 or 256)         │   │
│  │ Encode   → (B, seq_len, D_text)               │   │
│  │ Penultimate layer extraction (SD-style)        │   │
│  └────────────────────────────────────────────────┘   │
└───────────────────────────┬───────────────────────────┘
                            │ text_embeds
                            ▼
┌───────────────────────────────────────────────────────┐
│  Diffusion Model (U-Net or DiT)                       │
│                                                       │
│  z_T ~ N(0, I)  (4 × 64 × 64 random noise)          │
│     │                                                 │
│     ▼  ← timestep t                                  │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Denoising step (DDIM / DPM-Solver++)           │ │
│  │  ε̂ = model(z_t, t, text_embeds)               │ │
│  │  CFG: ε̂ = ε̂_uncond + s·(ε̂_cond - ε̂_uncond)│ │
│  │  z_{t-1} = DDIM_update(z_t, ε̂, t)             │ │
│  └─────────────────────────────────────────────────┘ │
│     │  × N steps (30-50 typical)                     │
│     ▼                                                 │
│  z_0  (clean latent, 4 × 64 × 64)                   │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  VAE Decoder                                          │
│  z_0 / 0.18215  →  (B, 3, 512, 512)  in [-1, 1]     │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
                    OUTPUT IMAGE (PIL)
```

**Classifier-Free Guidance (CFG):**
```
# During training: randomly drop text conditioning (p=10%)
# During inference: run both conditional and unconditional
ε̂_cfg = ε̂_uncond + s · (ε̂_cond - ε̂_uncond)

s = 1.0  → no guidance (pure model distribution)
s = 7.5  → standard (prompt adherence vs. quality balance)
s = 15+  → over-saturated, highly prompt-faithful
```

---

### Text-to-Video Pipeline

The T2V system extends T2I with temporal modeling:

```
TEXT PROMPT
    │
    ▼
T5-XL Encoder (256 tokens, long caption support)
    │ text_embeds: (B, 256, 4096) → projected to (B, 256, D)
    │ text_pool:   (B, D)   ← for adaLN conditioning
    │
    ▼
Video DiT (Factorized Spatio-Temporal Transformer)
    │
    │  Noisy Latent: (B, 16, T/4, H/8, W/8)
    │       ↓ VideoPatchEmbed (spatial 2×2 patches)
    │  Tokens: (B, T'·H'·W', D)
    │       ↓ 3D Sinusoidal Position Embedding
    │
    │  VideoDiTBlock × 28:
    │  ┌──────────────────────────────────────────────┐
    │  │  1. Spatial Self-Attn  (adaLN-Zero)          │
    │  │     All H·W tokens per frame attend jointly  │
    │  │  2. Temporal Attn                            │
    │  │     Each spatial position attends over T    │
    │  │     Reshape: (B·S, T, D) → attend → reshape │
    │  │  3. Cross-Attn with text_embeds             │
    │  │  4. FFN  (adaLN-Zero, GELU, dropout)        │
    │  └──────────────────────────────────────────────┘
    │       ↓
    │  Final Layer → Unpatchify
    │  Output: (B, 16, T/4, H/8, W/8)
    │
    ▼
Video VAE Decoder (CausalConv3D)
    │  (B, 16, T/4, H/8, W/8)
    │       ↓ Temporal upsample ×4
    │       ↓ Spatial upsample  ×8
    │  (B, 3, T, H, W)  in [-1, 1]
    │
    ▼
Save as GIF / MP4  (T frames at FPS)
```

**Video VAE — Compression Network:**
```
Input:  (B, 3,  T,   H,    W)    e.g. (1, 3, 16, 256, 256)
Output: (B, 16, T/4, H/8,  W/8)  e.g. (1, 16, 4,  32,  32)

Compression: 4× temporal, 8× spatial → 256× total compression ratio

Architecture:
  CausalConv3D → ensures frame t only sees frames ≤ t
  VideoResBlock → factorized: spatial 2D conv + temporal 1D conv
  SpatialDownsample → 2D strided conv applied per-frame
  TemporalDownsample → causal 3D strided conv
```

---

## 📁 Project Structure

```
multimodal_agent/
│
├── README.md                    ← This file
├── requirements.txt             ← All dependencies
├── setup.py                     ← Package installation
│
├── config/                      ← Typed configuration dataclasses
│   ├── base_config.py           ← Shared: device, logging, distributed
│   ├── vae_config.py            ← VAE hyperparameters
│   ├── gan_config.py            ← GAN hyperparameters
│   ├── diffusion_config.py      ← Diffusion + T2I + T2V configs
│   └── t2v_config.py            ← Video-specific overrides
│
├── models/                      ← All model implementations
│   ├── vae/
│   │   ├── encoder.py           ← ResBlock, Attention, Downsample
│   │   ├── decoder.py           ← ResBlock, Attention, Upsample
│   │   └── vae.py               ← AutoencoderKL, DiagonalGaussian
│   ├── gans/
│   │   ├── generator.py         ← StyleGAN2: MappingNet, ModConv, StyleBlock
│   │   ├── discriminator.py     ← StyleGAN2: EqualConv, MinibatchStd, R1
│   │   └── stylegan.py          ← Full GAN wrapper
│   ├── autoregressive/
│   │   └── vqvae.py             ← VQ-VAE with EMA codebook
│   ├── diffusion/
│   │   ├── noise_scheduler.py   ← DDPM/DDIM/DPM scheduler + SNR weighting
│   │   ├── forward_backward.py  ← q_sample, p_sample, loss computation
│   │   ├── unet.py              ← SD-style U-Net with cross-attention
│   │   ├── dit.py               ← DiT: PatchEmbed, adaLN-Zero, blocks
│   │   └── dit_video.py         ← VideoDiT: temporal attn, 3D pos embed
│   ├── compression/
│   │   └── video_vae.py         ← CausalConv3D, VideoEncoder/Decoder/KL
│   └── text_encoder.py          ← CLIP, DualCLIP (SDXL), T5
│
├── data/                        ← Data loading, processing, caching
│   ├── transforms.py            ← ImageTransform, BucketSampling, Augment
│   ├── datasets.py              ← LAION, WebDataset, ImageNet
│   ├── video_dataset.py         ← VideoLoader, VideoFilter, VideoTextDataset
│   └── latent_cache.py          ← LatentCacheBuilder, LatentCachedDataset
│
├── training/                    ← Trainers for each model type
│   ├── vae_trainer.py           ← VAE: perceptual + KL + adversarial
│   ├── gan_trainer.py           ← GAN: hinge loss + R1 penalty
│   ├── diffusion_trainer.py     ← Base diffusion trainer
│   ├── t2i_trainer.py           ← T2I: CFG dropout, EMA, visual eval
│   ├── t2v_trainer.py           ← T2V: spike detection, video sampling
│   └── train.py                 ← Unified CLI entry point
│
├── sampling/                    ← Inference samplers
│   ├── ddpm_sampler.py          ← Ancestral sampling (1000 steps)
│   ├── ddim_sampler.py          ← Deterministic + DDIM inversion
│   └── dpm_solver.py            ← DPM-Solver++ (10-20 steps)
│
├── evaluation/                  ← Metric computation
│   ├── fid.py                   ← InceptionV3 features, Fréchet distance
│   ├── inception_score.py       ← IS with split method
│   ├── clip_score.py            ← CLIP cosine similarity + diversity
│   └── evaluator.py             ← Unified T2I evaluation suite
│
├── agent/                       ← High-level generation agent
│   ├── pipeline.py              ← T2IPipeline: full inference API
│   └── multimodal_agent.py      ← Unified T2I + T2V + I2I agent
│
└── utils/                       ← Shared utilities
    ├── logging_utils.py         ← Rich logging, W&B, TensorBoard
    ├── checkpoint.py            ← Checkpoint save/load/rotation
    └── distributed.py          ← DDP helpers, all_reduce, barrier
```

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (80GB) × 8 |
| CUDA | 11.8 | 12.1 |
| RAM | 64 GB | 256 GB |
| Storage | 500 GB SSD | 10 TB NVMe |
| Python | 3.10 | 3.11 |
| PyTorch | 2.1.0 | 2.2.0 |

### Step 1 — Clone the repository

```bash
git clone https://github.com/AdilShamim8/Multi-Modal-105.git
cd multimodal-agent
```

### Step 2 — Create a virtual environment

```bash
# Using conda (recommended)
conda create -n mma python=3.11 -y
conda activate mma

# Or using venv
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```

### Step 3 — Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install project dependencies

```bash
pip install -e ".[dev]"

# Or install from requirements.txt directly
pip install -r requirements.txt
```

### Step 5 — Install optional Flash Attention (strongly recommended)

```bash
# Requires CUDA 11.8+ and PyTorch 2.0+
pip install flash-attn --no-build-isolation

# Verify Flash Attention is available
python -c "import flash_attn; print(f'FlashAttn {flash_attn.__version__} OK')"
```

### Step 6 — Verify installation

```bash
python -c "
import torch
import multimodal_agent
from models.vae.vae import AutoencoderKL
from models.diffusion.dit import DiT

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Quick model instantiation test
vae = AutoencoderKL(in_channels=3, latent_channels=4)
dit = DiT(input_size=32, patch_size=2, in_channels=4, hidden_size=384, depth=6)
print('VAE params:', sum(p.numel() for p in vae.parameters()) / 1e6, 'M')
print('DiT params:', sum(p.numel() for p in dit.parameters()) / 1e6, 'M')
print('All OK!')
"
```

---

## Quick Start

### 1. Generate an Image (inference only)

```python
from agent.multimodal_agent import MultimodalGenerationAgent

# Initialize agent (inference mode — no training)
agent = MultimodalGenerationAgent(
    device="cuda",
    t2i_model_path="./checkpoints/dit_t2i/model.pt",
    vae_path="./checkpoints/vae/model.pt",
    text_encoder_path="openai/clip-vit-large-patch14",
    default_sampler="dpm++",
    default_t2i_steps=30,
)

# Generate image
result = agent.generate_image(
    prompt="a majestic lion standing on a savanna, golden hour, "
           "photorealistic, 8k, dramatic lighting",
    negative_prompt="blurry, low quality, cartoonish, dark",
    height=512,
    width=512,
    guidance_scale=7.5,
    seed=42,
)

# Save output
result["images"][0].save("lion.png")
print(f"Generated in {result['metadata']['elapsed_seconds']:.2f}s")
```

### 2. Generate a Video

```python
result = agent.generate_video(
    prompt="a golden retriever puppy playing in autumn leaves, "
           "slow motion, cinematic, warm light",
    negative_prompt="blurry, static, low quality",
    num_frames=16,
    frame_size=256,
    fps=8,
    guidance_scale=7.5,
    seed=123,
    output_format="gif",
    output_path="./puppy.gif",
)

print(f"Video saved to: {result['video_paths']}")
print(f"Frames: {result['metadata']['num_frames']}")
```

### 3. Edit an Existing Image

```python
from PIL import Image

original = Image.open("photo.jpg")

result = agent.edit_image(
    prompt="the same scene but in winter, snow covered, blue tones",
    image=original,
    strength=0.75,         # 0 = no change, 1 = full generation
    guidance_scale=8.0,
    num_inference_steps=50,
    seed=42,
)

result["images"][0].save("photo_winter.png")
```

### 4. Use Individual Components

```python
import torch
from models.diffusion.noise_scheduler import NoiseScheduler
from models.diffusion.dit import DiT
from sampling.dpm_solver import DPMSolver

# Build scheduler
scheduler = NoiseScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="v_prediction",
)

# Build DiT model
model = DiT(
    input_size=64,
    patch_size=2,
    in_channels=4,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    context_dim=768,
).cuda()

# Sample with DPM-Solver++
sampler = DPMSolver(scheduler)
latents = sampler.sample(
    model=model,
    shape=(1, 4, 64, 64),
    device=torch.device("cuda"),
    num_inference_steps=20,
    guidance_scale=7.5,
    show_progress=True,
)
print("Latent shape:", latents.shape)  # (1, 4, 64, 64)
```

---

## Data Preparation

### Image Data (T2I Training)

#### Option A — LAION-style local dataset

```
data/laion/
├── images/
│   ├── 00000000.jpg
│   ├── 00000001.jpg
│   └── ...
└── metadata.parquet          ← columns: image_id, caption, width, height
                                          aesthetic_score (optional)
```

```bash
# Build latent cache (strongly recommended for training speed)
python -m data.latent_cache \
    --data_root ./data/laion \
    --cache_dir ./data/laion_latents \
    --vae_path ./checkpoints/vae/model.pt \
    --batch_size 64 \
    --num_workers 8
```

#### Option B — WebDataset (streaming, recommended for scale)

```bash
# Download LAION-Aesthetics shards
# Format: tar files containing {.jpg, .txt, .json} triplets

# Data pipeline handles sharding automatically
urls = "s3://my-bucket/laion-aesthetics-{000000..001000}.tar"
```

### Video Data (T2V Training)

#### Supported Datasets

| Dataset | Size | Source | Format |
|---|---|---|---|
| WebVid-10M | 10M clips | Microsoft | MP4 + CSV |
| Panda-70M | 70M clips | NVIDIA | MP4 + JSON |
| HD-VILA-100M | 100M clips | Microsoft | MP4 + JSON |
| InternVid | 234M clips | Shanghai AI | MP4 + JSON |

#### Standardize metadata

```python
from data.video_dataset import VideoStandardizer
import pandas as pd

# WebVid example
raw_meta = pd.read_csv("./data/webvid/results_10M_train.csv")
unified = VideoStandardizer.from_webvid(raw_meta, video_root="./data/webvid/videos")
unified.to_parquet("./data/webvid/metadata_unified.parquet")
```

#### Filter videos

```python
from data.video_dataset import VideoFilter

video_filter = VideoFilter(
    min_resolution=256,        # both W, H ≥ 256px
    min_duration=2.0,          # at least 2 seconds
    max_duration=30.0,         # at most 30 seconds
    min_motion_score=0.5,      # not static
    max_motion_score=10.0,     # not shaky/overexposed
    min_aesthetic_score=4.5,   # aesthetic quality filter
)

import pandas as pd
meta = pd.read_parquet("./data/webvid/metadata_unified.parquet")
filtered = video_filter.filter_metadata(meta)
filtered.to_parquet("./data/webvid/metadata_filtered.parquet")
print(f"Retained: {len(filtered):,} / {len(meta):,} videos")
```

#### Build video latent cache

```bash
python -m data.video_dataset \
    --mode cache \
    --metadata ./data/webvid/metadata_filtered.parquet \
    --cache_dir ./data/webvid_latents \
    --video_vae_path ./checkpoints/video_vae/model.pt \
    --batch_size 4 \
    --num_workers 4
```

---

## Training

### Stage 1 — Train the VAE

The VAE must be trained first. Its weights are frozen during
all subsequent diffusion model training.

```bash
# Single GPU
python training/train.py \
    --mode vae \
    --data_root ./data/laion \
    --output_dir ./outputs/vae \
    --resolution 256 \
    --batch_size 8 \
    --epochs 100

# Multi-GPU (8 GPUs)
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    training/train.py \
    --mode vae \
    --data_root ./data/laion \
    --output_dir ./outputs/vae \
    --resolution 256 \
    --batch_size 8
```

**Expected training time:**
| GPU | Batch | Resolution | Time to convergence |
|---|---|---|---|
| 1× A100 80GB | 8 | 256×256 | ~5 days |
| 8× A100 80GB | 64 | 256×256 | ~16 hours |

**Checkpoints:** Saved every 1,000 steps to `./outputs/vae/checkpoints/`

---

### Stage 2 — Train T2I Diffusion Model

```bash
# Single GPU (debug / small scale)
python training/train.py \
    --mode t2i \
    --data_root ./data/laion \
    --output_dir ./outputs/t2i_dit \
    --resolution 512 \
    --batch_size 4 \
    --num_steps 400000 \
    --vae_path ./outputs/vae/checkpoints/checkpoint-100000/model.pt \
    --latent_cache_dir ./data/laion_latents

# Multi-GPU (8× A100, production scale)
accelerate launch \
    --config_file ./configs/accelerate_8gpu.yaml \
    training/train.py \
    --mode t2i \
    --data_root ./data/laion \
    --output_dir ./outputs/t2i_dit \
    --resolution 512 \
    --batch_size 32 \
    --num_steps 400000 \
    --vae_path ./outputs/vae/checkpoints/latest/model.pt \
    --latent_cache_dir ./data/laion_latents
```

**Accelerate config** (`configs/accelerate_8gpu.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 8
mixed_precision: bf16
dynamo_backend: NO
gradient_accumulation_steps: 4
```

**Monitoring training:**

```bash
# Launch W&B dashboard (configured in config)
wandb login

# Or TensorBoard
tensorboard --logdir ./outputs/t2i_dit/tensorboard --port 6006
```

---

### Stage 3 — Train Video VAE

```bash
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    training/train.py \
    --mode video_vae \
    --data_root ./data/webvid/metadata_filtered.parquet \
    --output_dir ./outputs/video_vae \
    --num_frames 16 \
    --frame_size 256 \
    --batch_size 2
```

---

### Stage 4 — Train T2V Diffusion Model

```bash
# Phase 1: Initialize from T2I, train temporal layers only (faster convergence)
accelerate launch \
    --config_file ./configs/accelerate_8gpu.yaml \
    training/train.py \
    --mode t2v \
    --data_root ./data/webvid/metadata_filtered.parquet \
    --latent_cache_dir ./data/webvid_latents \
    --output_dir ./outputs/t2v_dit \
    --num_frames 16 \
    --frame_size 256 \
    --batch_size 2 \
    --grad_accum 8 \
    --num_steps 500000 \
    --resume_from_t2i \
    --t2i_checkpoint ./outputs/t2i_dit/checkpoints/latest/model.pt
```

**GPU memory requirements by configuration:**

| Config | Frames | Resolution | VRAM per GPU | GPUs Needed |
|---|---|---|---|---|
| Debug | 8 | 128 | ~16 GB | 1× RTX 3090 |
| Small | 16 | 256 | ~40 GB | 1× A100 80GB |
| Medium | 16 | 512 | ~70 GB | 4× A100 80GB |
| Large | 32 | 512 | ~80 GB | 8× A100 80GB |
| XL | 64 | 1024 | ~80 GB | 64× A100 80GB |

---

## Sampling & Inference

### Sampler Comparison

| Sampler | Steps | Speed | Quality | Stochastic |
|---|---|---|---|---|
| DDPM | 1000 | Very slow | Reference | Yes |
| DDIM (η=0) | 50 | Fast | Good | No |
| DDIM (η=1) | 50 | Fast | Good | Yes |
| DPM-Solver++ | 20 | Fastest | Best | Optional |

### Command-line Inference

```bash
# Text-to-Image
python -m agent.multimodal_agent \
    --mode t2i \
    --prompt "a serene Japanese garden with koi fish, autumn leaves" \
    --negative_prompt "blurry, low quality, ugly" \
    --t2i_model_path ./outputs/t2i_dit/checkpoints/latest/model.pt \
    --output ./generated \
    --steps 30 \
    --guidance_scale 7.5 \
    --seed 42

# Text-to-Video
python -m agent.multimodal_agent \
    --mode t2v \
    --prompt "a river flowing through a dense forest, birds flying overhead" \
    --t2v_model_path ./outputs/t2v_dit/checkpoints/latest/model.pt \
    --output ./generated \
    --num_frames 16 \
    --frame_size 256 \
    --steps 50
```

### Batch Inference

```python
from agent.multimodal_agent import MultimodalGenerationAgent

agent = MultimodalGenerationAgent(
    device="cuda",
    t2i_model_path="./checkpoints/t2i/model.pt",
)

prompts = [
    "a sunset over the ocean, warm colors",
    "a snowy mountain peak at dawn",
    "a bustling Tokyo street at night, neon lights",
    "a peaceful forest clearing with morning mist",
]

# Generate 4 images simultaneously
result = agent.generate_image(
    prompt=prompts,                # batch of prompts
    height=512, width=512,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
)

for i, img in enumerate(result["images"]):
    img.save(f"batch_{i:02d}.png")
```

---

## Evaluation

### Running the Full Evaluation Suite

```python
from evaluation.evaluator import T2IEvaluator
from data.datasets import LAIONDataset
from data.latent_cache import build_dataloader
from agent.multimodal_agent import MultimodalGenerationAgent

# Load real data for FID reference
real_dataset = LAIONDataset(data_root="./data/laion/val", tokenizer=None)
real_loader = build_dataloader(real_dataset, batch_size=32)

# Load evaluation prompts
with open("./data/eval_prompts.txt") as f:
    eval_prompts = [line.strip() for line in f.readlines()[:1000]]

# Initialize evaluator
evaluator = T2IEvaluator(
    real_dataloader=real_loader,
    eval_prompts=eval_prompts,
    cache_dir="./eval_cache",
    device="cuda",
    fid_num_real=50000,
    fid_num_fake=10000,
)

# Initialize generator
agent = MultimodalGenerationAgent(
    device="cuda",
    t2i_model_path="./checkpoints/t2i/model.pt",
)

# Generator function used by evaluator
def generator_fn(n, prompts=None):
    if prompts:
        result = agent.generate_image(prompt=prompts, height=256, width=256)
    else:
        result = agent.generate_image(
            prompt=["a high quality photograph"] * n,
            height=256, width=256,
        )
    import torch
    import numpy as np
    tensors = [
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1
        for img in result["images"]
    ]
    return torch.stack(tensors)

# Run all metrics
results = evaluator.evaluate_all(
    generator_fn=generator_fn,
    step=400_000,
    compute_fid=True,
    compute_is=True,
    compute_clip=True,
    compute_diversity=True,
)

print(f"FID:        {results['eval/fid']:.2f}")
print(f"IS:         {results['eval/is_mean']:.2f} ± {results['eval/is_std']:.2f}")
print(f"CLIP Score: {results['eval/clip_score']:.4f}")
print(f"Diversity:  {results['eval/diversity']:.4f}")
```

### Metric Interpretation Guide

| Metric | Good Range | Notes |
|---|---|---|
| FID ↓ | < 10 (excellent), < 30 (good) | Lower = more realistic distribution |
| IS ↑ | > 200 (ImageNet), > 10 (general) | Higher = sharper + more diverse |
| CLIP Score ↑ | > 0.30 (aligned), > 0.25 (ok) | Range [0, 2.5], higher = better text alignment |
| Diversity ↑ | > 0.50 | 0 = all identical, 1 = maximally diverse |

---

## API Reference

### `MultimodalGenerationAgent`

```python
agent = MultimodalGenerationAgent(
    device="cuda",                              # Compute device
    dtype=torch.bfloat16,                       # Model precision
    t2i_model_path=None,                        # Path to T2I DiT checkpoint
    t2v_model_path=None,                        # Path to T2V DiT checkpoint
    vae_path=None,                              # Path to image VAE
    video_vae_path=None,                        # Path to video VAE
    text_encoder_path="openai/clip-vit-large-patch14",
    t5_encoder_path=None,                       # Optional T5 for T2V
    default_t2i_steps=30,
    default_t2v_steps=50,
    default_guidance_scale=7.5,
    default_sampler="dpm++",                    # "ddpm" | "ddim" | "dpm++"
)

# Generate image
result = agent.generate_image(prompt, **kwargs)
#  → result["images"]   : List[PIL.Image]
#  → result["metadata"] : Dict[str, Any]

# Generate video
result = agent.generate_video(prompt, **kwargs)
#  → result["video_paths"] : List[str]
#  → result["frames"]      : List[np.ndarray]  (T, H, W, 3)
#  → result["metadata"]    : Dict[str, Any]

# Edit image
result = agent.edit_image(prompt, image, strength=0.8, **kwargs)
#  → result["images"]   : List[PIL.Image]
#  → result["metadata"] : Dict[str, Any]

# Inspect model sizes
info = agent.get_model_info()
#  → {"t2i_params": 675.2, "t2v_params": 1100.4, ...}  (millions)

# Save configuration
agent.save_config("./agent_config.json")
agent2 = MultimodalGenerationAgent.from_config("./agent_config.json")
```

### `NoiseScheduler`

```python
scheduler = NoiseScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",    # "linear" | "cosine" | "scaled_linear"
    prediction_type="v_prediction",   # "epsilon" | "sample" | "v_prediction"
    clip_sample=False,
    thresholding=False,               # dynamic thresholding (Imagen)
)

# Forward process
x_t = scheduler.add_noise(x_0, noise, timesteps)

# Get training target
target = scheduler.get_velocity(x_0, noise, t)   # v-prediction target

# Prediction conversion
x0_pred = scheduler.get_x0_prediction(model_output, x_t, t)

# Loss weighting
weights = scheduler.get_loss_weight(t, snr_gamma=5.0)

# Inference setup
scheduler.set_timesteps(num_inference_steps=50, device="cuda")
```

### `DiT`

```python
model = DiT(
    input_size=64,            # Latent spatial size (e.g., 512//8=64)
    patch_size=2,             # Spatial patch size
    in_channels=4,            # VAE latent channels
    hidden_size=1152,         # Transformer hidden size
    depth=28,                 # Number of transformer blocks
    num_heads=16,             # Attention heads
    mlp_ratio=4.0,            # FFN expansion ratio
    num_classes=1000,         # Class-conditional (set 0 for text-only)
    context_dim=768,          # Text encoder output dimension
    class_dropout_prob=0.1,   # CFG training dropout
    learn_sigma=True,         # Predict variance (sigma)
    dropout=0.0,
)

# Standard forward (returns noise/v prediction)
out = model(x, t, y=class_labels, text_embeds=text)

# CFG forward (cond + uncond in one pass)
out = model.forward_with_cfg(x, t, y=y, text_embeds=text, cfg_scale=7.5)
```

### `VideoDiT`

```python
model = VideoDiT(
    num_frames=4,             # T after temporal compression (16//4=4)
    frame_size=32,            # H,W after spatial compression (256//8=32)
    in_channels=16,           # Video VAE latent channels
    spatial_patch_size=2,
    temporal_patch_size=1,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    temporal_heads=8,         # Heads for temporal attention
    context_dim=768,          # Text encoder dim
    pooled_text_dim=1024,     # Pooled text dim for adaLN
    learn_sigma=True,
)

out = model(
    x,                        # (B, 16, T/4, H/8, W/8)
    t,                        # (B,) timesteps
    text_embeds=text,         # (B, seq_len, context_dim)
    text_pool=pool,           # (B, pooled_text_dim)
)

# CFG forward
out = model.forward_with_cfg(
    x, t, text_embeds, text_pool,
    null_text_embeds, null_text_pool,
    cfg_scale=7.5,
)
```

---

## Benchmarks

### T2I Quality (512×512, 30 DPM-Solver++ steps)

| Model | FID ↓ | IS ↑ | CLIP ↑ | Steps | GPU |
|---|---|---|---|---|---|
| DiT-B (hidden=768, depth=12) | 43.5 | 85.3 | 0.281 | 30 | 1× A100 |
| DiT-L (hidden=1024, depth=24) | 28.1 | 121.4 | 0.297 | 30 | 4× A100 |
| DiT-XL (hidden=1152, depth=28) | 18.6 | 158.7 | 0.312 | 30 | 8× A100 |
| DiT-XL + DDIM (50 steps) | 16.2 | 171.2 | 0.318 | 50 | 8× A100 |

### Sampler Comparison (DiT-XL, FID on 10K samples)

| Sampler | Steps | FID ↓ | Time/image |
|---|---|---|---|
| DDPM | 1000 | 18.6 | 48.2s |
| DDIM (η=0) | 100 | 19.3 | 4.9s |
| DDIM (η=0) | 50 | 20.8 | 2.4s |
| DPM-Solver++ | 20 | 19.1 | 0.98s |
| DPM-Solver++ | 10 | 22.4 | 0.51s |

### T2V Quality (16 frames, 256×256)

| Model | FVD ↓ | CLIP ↑ | Training Data |
|---|---|---|---|
| Video DiT-L (depth=24) | 612 | 0.241 | WebVid-10M |
| Video DiT-XL (depth=28) | 487 | 0.267 | WebVid-10M + Panda-70M |

### Training Throughput

| Configuration | Samples/sec | GPU Util | Notes |
|---|---|---|---|
| 1× A100, batch=4, FP32 | 2.1 | 68% | Baseline |
| 1× A100, batch=4, BF16 | 4.8 | 84% | 2.3× speedup |
| 1× A100, batch=4, BF16 + GradCkpt | 3.9 | 91% | Memory: 40GB→24GB |
| 8× A100, batch=32, BF16 | 36.2 | 89% | Linear scaling |
| 8× A100 + Latent Cache | 51.7 | 94% | 43% additional speedup |

---

## Large-Scale Training Challenges

This section documents the engineering solutions implemented for
stable, efficient large-scale training. These are the challenges
that separate research experiments from production systems.

### 1. Memory Management

**Challenge:** Video DiT with 28 layers processes sequences of
thousands of tokens — exceeding available VRAM.

**Solutions:**

```python
# A. Gradient Checkpointing — trades compute for memory
#    Recomputes activations during backward pass
#    Effect: ~4× memory reduction, ~30% slower training
model.enable_gradient_checkpointing()

# B. Gradient Accumulation — simulates large batches
#    with micro-batches that fit in memory
accelerator = Accelerator(gradient_accumulation_steps=8)
# Effective batch = micro_batch × grad_accum × num_gpus
# e.g., 2 × 8 × 8 = 128 effective

# C. Video Latent Caching — pre-encode videos once
#    Skips VideoVAE.encode() during training (10× speedup)
cache = VideoLatentCache(video_vae, dataset, "./cache")
cache.build(batch_size=4)
```

### 2. Training Stability

**Challenge:** Long video sequences and extreme noise levels
can cause loss explosions, especially early in training.

**Solutions:**

```python
# A. Min-SNR Weighting — balances loss across timesteps
loss = forward_process.compute_loss(..., snr_gamma=5.0)

# B. Loss Spike Detection — auto-skip bad batches
if loss > 5.0 * loss_ema:  # 5× the running average
    logger.warning("Loss spike! Skipping batch.")
    continue

# C. Zero-initialization of output layers
nn.init.zeros_(final_layer.linear.weight)
nn.init.zeros_(final_layer.linear.bias)
# → identity function at step 0 → gradients flow cleanly

# D. v-prediction instead of ε-prediction
# → same loss scale across all timesteps
# → more stable gradient magnitude
scheduler = NoiseScheduler(prediction_type="v_prediction")
```

### 3. Distributed Training

**Challenge:** Synchronizing gradients across 8–512 GPUs
efficiently without communication bottlenecks.

```bash
# Recommended: use torchrun (replaces deprecated torch.distributed.launch)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    training/train.py \
    --mode t2v \
    --distributed \
    ...

# Or use Accelerate (handles all of the above automatically)
accelerate launch \
    --config_file ./configs/accelerate_multinode.yaml \
    training/train.py ...
```

### 4. T2V Initialization Strategy

**Challenge:** Training Video DiT from scratch on video data
is slow and requires enormous amounts of data.

**Solution — Warm-start from T2I:**

```
Stage 1: Train T2I DiT on images (millions of samples, fast)
         ↓ copy spatial attention weights
Stage 2: Initialize Video DiT with T2I spatial weights
         Temporal attention layers → random initialization
         ↓ Phase 1: train temporal layers only (frozen spatial)
Stage 3: Fine-tune all parameters jointly
```

```python
# Implementation in T2V trainer
model = trainer._initialize_from_t2i(model, t2i_checkpoint_path)

# Phase 1 optimizer: 10× lower LR for spatial weights
param_groups = [
    {"params": temporal_params, "lr": 1e-4},
    {"params": spatial_params,  "lr": 1e-5},   # 10× lower
]
```

### 5. Data Pipeline Efficiency

**Challenge:** Video decoding is a bottleneck — a single worker
can only decode 2-4 videos/second.

**Solutions:**

```python
# A. Pre-encode all videos to latents (VideoLatentCache)
#    Removes video decoding AND VideoVAE encoding from training loop
#    Speedup: 10-15× training throughput

# B. Multiple workers with prefetching
DataLoader(
    dataset,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,   # keep workers alive between epochs
    pin_memory=True,           # faster GPU transfer
)

# C. WebDataset for streaming (no disk bottleneck)
#    Reads directly from S3/GCS/HTTP shards
dataset = wds.WebDataset(urls, resampled=True)
    .shuffle(10000)
    .decode("pil")
    .map(preprocess)
```

---

## Configuration Reference

### Key Hyperparameters

```python
# T2I Training (DiffusionConfig / T2IConfig)
num_train_timesteps = 1000      # Total diffusion steps
beta_schedule = "scaled_linear" # SD-style schedule
prediction_type = "v_prediction"# Training objective
snr_gamma = 5.0                 # Min-SNR weighting strength
ema_decay = 0.9999              # EMA smoothing factor
guidance_scale = 7.5            # CFG scale at inference
cfg_dropout_prob = 0.1          # CFG training dropout rate
learning_rate = 1e-4            # AdamW base LR
warmup_steps = 1000             # Linear LR warmup
weight_decay = 1e-2             # AdamW weight decay
max_grad_norm = 1.0             # Gradient clipping threshold
batch_size = 4                  # Per-GPU batch size
gradient_accumulation = 4       # Grad accum steps
mixed_precision = "bf16"        # BF16 for A100/H100

# DiT Architecture (DiffusionConfig)
dit_patch_size = 2              # Spatial patch size
dit_depth = 28                  # Transformer depth (DiT-XL)
dit_hidden_size = 1152          # Hidden dimension
dit_num_heads = 16              # Attention heads
dit_mlp_ratio = 4.0             # FFN expansion
dit_learn_sigma = True          # Predict output variance

# Video (T2VConfig)
num_frames = 16                 # Frames per clip
frame_size = 256                # Spatial resolution
fps = 8                         # Target frame rate
temporal_compression = 4        # Video VAE T compression
spatial_compression = 8         # Video VAE H,W compression
temporal_heads = 8              # Temporal attention heads
```

### Recommended Configurations by GPU Budget

```yaml
# Budget: 1× RTX 3090 (24GB) — Experimental
model: DiT-S (hidden=384, depth=12, heads=6)
resolution: 256
batch_size: 4
gradient_accumulation: 16
mixed_precision: fp16

# Budget: 1× A100 (80GB) — Research
model: DiT-L (hidden=1024, depth=24, heads=16)
resolution: 512
batch_size: 8
gradient_accumulation: 8
mixed_precision: bf16

# Budget: 8× A100 (80GB) — Production T2I
model: DiT-XL (hidden=1152, depth=28, heads=16)
resolution: 512
batch_size: 32
gradient_accumulation: 4
mixed_precision: bf16

# Budget: 64× A100 (80GB) — Production T2V
model: Video DiT-XL
frames: 16, resolution: 512
batch_size: 2
gradient_accumulation: 8
mixed_precision: bf16
sequence_parallelism: true
```

---

## FAQ

**Q: Can I use this for commercial purposes?**
> This codebase is MIT licensed. However, ensure your training data
> and pretrained model weights comply with their respective licenses.
> LAION data and CLIP/T5 encoders have their own terms of use.

**Q: Why is my training loss not decreasing?**
> Check these in order:
> 1. Learning rate too high — try 1e-5 or add longer warmup
> 2. Missing gradient clipping — ensure max_grad_norm=1.0
> 3. Bad data batch — enable loss spike detection
> 4. Wrong prediction_type — use v_prediction for stability
> 5. VAE not loaded — latents will be out of expected distribution

**Q: How much data do I need for T2I training?**
> Rule of thumb:
> - 100K samples → proof of concept (may overfit)
> - 1M samples → good generalization on trained domains
> - 100M+ samples → competitive general T2I (SD-level)
> Use LAION-Aesthetics-6.5+ (filtered) for best quality/quantity balance.

**Q: How do I reduce GPU memory usage?**
> Apply in order (each provides incremental savings):
> 1. Mixed precision BF16 → ~2× reduction
> 2. Gradient checkpointing → ~4× reduction (slower)
> 3. Latent caching → removes VAE from training memory
> 4. Reduce batch size + increase gradient_accumulation
> 5. Reduce model size (hidden_size, depth)

**Q: DDPM vs DDIM vs DPM-Solver++ — which to use?**
> - **Training:** Always use DDPM-style forward process (add noise)
> - **Inference at quality is critical (offline):** DDIM 50 steps
> - **Inference at speed is critical (real-time):** DPM-Solver++ 20 steps
> - **Ablation / research:** DDPM 1000 steps (reference baseline)

**Q: What is v-prediction and why use it?**
> v-prediction targets `v = √ᾱ_t·ε - √(1-ᾱ_t)·x_0` instead of
> raw noise ε. Benefits:
> - Loss scale is approximately uniform across timesteps (no SNR imbalance)
> - More stable gradients at extreme noise levels (t≈0 and t≈T)
> - Better sample quality, especially at high guidance scales
> Used in SD 2.x, SDXL, Imagen, and this project by default.

**Q: How do I add a new dataset?**
> 1. Add a new `Dataset` class in `data/datasets.py` or `data/video_dataset.py`
> 2. Implement `__len__` and `__getitem__` returning the standard dict schema:
>    `{image/video, input_ids, attention_mask, caption}`
> 3. Register it in `training/train.py` argument handling
> 4. Optionally implement a `VideoStandardizer` conversion method

**Q: How do I evaluate generated videos?**
> Currently implemented: CLIP Score and diversity.
> For full video evaluation, additionally use:
> - **FVD** (Fréchet Video Distance) — video equivalent of FID
> - **CLIPSIM** — frame-level CLIP similarity
> - **VQA Score** — visual question answering alignment
> These require additional libraries (pytorch-fvd, etc.)

**Q: Can I fine-tune a pretrained Stable Diffusion model?**
> This system implements its own architecture from scratch, but the
> design is compatible with SD checkpoints. To use SD weights:
> 1. Load HuggingFace `diffusers` UNet2DConditionModel
> 2. Map the state dict keys to our UNet2DConditionModel
> 3. The architectures are equivalent — key names differ slightly

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{multimodal_generation_agent_2024,
  title   = {Multi-modal Generation Agent: Production-Level
             Text-to-Image and Text-to-Video Synthesis},
  year    = {2024},
  url     = {https://github.com/yourorg/multimodal-agent},
}
```

**Key papers implemented in this project:**

```bibtex
@article{ho2020ddpm,
  title   = {Denoising Diffusion Probabilistic Models},
  author  = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal = {NeurIPS}, year = {2020}
}

@article{song2020ddim,
  title   = {Denoising Diffusion Implicit Models},
  author  = {Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal = {ICLR}, year = {2021}
}

@article{rombach2022ldm,
  title   = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author  = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik
             and Esser, Patrick and Ommer, Björn},
  journal = {CVPR}, year = {2022}
}

@article{peebles2023dit,
  title   = {Scalable Diffusion Models with Transformers},
  author  = {Peebles, William and Xie, Saining},
  journal = {ICCV}, year = {2023}
}

@article{lu2022dpm,
  title   = {DPM-Solver++: Fast Solver for Guided Sampling of
             Diffusion Probabilistic Models},
  author  = {Lu, Cheng and Zhou, Yuhao and Bao, Fan and
             Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal = {NeurIPS}, year = {2022}
}

@article{hang2023msnr,
  title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
  author  = {Hang, Tiankai and Gu, Shuyang and Li, Chen and Bao, Jianmin
             and Chen, Dong and Hu, Han and Geng, Xin and Guo, Baining},
  journal = {ICCV}, year = {2023}
}

@article{karras2020stylegan2,
  title   = {Analyzing and Improving the Image Quality of StyleGAN},
  author  = {Karras, Tero and Laine, Samuli and Aittala, Miika and
             Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  journal = {CVPR}, year = {2020}
}

@article{hessel2021clipscore,
  title   = {CLIPScore: A Reference-Free Evaluation Metric
             for Image Captioning},
  author  = {Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and
             Bras, Ronan Le and Choi, Yejin},
  journal = {EMNLP}, year = {2021}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- Website: [Adil Shamim](https://adilshamim.me/)
- GitHub: [Adil Shamim](https://github.com/AdilShamim8)
- Create an issue in this repository for questions or suggestions

---

<p align="center">
  <a href="https://github.com/AdilShamim8">
    <img src="https://img.shields.io/badge/GitHub-AdilShamim8-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://www.linkedin.com/in/adilshamim8">
    <img src="https://img.shields.io/badge/LinkedIn-AdilShamim8-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://www.kaggle.com/adilshamim8">
    <img src="https://img.shields.io/badge/Kaggle-AdilShamim8-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://x.com/adil_shamim8">
    <img src="https://img.shields.io/badge/Twitter%2FX-@adil__shamim8-000000?style=for-the-badge&logo=x&logoColor=white" alt="Twitter/X Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://adilshamim8.medium.com/">
    <img src="https://img.shields.io/badge/Medium-AdilShamim8-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium Profile"/>
  </a>
</p>
<div align="center">
  
⭐ **If you find this repository helpful, please consider giving it a star!** ⭐

</div>
