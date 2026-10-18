# Glossary

> Short definitions, platform-specific usage. Last updated: 2026-10-18.

**AdaLN-Zero** — adaptive layer norm whose modulation gates are
zero-initialized, so each transformer block starts as identity; the
property that lets deep DiTs train at all. In `models/diffusion/dit.py`.

**Admission control** — rejecting jobs (503 + Retry-After) when queue
depth exceeds `MM_QUEUE_MAX_DEPTH`, instead of promising unbounded waits.

**Artifact** — generated media bytes in the artifact store, addressed by a
content-derived key (`job_id/kind-sha16.ext`); duplicates are impossible.

**BRPOPLPUSH** — Redis reliable-queue primitive: pop from queue, park in a
processing list, ack by removal. The pattern that makes worker crashes
recoverable.

**Bundle** — deployable model artifact: safetensors weights +
`bundle.json` manifest (sha256, versions). The ONLY thing serving loads.

**CFG (classifier-free guidance)** — `ε_u + s·(ε_c − ε_u)`; steering
between conditional and unconditional predictions. Requires caption
dropout (10%) during training.

**Causal 3D VAE** — video autoencoder whose temporal convolutions only
look backwards: enables streaming decode and clip extension.

**Continuous batching** — coalescing queued jobs with the same execution
signature into one GPU batch under a latency budget (`serving/batching.py`).

**Correlation id** — per-request identifier flowing through API → queue →
worker → logs → job record; the triage handle for every incident.

**DiT** — Diffusion Transformer: latents patchified into tokens, denoised
by a transformer with timestep conditioning. The platform's primary
denoiser (ADR-006).

**DPM-Solver++** — fast ODE sampler; the production default (30 steps at
50-step DDIM quality).

**EMA** — exponential moving average of weights; what evaluation and
export consume, because it averages out mode-collapse wobble.

**FID** — Fréchet Inception Distance: distributional gap between generated
and reference images. Release gate ≤ 25. Fooled by small samples.

**IS** — Inception Score: class coverage × confidence. Release gate ≥ 8.

**KEDA** — Kubernetes event-driven autoscaler; scales GPU workers on
queue depth, not CPU (workers are GPU-bound and CPU-idle).

**Latent caching** — pre-encoding dataset media through the (frozen)
VAE once, then training on cached latents every epoch; ~40% step-time
saving, the large-scale T2V unlock.

**LDM** — latent diffusion: denoise in VAE latent space (48× fewer
tokens), decode once at the end.

**min-SNR-γ** — loss-weighting that clips per-timestep weights at
signal-to-noise γ=5; measurably faster convergence.

**Processing list** — the Redis list holding jobs currently executing;
its strays are requeued by `recover_stuck()` at worker startup.

**R1 (lazy)** — discriminator gradient penalty applied every 16 steps,
amortized ÷16 — equal pressure at 1/16 the compute.

**Signature** — execution-relevant grouping key (modality, geometry,
steps, sampler); batching only merges jobs with identical signatures.

**SLO / error budget** — the user-facing contract (e.g. p95 ≤ 60 s) and
the minutes of tolerable breach per window; budget exhaustion triggers
reliability freeze.

**Straight-through estimator** — VQ-VAE trick: forward value is the
codebook entry, backward gradient is the identity — tested verbatim.

**Truncation trick** — pulling StyleGAN w toward its running mean by ψ<1;
fidelity/diversity dial at sample time.

**v-prediction** — predicting `v = √ᾱ·ε − √(1−ᾱ)·x₀` instead of ε; flat
SNR profile, stable at high resolution. The scheduler default.
