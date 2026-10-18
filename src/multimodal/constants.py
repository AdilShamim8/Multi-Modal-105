"""Platform-wide constants.

Anything referenced by more than one subsystem and expected to change
rarely belongs here — magic numbers duplicated across modules are how
"works on my machine" incidents start.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# ── Environment names ───────────────────────────────────────────────────────
ENV_DEVELOPMENT = "development"
ENV_STAGING = "staging"
ENV_PRODUCTION = "production"
ENV = os.getenv("MM_ENV", ENV_DEVELOPMENT)

# ── Latent-space geometry (matches the migrated VAE) ────────────────────────
LATENT_CHANNELS = 4                 # AutoencoderKL latent depth
VAE_DOWNSAMPLING = 8                # spatial compression factor
VAE_SCALING_FACTOR = 0.18215        # SD-compatible latent normalizer

# ── Diffusion defaults ──────────────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_BETA_SCHEDULE = "scaled_linear"
DEFAULT_PREDICTION_TYPE = "v_prediction"
DEFAULT_SAMPLER = "dpm++"
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5

# ── Serving limits (defence in depth; API schema repeats these) ─────────────
MAX_STEPS = _env_int("MM_MAX_STEPS", 100)
MAX_BATCH_SIZE = _env_int("MM_MAX_BATCH_SIZE", 8)
MAX_PROMPT_CHARS = _env_int("MM_MAX_PROMPT_CHARS", 2000)
MAX_VIDEO_FRAMES = 64
SYNC_TIMEOUT_S = _env_float("MM_SYNC_TIMEOUT_S", 30.0)

# ── Queue ───────────────────────────────────────────────────────────────────
QUEUE_NAME = os.getenv("MM_QUEUE_NAME", "mm:generation:jobs")
QUEUE_MAX_DEPTH = _env_int("MM_QUEUE_MAX_DEPTH", 10_000)
JOB_TTL_S = _env_int("MM_JOB_TTL_S", 86_400)
JOB_HASH_PREFIX = "mm:job:"

# ── Retry / resilience ──────────────────────────────────────────────────────
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_S = 0.5
OOM_RETRIES = 2                     # retry generation with smaller batch
WORKER_GRACE_PERIOD_S = 30.0        # drain in-flight jobs before exit

# ── Content safety ──────────────────────────────────────────────────────────
MODERATION_ENABLED = os.getenv("MM_MODERATION_ENABLED", "true").lower() == "true"

__all__ = [name for name in dir() if name.isupper()]
