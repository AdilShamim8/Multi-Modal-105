"""Shared test fixtures.

The suite is CPU-first by design (PR gate < 90 s): tiny geometries,
deterministic seeds, and fakes for anything with a network or GPU.
`src` is prepended to sys.path so tests run from a checkout without
installing (CI installs the package; both work).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _deterministic():
    """Every test starts from a known RNG state."""
    from multimodal.utils.seeds import seed_everything

    seed_everything(1234)


@pytest.fixture()
def tiny_vae_config():
    from multimodal.config.vae_config import VAEConfig

    return VAEConfig(
        base_channels=32,  # divisible by GroupNorm's 32 groups
        channel_multipliers=(1, 2, 4, 8),  # 4 stages → symmetric 8x encoder/decoder
        num_res_blocks=1,
        attention_resolutions=(8,),
        image_size=64,
    )


@pytest.fixture()
def tiny_scheduler():
    from multimodal.models.diffusion.noise_scheduler import NoiseScheduler

    return NoiseScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
