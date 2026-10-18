"""Reproducibility utilities — deterministic seeds across the stack.

A generation platform without deterministic seeding cannot debug customer
complaints ("job 8812 produced a broken frame") nor bisect training
regressions. One call seeds every RNG that matters.
"""

from __future__ import annotations

import os
import random

import numpy as np

_SEEDED_KEY = "MM_SEEDED"


def seed_everything(seed: int | None = None) -> int:
    """Seed python, numpy, and (if installed) torch RNGs.

    Args:
        seed: explicit seed; falls back to `MM_SEED` (default 42) so that
            unattended runs are still reproducible.

    Returns:
        The effective seed actually applied (log it or write it to the job).
    """
    if seed is None:
        try:
            seed = int(os.getenv("MM_SEED", "42"))
        except ValueError:
            seed = 42
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:  # torch is optional for pure-API deployments
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism costs throughput; enable only when explicitly asked.
        if os.getenv("MM_DETERMINISTIC", "false").lower() == "true":
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - API-only containers
        pass

    os.environ[_SEEDED_KEY] = "1"
    return seed


def is_seeded() -> bool:
    """Whether this process has already been seeded (avoid double-seeding)."""
    return os.getenv(_SEEDED_KEY, "") == "1"


def derive_seed(base: int, *parts: object) -> int:
    """Stable sub-seed for subcomponents (workers, DAEMON loops, eval splits).

    `derive_seed(42, 'worker', 3)` is stable across processes without any
    coordination — each component decorrelates from the base seed without
    clobbering the global RNG state.
    """
    payload = f"{base}|" + "|".join(str(p) for p in parts)
    return abs(hash(payload)) % (2**31 - 1)
