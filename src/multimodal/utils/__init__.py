"""Utility subpackage: reproducibility, devices, timing, checkpoints, distributed.

Import-time policy: this package must stay importable in torch-free
containers (API-only deployments). Modules that require torch
(`checkpoint`, `distributed`, `logging_utils`) are resolved lazily via
PEP 562 so they are only loaded when actually used.
"""

from typing import TYPE_CHECKING

from .devices import (
    accelerator_info,
    empty_cache,
    free_device_memory_gb,
    resolve_device,
)
from .seeds import derive_seed, is_seeded, seed_everything
from .timer import Stopwatch, timer

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .checkpoint import CheckpointManager
    from .distributed import all_reduce_mean, is_main_process
    from .logging_utils import MetricLogger

__all__ = [
    "seed_everything",
    "is_seeded",
    "derive_seed",
    "resolve_device",
    "accelerator_info",
    "free_device_memory_gb",
    "empty_cache",
    "timer",
    "Stopwatch",
    "CheckpointManager",
    "is_main_process",
    "all_reduce_mean",
    "MetricLogger",
]

_LAZY = {
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "is_main_process": (".distributed", "is_main_process"),
    "all_reduce_mean": (".distributed", "all_reduce_mean"),
    "MetricLogger": (".logging_utils", "MetricLogger"),
}


def __getattr__(name: str):
    import importlib

    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0], __package__)
    return getattr(module, target[1])
