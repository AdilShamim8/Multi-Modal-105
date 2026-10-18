"""Device selection and accelerator housekeeping.

Rules of the platform:
* Never hardcode `"cuda"` — CI, M-series laptops, and CPU canaries exist.
* Query free memory before admitting a batch; OOM should be *predicted*,
  not discovered by the allocator at 3 a.m.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from ..exceptions import DeviceError


def resolve_device(requested: str | None = None) -> str:
    """Resolve the best available device honouring the request and CUDA_VISIBLE_DEVICES.

    Preference order: explicit request → cuda (if usable) → mps → cpu.
    The returned string is always a valid `torch.device` string.
    """
    want = (requested or os.getenv("MM_DEVICE", "auto")).lower()
    if want == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"
    if want.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                raise DeviceError(
                    "cuda requested but not available",
                    details={"requested": want},
                )
        except ImportError:  # pragma: no cover
            return "cpu"
    return want


def accelerator_info() -> dict[str, object]:
    """Free-form accelerator report for /healthz and job metadata."""
    info: dict[str, object] = {"device": resolve_device()}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info()
            info.update(
                gpu_name=props.name,
                total_memory_gb=round(props.total_memory / 1e9, 1),
                free_memory_gb=round(free_b / 1e9, 1),
                compute_capability=f"{props.major}.{props.minor}",
            )
    except ImportError:  # pragma: no cover
        pass
    return info


def free_device_memory_gb() -> float:
    """Free GPU memory in GB (0.0 on CPU-only hosts)."""
    try:
        import torch

        if torch.cuda.is_available():
            free_b, _total = torch.cuda.mem_get_info()
            return free_b / 1e9
    except ImportError:  # pragma: no cover
        pass
    return 0.0


def empty_cache() -> None:
    """Release cached allocator blocks (call between heterogeneous jobs)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def memory_watch(label: str, threshold_gb: float = 1.0) -> Iterator[None]:
    """Warn (structured) when a block leaves less free VRAM than threshold."""
    from ..logging import get_logger

    log = get_logger(__name__)
    before = free_device_memory_gb()
    try:
        yield
    finally:
        after = free_device_memory_gb()
        if before - after > threshold_gb:
            log.warning(
                "device memory grew during block",
                extra={"label": label, "delta_gb": round(before - after, 2)},
            )
