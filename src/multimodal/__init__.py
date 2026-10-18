"""Multi-Modal Generation Platform.

A production-grade platform for text-to-image and text-to-video synthesis
built from first principles: VAE compression networks, GANs, autoregressive
token models, and latent diffusion transformers (U-Net and DiT), wrapped in
a serving plane (FastAPI + Redis queue + GPU workers) designed for
internet-scale traffic.

Import cost is kept low: heavy subpackages (models, serving) are imported
lazily through `__getattr__` so that `import multimodal` is instant even on
cold starts — important for CLI responsiveness and serverless-style workers.

Release: 1.0.1 (2026-10-18). Maintainer: Adil Shamim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .version import RELEASE_DATE, __version__

if TYPE_CHECKING:  # pragma: no cover - static imports for type checkers only
    from .exceptions import MultimodalError
    from .registry import Registry

__all__ = [
    "__version__",
    "RELEASE_DATE",
    # Lazily resolved via __getattr__ (PEP 562); statically typed via the
    # TYPE_CHECKING imports above so IDEs and mypy see the real types.
    "MultimodalError",
    "Registry",
]

_LAZY = {
    "MultimodalError": (".exceptions", "MultimodalError"),
    "Registry": (".registry", "Registry"),
    "get_logger": (".logging", "get_logger"),
    "seed_everything": (".utils.seeds", "seed_everything"),
}


def __getattr__(name: str):
    """Lazily resolve a small set of core symbols (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'multimodal' has no attribute '{name}'")
    import importlib

    module = importlib.import_module(target[0], __name__)
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__ + list(_LAZY))
