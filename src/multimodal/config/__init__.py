"""Experiment configuration for the Multi-Modal Generation Platform.

Dataclass-based configs (validated on construction) for every training
plane: VAE, GAN, diffusion (U-Net / DiT), T2I, and T2V. YAML-driven
loading with environment interpolation lives in `multimodal.config.loader`.
"""

from .base_config import BaseConfig, OptimizerConfig
from .vae_config import VAEConfig
from .gan_config import GANConfig
from .diffusion_config import DiffusionConfig
from .t2i_config import T2IConfig
from .t2v_config import T2VConfig

__all__ = [
    "BaseConfig",
    "OptimizerConfig",
    "VAEConfig",
    "GANConfig",
    "DiffusionConfig",
    "T2IConfig",
    "T2VConfig",
]
