from dataclasses import dataclass

from .base_config import BaseConfig, OptimizerConfig


@dataclass
class VAEConfig(BaseConfig):
    # Architecture
    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 128
    channel_multipliers: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16,)
    dropout: float = 0.0
    image_size: int = 256

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    kl_weight: float = 1e-6
    perceptual_weight: float = 1.0
    adversarial_weight: float = 0.5
    disc_start_step: int = 50_001

    optimizer: OptimizerConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig(
                learning_rate=4.5e-6,
                warmup_steps=0,
            )
