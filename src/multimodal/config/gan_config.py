"""GAN configuration (StyleGAN2 family).

Fills the previously empty placeholder. Parameters mirror the constructors
of `StyleGAN2Generator` / `StyleGAN2Discriminator` so a config object can
build either network without glue code.
"""

from dataclasses import dataclass

from .base_config import BaseConfig, OptimizerConfig


@dataclass
class GANConfig(BaseConfig):
    """StyleGAN2 training and architecture configuration."""

    # ── Shared geometry ───────────────────────────────────────────────────
    img_resolution: int = 256
    img_channels: int = 3

    # ── Generator ─────────────────────────────────────────────────────────
    z_dim: int = 512
    w_dim: int = 512
    mapping_num_layers: int = 8
    mapping_lr_mul: float = 0.01
    channel_base: int = 32768
    channel_max: int = 512
    synthesis_lr_mul: float = 1.0

    # ── Discriminator ─────────────────────────────────────────────────────
    mbstd_group_size: int = 4

    # ── Adversarial objective ─────────────────────────────────────────────
    loss_type: str = "softplus"  # "softplus" | "hinge" | "wgan-gp"
    r1_gamma: float = 10.0  # R1 lazy regularization strength
    lazy_reg_interval: int = 16  # apply R1 every N critic steps
    gp_lambda: float = 10.0  # only used when loss_type == "wgan-gp"

    # ── Optimization (StyleGAN2 defaults) ─────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 100
    gen_lr: float = 2e-3
    disc_lr: float = 2e-3
    beta1: float = 0.0
    beta2: float = 0.99

    # ── Stability ─────────────────────────────────────────────────────────
    ema_beta: float = 0.999  # generator weight EMA for evaluation
    style_mixing_prob: float = 0.9  # path regularization via style mixing
    num_train_steps: int = 2_000_000
    # Adversarial objectives are gradient-unstable by construction; both the
    # critic and generator are clipped every step (GANTrainer.train_step).
    max_grad_norm: float = 1.0

    optimizer: OptimizerConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.loss_type not in ("softplus", "hinge", "wgan-gp"):
            raise ValueError(f"loss_type must be softplus|hinge|wgan-gp, got '{self.loss_type}'")
        if self.img_resolution & (self.img_resolution - 1) != 0:
            raise ValueError("img_resolution must be a power of two")
        if self.optimizer is None:
            self.optimizer = OptimizerConfig(
                optimizer="adam",
                learning_rate=self.gen_lr,
                beta1=self.beta1,
                beta2=self.beta2,
                warmup_steps=0,
            )
