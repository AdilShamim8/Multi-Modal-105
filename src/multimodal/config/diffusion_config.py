from dataclasses import dataclass

from .base_config import BaseConfig, OptimizerConfig


@dataclass
class DiffusionConfig(BaseConfig):
    # ── Noise schedule ────────────────────────────────────────────────────────
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"  # "linear"|"cosine"|"scaled_linear"
    prediction_type: str = "v_prediction" # "epsilon"|"sample"|"v_prediction"
    clip_sample: bool = False

    # ── UNet architecture ─────────────────────────────────────────────────────
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_base_channels: int = 320
    unet_channel_multipliers: tuple = (1, 2, 4, 4)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: tuple = (4, 2, 1)
    unet_num_heads: int = 8
    unet_head_dim: int = 64
    unet_dropout: float = 0.1

    # ── DiT architecture ──────────────────────────────────────────────────────
    dit_patch_size: int = 2
    dit_depth: int = 28
    dit_hidden_size: int = 1152
    dit_num_heads: int = 16
    dit_mlp_ratio: float = 4.0
    dit_class_dropout_prob: float = 0.1
    dit_num_classes: int = 1000
    dit_learn_sigma: bool = True

    # ── Text conditioning ─────────────────────────────────────────────────────
    text_encoder_name: str = "openai/clip-vit-large-patch14"
    text_encoder_dim: int = 768
    cross_attention_dim: int = 768

    # ── Training ──────────────────────────────────────────────────────────────
    image_size: int = 512
    latent_size: int = 64           # image_size // vae_downsample_factor (8)
    batch_size: int = 4
    num_train_steps: int = 200_000
    ema_decay: float = 0.9999
    snr_gamma: float | None = 5.0   # min-SNR weighting

    optimizer: OptimizerConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig(
                learning_rate=1e-4,
                warmup_steps=1000,
            )


# NOTE: T2IConfig and T2VConfig live in their own modules (t2i_config.py,
# t2v_config.py) — import them from there directly; re-exporting here would
# create an import cycle (they subclass DiffusionConfig).
