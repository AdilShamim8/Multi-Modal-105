from dataclasses import dataclass, field
from typing import Optional, List
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
    snr_gamma: Optional[float] = 5.0   # min-SNR weighting

    optimizer: OptimizerConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig(
                learning_rate=1e-4,
                warmup_steps=1000,
            )


@dataclass
class T2IConfig(DiffusionConfig):
    """Text-to-Image specific configuration."""
    vae_model_path: str = "stabilityai/sd-vae-ft-mse"
    text_encoder_name: str = "openai/clip-vit-large-patch14"
    tokenizer_name: str = "openai/clip-vit-large-patch14"
    max_token_length: int = 77
    guidance_scale: float = 7.5

    # Data
    data_root: str = "./data/laion"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True

    batch_size: int = 4
    num_train_steps: int = 400_000


@dataclass
class T2VConfig(DiffusionConfig):
    """Text-to-Video specific configuration."""
    # Video specifics
    num_frames: int = 16
    frame_size: int = 256
    fps: int = 8
    max_video_length: float = 10.0      # seconds

    # Video DiT
    temporal_attention: bool = True
    temporal_heads: int = 8
    temporal_depth: int = 4

    # Compression network
    video_vae_path: str = "./checkpoints/video_vae"
    temporal_compression: int = 4      # T // 4 latent frames
    spatial_compression: int = 8       # H,W // 8 latent spatial

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 1_000_000

    # Data
    video_data_root: str = "./data/webvid"
    min_resolution: int = 256
    max_resolution: int = 1024