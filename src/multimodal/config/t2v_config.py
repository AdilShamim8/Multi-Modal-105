"""Text-to-Video pipeline configuration.

Canonical home of `T2VConfig` (previously mis-homed in `diffusion_config`,
with an empty placeholder at this path). Covers the compression network,
factorized video DiT, and the memory-constrained training regime that
distinguishes video from image diffusion.
"""

from dataclasses import dataclass

from .diffusion_config import DiffusionConfig


@dataclass
class T2VConfig(DiffusionConfig):
    """Text-to-Video specific configuration."""

    # ── Clip geometry ─────────────────────────────────────────────────────
    num_frames: int = 16
    frame_size: int = 256
    fps: int = 8
    max_video_length: float = 10.0  # seconds

    # ── Video DiT (factorized space-time attention) ───────────────────────
    temporal_attention: bool = True
    temporal_heads: int = 8
    temporal_depth: int = 4
    rope_3d: bool = True  # factorized time/H/W rotary embeddings

    # ── Compression network (causal 3D VAE) ───────────────────────────────
    video_vae_path: str = "./checkpoints/video_vae"
    temporal_compression: int = 4  # T // 4 latent frames
    spatial_compression: int = 8  # H,W // 8 latent spatial

    # ── Memory-constrained training regime ────────────────────────────────
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 1_000_000
    gradient_checkpointing: bool = True
    activation_offload: bool = False  # CPU offload of activations (ZeRO-style)
    latent_caching: bool = True  # pre-encode clips once, reuse every epoch
    latent_cache_dir: str = "./latent_cache"

    # ── Data ──────────────────────────────────────────────────────────────
    video_data_root: str = "./data/webvid"
    min_resolution: int = 256
    max_resolution: int = 1024
    min_motion_score: float = 0.15  # drop near-static clips at prep time
