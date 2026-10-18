"""Text-to-Image pipeline configuration.

Canonical home of `T2IConfig` (previously mis-homed in `diffusion_config`,
with an empty placeholder at this path). Aggregates diffusion architecture,
text conditioning, and data-curation parameters for the T2I plane.
"""

from dataclasses import dataclass

from .diffusion_config import DiffusionConfig


@dataclass
class T2IConfig(DiffusionConfig):
    """Text-to-Image specific configuration."""

    # ── Conditioning components ───────────────────────────────────────────
    vae_model_path: str = "stabilityai/sd-vae-ft-mse"
    text_encoder_name: str = "openai/clip-vit-large-patch14"
    tokenizer_name: str = "openai/clip-vit-large-patch14"
    max_token_length: int = 77
    guidance_scale: float = 7.5

    # ── Data curation ─────────────────────────────────────────────────────
    data_root: str = "./data/laion"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    min_caption_chars: int = 5  # drop junk captions at prep time
    min_aesthetic_score: float = 4.5  # CLIP-aesthetic pre-filter hook

    # ── Training scale ────────────────────────────────────────────────────
    batch_size: int = 4
    num_train_steps: int = 400_000

    # ── Classifier-free guidance dropout (training time) ──────────────────
    caption_dropout_prob: float = 0.1
