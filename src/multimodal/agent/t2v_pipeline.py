"""Production text-to-video inference pipeline.

Complements :class:`multimodal.agent.pipeline.T2IPipeline` for the video
modality: a causal 3D video VAE for compression, a VideoDiT denoiser, and a
T5/CLIP-style text encoder, driven by a deterministic DDIM update loop.
Deployable via export bundles — ``T2VPipeline.from_pretrained`` loads the
registry layout produced by ``mma export`` (bundle.json + config.json +
weights.safetensors [+ video_vae.safetensors]).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from multimodal.models.compression.video_vae import VideoAutoencoderKL
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler

from .export import read_manifest

logger = logging.getLogger(__name__)


class TextEncoderLike(Protocol):
    """Structural contract satisfied by T5TextEncoder and CLIPTextEncoder."""

    def tokenize(self, texts: str | list[str]) -> dict[str, torch.Tensor]: ...

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class T2VPipeline:
    """Text-to-Video latent diffusion pipeline (video VAE + VideoDiT)."""

    def __init__(
        self,
        model: torch.nn.Module,
        video_vae: VideoAutoencoderKL,
        text_encoder: TextEncoderLike,
        scheduler: NoiseScheduler,
        device: str | torch.device = "cpu",
        default_num_steps: int = 50,
        default_guidance_scale: float = 7.5,
        fps: int = 8,
    ):
        self.model = model.eval().to(device)
        self.video_vae = video_vae.eval().to(device)
        if isinstance(text_encoder, torch.nn.Module):  # T5/CLIP encoders are Modules
            text_encoder = text_encoder.eval().to(device)
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.device = device
        self.default_num_steps = default_num_steps
        self.default_guidance_scale = default_guidance_scale
        self.fps = fps

    # ── Generation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 256,
        height: int = 256,
        num_frames: int = 16,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        num_outputs: int = 1,
        seed: int | None = None,
        output_format: str = "mp4",
        **_extra: Any,
    ) -> dict[str, Any]:
        """Generate video clips from text prompt(s).

        Returns ``{"videos": [...]}`` where each item is MP4-encoded bytes
        (``output_format="mp4"``) or a ``(3, T, H, W)`` uint8 tensor
        (``output_format="raw"`` — no imageio dependency, used by tests).

        Engine-forward kwargs the T2I side honors but video ignores
        (e.g. ``sampler``) are accepted and ignored: the deterministic DDIM
        update is the only production-validated video sampler.
        """
        if output_format not in ("mp4", "raw"):
            raise ValueError(f"output_format must be 'mp4' or 'raw', got {output_format!r}")

        num_steps = num_steps or self.default_num_steps
        cfg_scale = guidance_scale or self.default_guidance_scale

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        negatives = (
            [negative_prompt] * len(prompts)
            if negative_prompt is None or isinstance(negative_prompt, str)
            else list(negative_prompt)
        )
        if negative_prompt is None:
            negatives = [""] * len(prompts)

        B = len(prompts) * num_outputs
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ── Conditioning ──────────────────────────────────────────────
        pos = self.text_encoder.tokenize(prompts)
        pos_embeds = self.text_encoder.forward(
            pos["input_ids"], pos.get("attention_mask")
        ).repeat_interleave(num_outputs, dim=0)
        neg = self.text_encoder.tokenize(negatives)
        neg_embeds = self.text_encoder.forward(
            neg["input_ids"], neg.get("attention_mask")
        ).repeat_interleave(num_outputs, dim=0)

        # ── Latent geometry from the compression network ──────────────
        vae = self.video_vae
        t_lat = max(num_frames // vae.temporal_compression, 1)
        h_lat = max(height // vae.spatial_compression, 1)
        w_lat = max(width // vae.spatial_compression, 1)
        shape = (B, vae.latent_channels, t_lat, h_lat, w_lat)

        x = torch.randn(*shape, device=self.device)

        # ── Deterministic DDIM reverse pass ────────────────────────────
        self.scheduler.set_timesteps(num_steps, device=str(self.device))
        timesteps = self.scheduler.timesteps

        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(B)

            # CFG: conditional + unconditional in one doubled batch.
            x_double = torch.cat([x, x], dim=0)
            t_double = t_batch.repeat(2)
            emb_double = torch.cat([pos_embeds, neg_embeds], dim=0)
            out = self.model(x_double, t_double, text_embeds=emb_double, text_pool=None)
            cond, uncond = out.chunk(2, dim=0)
            model_out = uncond + cfg_scale * (cond - uncond)
            if model_out.shape[1] == x.shape[1] * 2:  # learn_sigma layouts
                model_out, _ = model_out.chunk(2, dim=1)

            x0_pred = self.scheduler.get_x0_prediction(model_out, x, t_batch).clamp(-1.0, 1.0)

            t_prev = int(timesteps[i + 1]) if i < len(timesteps) - 1 else 0
            alpha_prev = self.scheduler.alphas_cumprod[max(t_prev, 0)].to(x.device)

            eps = self.scheduler.predict_noise_from_x0(x, t_batch, x0_pred)
            dir_xt = (1 - alpha_prev).clamp(min=0).sqrt() * eps
            x = alpha_prev.sqrt() * x0_pred + dir_xt

        # ── Decode to pixels ───────────────────────────────────────────
        videos = self.video_vae.decode_from_latent(x)  # (B, 3, T, H, W)
        videos = ((videos.clamp(-1, 1) + 1) / 2 * 255).byte()

        if output_format == "raw":
            return {"videos": [videos[b] for b in range(B)]}

        # ── Encode to MP4 bytes (imageio is a lazy t2v extra) ──────────
        try:
            import imageio
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "mp4 video export requires imageio; install it via "
                "'pip install multimodal-agent[t2v]'"
            ) from exc

        encoded: list[bytes] = []
        for b in range(B):
            frames = videos[b].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
            buf = io.BytesIO()
            # format= accepts a name string at runtime (imageio docs); the
            # bundled stubs model only Format objects, hence the narrow ignore.
            imageio.mimwrite(  # type: ignore[call-overload]
                buf,
                list(frames),
                format="mp4",
                fps=self.fps,
                quality=8,
                macro_block_size=1,
            )
            encoded.append(buf.getvalue())
        return {"videos": encoded}

    # ── Bundle loading ────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str | torch.device = "cpu",
        video_vae: VideoAutoencoderKL | None = None,
        text_encoder: TextEncoderLike | None = None,
        scheduler: NoiseScheduler | None = None,
        default_num_steps: int = 50,
        default_guidance_scale: float = 7.5,
        fps: int = 8,
    ) -> T2VPipeline:
        """Build a video pipeline from an export bundle directory.

        Layout mirrors the T2I bundles; ``config.json["architecture"]``
        must be ``"video_dit"``. ``video_vae.safetensors`` is optional —
        without it the compression network starts from its config defaults
        and a warning is logged (decode quality depends on trained weights).
        ``text_encoder`` defaults to the T5 encoder named by
        ``config.json["text_encoder"]``.
        """
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        read_manifest(model_dir)  # checksum gate — raises CheckpointError

        config: dict[str, Any] = {}
        config_path = model_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())

        architecture = config.get("architecture", "video_dit")
        if architecture != "video_dit":
            raise ValueError(
                f"unsupported video bundle architecture: {architecture!r} " "(expected 'video_dit')"
            )

        from multimodal.models.diffusion.dit_video import VideoDiT

        model: torch.nn.Module = VideoDiT(
            num_frames=config.get("latent_num_frames", 4),
            frame_size=config.get("latent_frame_size", 32),
            in_channels=config.get("latent_channels", 16),
            spatial_patch_size=config.get("spatial_patch_size", 2),
            temporal_patch_size=config.get("temporal_patch_size", 1),
            hidden_size=config.get("hidden_size", 1152),
            depth=config.get("depth", 28),
            num_heads=config.get("num_heads", 16),
            temporal_heads=config.get("temporal_heads", 8),
            context_dim=config.get("context_dim", 768),
            pooled_text_dim=config.get("pooled_text_dim", 1024),
        )

        state = load_file(str(model_dir / "weights.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "bundle weights partially matched architecture",
                extra={"missing": len(missing), "unexpected": len(unexpected)},
            )

        if video_vae is None:
            video_vae = VideoAutoencoderKL(**config.get("video_vae", {}))
            vae_weights = model_dir / "video_vae.safetensors"
            if vae_weights.exists():
                video_vae.load_state_dict(load_file(str(vae_weights)), strict=False)
            else:
                logger.warning(
                    "no video_vae.safetensors in bundle — decode quality depends "
                    "on a trained compression network; inject one via "
                    "from_pretrained(video_vae=...) if the registry carries it "
                    "separately",
                )

        if text_encoder is None:
            from multimodal.models.text_encoder import T5TextEncoder

            text_encoder = T5TextEncoder(
                model_name=config.get("text_encoder", "google/t5-v1_1-xl"),
                max_length=config.get("max_token_length", 256),
            )

        if scheduler is None:
            scheduler = NoiseScheduler(
                num_train_timesteps=config.get("num_train_timesteps", 1000),
                beta_schedule=config.get("beta_schedule", "scaled_linear"),
                prediction_type=config.get("prediction_type", "v_prediction"),
            )

        return cls(
            model=model,
            video_vae=video_vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            device=device,
            default_num_steps=default_num_steps,
            default_guidance_scale=default_guidance_scale,
            fps=fps,
        )
