"""Production text-to-image inference pipeline.

Wraps VAE + text encoder + diffusion model into a unified API with
DDPM / DDIM / DPM-Solver++ sampling. Deployable via export bundles:
``T2IPipeline.from_pretrained`` loads the registry layout produced by
``mma export`` (bundle.json + config.json + weights.safetensors).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.models.text_encoder import CLIPTextEncoder
from multimodal.models.vae.vae import AutoencoderKL
from multimodal.sampling.ddim_sampler import DDIMSampler
from multimodal.sampling.ddpm_sampler import DDPMSampler
from multimodal.sampling.dpm_solver import DPMSolver

from .export import read_manifest

logger = logging.getLogger(__name__)


class T2IPipeline:
    """
    Production Text-to-Image inference pipeline.
    Wraps VAE + text encoder + diffusion model into a unified API.
    Supports DDPM / DDIM / DPM-Solver++ sampling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vae: AutoencoderKL,
        text_encoder: CLIPTextEncoder,
        scheduler: NoiseScheduler,
        device: str | torch.device = "cuda",
        sampler: str = "dpm++",  # "ddpm" | "ddim" | "dpm++"
        default_num_steps: int = 30,
        default_guidance_scale: float = 7.5,
    ):
        self.model = model.eval().to(device)
        self.vae = vae.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.scheduler = scheduler
        self.device = device
        self.default_num_steps = default_num_steps
        self.default_guidance_scale = default_guidance_scale

        # Build sampler
        self.sampler: DDPMSampler | DDIMSampler | DPMSolver
        if sampler == "ddpm":
            self.sampler = DDPMSampler(scheduler)
        elif sampler == "ddim":
            self.sampler = DDIMSampler(scheduler)
        elif sampler in ("dpm++", "dpm_solver"):
            self.sampler = DPMSolver(scheduler)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        self.sampler_name = sampler

    # ── Bundle loading ────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str | torch.device = "cpu",
        sampler: str = "dpm++",
        vae: AutoencoderKL | None = None,
        text_encoder: CLIPTextEncoder | None = None,
        scheduler: NoiseScheduler | None = None,
        default_num_steps: int = 30,
        default_guidance_scale: float = 7.5,
    ) -> T2IPipeline:
        """Build a pipeline from an export bundle directory.

        Layout (what ``mma export`` / the model registry writes):

            <model_dir>/bundle.json          integrity manifest (checksum)
            <model_dir>/config.json          architecture + component config
            <model_dir>/weights.safetensors  denoising model weights
            <model_dir>/vae.safetensors      optional image-VAE weights

        The denoising architecture is dispatched on
        ``config.json["architecture"]`` ("dit" default | "unet"). ``vae`` and
        ``text_encoder`` can be injected by callers that already hold them
        (tests, workers sharing one encoder across pipelines); otherwise the
        VAE comes from ``vae.safetensors`` when present and the text encoder
        is built from ``config.json["text_encoder"]`` (a HuggingFace id).
        """
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        read_manifest(model_dir)  # checksum gate — raises CheckpointError

        config: dict[str, Any] = {}
        config_path = model_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())

        architecture = config.get("architecture", "dit")
        if architecture == "dit":
            from multimodal.models.diffusion.dit import DiT

            model: torch.nn.Module = DiT(
                input_size=config.get("latent_size", 64),
                patch_size=config.get("patch_size", 2),
                hidden_size=config.get("hidden_size", 1152),
                depth=config.get("depth", 28),
                num_heads=config.get("num_heads", 16),
            )
        elif architecture == "unet":
            from multimodal.models.diffusion.unet import UNet2DConditionModel

            model = UNet2DConditionModel(**config.get("unet", {}))
        else:
            raise ValueError(
                f"unsupported bundle architecture: {architecture!r} " "(expected 'dit' or 'unet')"
            )

        state = load_file(str(model_dir / "weights.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "bundle weights partially matched architecture",
                extra={"missing": len(missing), "unexpected": len(unexpected)},
            )

        if vae is None:
            vae = AutoencoderKL(**config.get("vae", {}))
            vae_weights = model_dir / "vae.safetensors"
            if vae_weights.exists():
                vae.load_state_dict(load_file(str(vae_weights)), strict=False)
            else:
                logger.warning(
                    "no vae.safetensors in bundle — decode quality depends on "
                    "a trained VAE; inject one via from_pretrained(vae=...) if "
                    "the registry carries it separately",
                )

        if text_encoder is None:
            text_encoder = CLIPTextEncoder(
                model_name=config.get("text_encoder", "openai/clip-vit-large-patch14")
            )

        if scheduler is None:
            scheduler = NoiseScheduler(
                num_train_timesteps=config.get("num_train_timesteps", 1000),
                beta_schedule=config.get("beta_schedule", "scaled_linear"),
                prediction_type=config.get("prediction_type", "v_prediction"),
            )

        return cls(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            device=device,
            sampler=sampler,
            default_num_steps=default_num_steps,
            default_guidance_scale=default_guidance_scale,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> dict[str, Any]:
        """
        Generate images from text prompt(s).
        Returns:
            dict with 'images' (List[PIL.Image]) and optionally 'latents'
        """
        num_steps = num_inference_steps or self.default_num_steps
        cfg_scale = guidance_scale or self.default_guidance_scale

        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)

        B = len(prompt) * num_images_per_prompt

        # Seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Encode prompts
        pos_tokens = self.text_encoder.tokenize(prompt, device=self.device)
        neg_tokens = self.text_encoder.tokenize(negative_prompt, device=self.device)

        pos_embeds = self.text_encoder.encode(
            pos_tokens["input_ids"].repeat_interleave(num_images_per_prompt, dim=0),
            pos_tokens.get("attention_mask"),
        )
        neg_embeds = self.text_encoder.encode(
            neg_tokens["input_ids"].repeat_interleave(num_images_per_prompt, dim=0),
            neg_tokens.get("attention_mask"),
        )

        # Concatenate for CFG
        text_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)

        # Define model wrapper for CFG
        def model_with_cfg(x, t, **kwargs):
            x_in = torch.cat([x, x], dim=0)
            t_in = t.repeat(2)
            model_out = self.model(x_in, t_in, encoder_hidden_states=text_embeds)
            if model_out.shape[1] == x.shape[1] * 2:
                model_out, _ = model_out.chunk(2, dim=1)
            cond, uncond = model_out.chunk(2, dim=0)
            return uncond + cfg_scale * (cond - uncond)

        # Latent shape
        shape = (B, 4, height // 8, width // 8)

        # Sample
        denoised_latents = self.sampler.sample(
            model=model_with_cfg,
            shape=shape,
            device=self.device,
            num_inference_steps=num_steps,
            guidance_scale=1.0,  # CFG is handled by model_with_cfg
            show_progress=True,
            latents=latents,
        )

        # Decode
        images_tensor = self.vae.decode_from_latent(denoised_latents)
        images_tensor = (images_tensor.clamp(-1, 1) + 1) / 2.0  # [0, 1]

        # Convert to PIL
        pil_images = []
        for i in range(B):
            arr = (images_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(arr))

        result: dict[str, Any] = {"images": pil_images}
        if return_latents:
            result["latents"] = denoised_latents
        return result

    @torch.no_grad()
    def img2img(
        self,
        prompt: str | list[str],
        init_image: Image.Image | torch.Tensor,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """
        Image-to-image generation with DDIM inversion.
        strength: 0 = no change, 1 = full noise
        """
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(prompt, str):
            prompt = [prompt]

        # Preprocess init image
        if isinstance(init_image, Image.Image):
            init_image = init_image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            import torchvision.transforms as T

            t = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
            init_tensor = t(init_image).unsqueeze(0).to(self.device)
        else:
            init_tensor = init_image.to(self.device)

        # Encode to latent
        init_latent = self.vae.encode_to_latent(init_tensor)

        # Add noise up to strength level
        t_start = int(strength * self.scheduler.num_train_timesteps)
        t_tensor = torch.tensor([t_start], device=self.device)
        noise = torch.randn_like(init_latent)
        noisy_latent = self.scheduler.add_noise(init_latent, noise, t_tensor)

        result = self.__call__(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=noisy_latent,
        )
        return result["images"]
