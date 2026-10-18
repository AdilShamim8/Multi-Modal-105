import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging

from multimodal.models.vae.vae import AutoencoderKL
from multimodal.models.text_encoder import CLIPTextEncoder
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.sampling.ddim_sampler import DDIMSampler
from multimodal.sampling.dpm_solver import DPMSolver
from multimodal.sampling.ddpm_sampler import DDPMSampler

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
        device: str = "cuda",
        sampler: str = "dpm++",         # "ddpm" | "ddim" | "dpm++"
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
        if sampler == "ddpm":
            self.sampler = DDPMSampler(scheduler)
        elif sampler == "ddim":
            self.sampler = DDIMSampler(scheduler)
        elif sampler in ("dpm++", "dpm_solver"):
            self.sampler = DPMSolver(scheduler)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        self.sampler_name = sampler

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Dict[str, Any]:
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
            guidance_scale=1.0,       # CFG is handled by model_with_cfg
            show_progress=True,
            latents=latents,
        )

        # Decode
        images_tensor = self.vae.decode_from_latent(denoised_latents)
        images_tensor = (images_tensor.clamp(-1, 1) + 1) / 2.0   # [0, 1]

        # Convert to PIL
        pil_images = []
        for i in range(B):
            arr = (
                images_tensor[i].permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            pil_images.append(Image.fromarray(arr))

        result = {"images": pil_images}
        if return_latents:
            result["latents"] = denoised_latents
        return result

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[Image.Image, torch.Tensor],
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
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
            init_image = init_image.convert("RGB").resize((512, 512), Image.LANCZOS)
            import torchvision.transforms as T
            t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
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