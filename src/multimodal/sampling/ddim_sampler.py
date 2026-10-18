import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Tuple
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler


class DDIMSampler:
    """
    DDIM deterministic sampling (Song et al. 2020).
    Allows 50-100x fewer steps than DDPM with comparable quality.
    Supports eta ∈ [0, 1] for stochasticity interpolation.
    """

    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler

    def _get_ddim_timesteps(
        self, num_inference_steps: int, num_train_steps: int
    ) -> np.ndarray:
        """Uniformly spaced subset of training timesteps."""
        step_ratio = num_train_steps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        return timesteps.astype(np.int64)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device,
        conditioning: Optional[dict] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,              # 0 = deterministic, 1 = DDPM-like
        clip_denoised: bool = True,
        show_progress: bool = True,
        callback: Optional[Callable] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM reverse process.
        x_{t-1} = sqrt(ā_{t-1})*x̂_0 + dir_xt + σ_t*ε
        """
        timesteps = self._get_ddim_timesteps(
            num_inference_steps, self.scheduler.num_train_timesteps
        )
        cond = conditioning or {}
        do_cfg = guidance_scale > 1.0

        # Initialize
        x = latents if latents is not None else torch.randn(*shape, device=device)

        from tqdm import tqdm
        pbar = tqdm(timesteps, desc="DDIM sampling") if show_progress else timesteps

        for i, t in enumerate(pbar):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0

            # CFG model call
            if do_cfg:
                x_double = torch.cat([x, x], dim=0)
                t_double = t_tensor.repeat(2)
                model_out = model(x_double, t_double, **self._double_cond(cond, shape[0]))
                cond_out, uncond_out = model_out.chunk(2, dim=0)
                model_output = uncond_out + guidance_scale * (cond_out - uncond_out)
            else:
                model_output = model(x, t_tensor, **cond)

            # Predict x_0
            x0_pred = self.scheduler.get_x0_prediction(model_output, x, t_tensor)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update step
            x = self._ddim_step(x, x0_pred, t, t_prev, eta, device)

            if callback is not None:
                callback(i, t, x, x0_pred)

        return x

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        DDIM update:
        x_{t-1} = sqrt(ā_{t-1})*x0_pred
                + sqrt(1 - ā_{t-1} - σ²) * (x_t - sqrt(ā_t)*x0_pred)/sqrt(1-ā_t)
                + σ_t * ε
        """
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev].to(device) if t_prev >= 0 \
                         else torch.ones(1, device=device)

        sigma_t = eta * (
            (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            * (1 - alpha_bar_t / alpha_bar_prev)
        ).sqrt()

        # Direction pointing to x_t
        noise_pred = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1 - alpha_bar_t).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma_t ** 2).clamp(min=0).sqrt() * noise_pred

        noise = sigma_t * torch.randn_like(x_t) if eta > 0 else 0.0
        x_prev = alpha_bar_prev.sqrt() * x0_pred + dir_xt + noise

        return x_prev

    def _double_cond(self, cond: dict, batch_size: int) -> dict:
        doubled = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                null = torch.zeros_like(v)
                doubled[k] = torch.cat([v, null], dim=0)
            else:
                doubled[k] = v
        return doubled

    @torch.no_grad()
    def encode_latents(
        self,
        x0: torch.Tensor,
        t: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM inversion: encode real image into noisy latent at time t."""
        if noise is None:
            noise = torch.randn_like(x0)
        return self.scheduler.add_noise(x0, noise, torch.tensor([t]))