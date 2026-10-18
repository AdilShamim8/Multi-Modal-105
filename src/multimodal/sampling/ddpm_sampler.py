from collections.abc import Callable

import torch

from multimodal.models.diffusion.noise_scheduler import NoiseScheduler


#: Denoising models are invoked as ``model(x, t, **conditioning)``. Production
#: pipelines pass CFG wrapper callables, trainers pass raw nn.Modules — both
#: satisfy this calling convention, so the sampler accepts either.
Denoiser = Callable[..., torch.Tensor] | torch.nn.Module


class DDPMSampler:
    """
    DDPM ancestral sampling (Ho et al. 2020).
    Full T-step reverse process: slow but exact.
    """

    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        model: Denoiser,
        shape: tuple,
        device: str | torch.device,
        conditioning: dict | None = None,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        clip_denoised: bool = True,
        show_progress: bool = True,
        callback: Callable | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full DDPM reverse diffusion chain.
        Args:
            model:    denoising model (UNet or DiT) or CFG-wrapping callable
            shape:    output shape (B, C, H, W)
            device:   compute device
            conditioning: dict with keys like 'text_embeds', 'y', etc.
            guidance_scale: CFG scale (1.0 = no guidance)
            latents:  optional initial state (img2img / partial denoising);
                      truncated to the remaining timestep window
        """
        self.scheduler.set_timesteps(num_inference_steps, device=str(device))
        timesteps = self.scheduler.timesteps

        # Start from caller-provided state (img2img) or pure noise
        x = latents if latents is not None else torch.randn(*shape, device=device)

        cond = conditioning or {}
        do_cfg = guidance_scale > 1.0

        from tqdm import tqdm

        pbar = tqdm(timesteps, desc="DDPM sampling") if show_progress else timesteps

        for i, t in enumerate(pbar):
            t_batch = t.expand(shape[0])

            # Model prediction
            if do_cfg:
                x_in = torch.cat([x, x], dim=0)
                t_in = t_batch.repeat(2)
                # Conditional + unconditional
                model_out = model(x_in, t_in, **self._double_cond(cond, shape[0]))
                cond_out, uncond_out = model_out.chunk(2, dim=0)
                model_output = uncond_out + guidance_scale * (cond_out - uncond_out)
            else:
                model_output = model(x, t_batch, **cond)

            # Get x_0 prediction
            x0_pred = self.scheduler.get_x0_prediction(model_output, x, t_batch)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Compute posterior mean and sample
            mean, _, log_var = self.scheduler.q_posterior(x0_pred, x, t_batch)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = mean + (0.5 * log_var).exp() * noise

            if callback is not None:
                callback(i, t, x, x0_pred)

        return x

    def _double_cond(self, cond: dict, batch_size: int) -> dict:
        """Duplicate conditioning for CFG (cond + uncond)."""
        doubled = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                null = torch.zeros_like(v)
                doubled[k] = torch.cat([v, null], dim=0)
            else:
                doubled[k] = v
        return doubled
