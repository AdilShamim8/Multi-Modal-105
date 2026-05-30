import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .noise_scheduler import NoiseScheduler


class DiffusionForwardProcess:
    """
    Implements the forward (noising) process of diffusion models.
    q(x_1:T | x_0) = prod_t q(x_t | x_{t-1})
    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)*x_{t-1}, β_t*I)
    """

    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        low: int = 0,
        high: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample random timesteps uniformly for training."""
        high = high or self.scheduler.num_train_timesteps
        return torch.randint(low, high, (batch_size,), device=device).long()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t given x_0 and timestep t.
        Returns (x_t, noise) — noise is needed for loss computation.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.scheduler.add_noise(x_start, noise, t)
        return x_t, noise

    def compute_loss(
        self,
        model_output: torch.Tensor,
        noise: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        snr_gamma: Optional[float] = 5.0,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss with optional SNR weighting.
        Supports epsilon, v-prediction, and sample prediction targets.
        """
        # Build target based on prediction type
        if self.scheduler.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x_start, noise, t)
        elif self.scheduler.prediction_type == "sample":
            target = x_start
        else:
            raise ValueError(f"Unknown prediction_type: {self.scheduler.prediction_type}")

        # Element-wise MSE
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # (B,)

        # SNR weighting
        weights = self.scheduler.get_loss_weight(t, snr_gamma=snr_gamma)
        loss = (loss * weights).mean()

        return loss


class DiffusionBackwardProcess:
    """
    Implements the reverse (denoising) process.
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    """

    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler

    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute posterior mean and variance of p(x_{t-1} | x_t).
        """
        # Get x_0 prediction
        x0_pred = self.scheduler.get_x0_prediction(model_output, x_t, t)

        if clip_denoised:
            if self.scheduler.thresholding:
                x0_pred = self.scheduler._threshold_sample(x0_pred)
            else:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Compute posterior
        mean, variance, log_variance = self.scheduler.q_posterior(x0_pred, x_t, t)

        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
            "x0_pred": x0_pred,
        }

    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        return_x0: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        One step of DDPM reverse process:
        x_{t-1} ~ p_θ(x_{t-1} | x_t)
        """
        out = self.p_mean_variance(model_output, x_t, t, clip_denoised)

        # Only add noise if t > 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))

        x_prev = out["mean"] + nonzero_mask * (0.5 * out["log_variance"]).exp() * noise

        if return_x0:
            return x_prev, out["x0_pred"]
        return x_prev