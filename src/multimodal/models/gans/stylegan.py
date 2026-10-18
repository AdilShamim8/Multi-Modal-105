"""StyleGAN2 system: generator + discriminator + losses + sampling utilities.

Fills the previously empty placeholder. This module is the *system-level*
composition layer: raw networks live in `generator.py` / `discriminator.py`,
while everything needed to actually train and sample with them (adversarial
losses, lazy R1 penalty, the truncation trick, generator EMA) lives here.
`GANTrainer` (multimodal.training.gan_trainer) consumes this class.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import StyleGAN2Generator
from .discriminator import StyleGAN2Discriminator


class StyleGAN2(nn.Module):
    """Bundled StyleGAN2 system with production training helpers."""

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_resolution: int = 256,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_mapping_layers: int = 8,
        loss_type: str = "softplus",
        r1_gamma: float = 10.0,
        lazy_reg_interval: int = 16,
    ):
        super().__init__()
        if loss_type not in ("softplus", "hinge", "wgan-gp"):
            raise ValueError(f"unsupported loss_type: {loss_type}")
        self.generator = StyleGAN2Generator(
            z_dim=z_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            channel_base=channel_base,
            channel_max=channel_max,
            num_mapping_layers=num_mapping_layers,
        )
        self.discriminator = StyleGAN2Discriminator(
            img_resolution=img_resolution,
            img_channels=img_channels,
            channel_base=channel_base,
            channel_max=channel_max,
        )
        self.loss_type = loss_type
        self.r1_gamma = r1_gamma
        self.lazy_reg_interval = lazy_reg_interval
        # Running mean of w for the truncation trick (updated in eval phases).
        self.register_buffer("w_avg", torch.zeros(w_dim))
        self.register_buffer("w_avg_n", torch.zeros(()))

    # ── Adversarial losses ────────────────────────────────────────────────
    def d_loss(self, real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "softplus":  # logistic (non-saturating, symmetric)
            return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
        if self.loss_type == "hinge":
            return F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
        # wgan-gp handled by GANTrainer (needs gradient penalty term)
        return (fake_logit - real_logit).mean()

    def g_loss(self, fake_logit: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "softplus":
            return F.softplus(-fake_logit).mean()
        if self.loss_type == "hinge":
            return (-fake_logit).mean()
        return (-fake_logit).mean()

    def gradient_penalty(
        self, real_images: torch.Tensor, fake_images: torch.Tensor, lam: float = 10.0
    ) -> torch.Tensor:
        """WGAN-GP interpolates between real and fake; R1 users call the
        discriminator's own `r1_penalty` on the lazy interval instead."""
        eps = torch.rand(real_images.shape[0], 1, 1, 1, device=real_images.device)
        interp = (eps * real_images + (1 - eps) * fake_images).requires_grad_(True)
        logit = self.discriminator(interp)
        grad = torch.autograd.grad(logit.sum(), interp, create_graph=True)[0]
        return lam * (grad.norm(2, dim=1) - 1).pow(2).mean()

    def r1_lazy(self, real_images: torch.Tensor, step: int) -> torch.Tensor | None:
        """Lazy R1: computed only every `lazy_reg_interval` critic steps,
        amortized by the interval (StyleGAN2 Appendix B)."""
        if self.r1_gamma <= 0 or step % self.lazy_reg_interval != 0:
            return None
        return (
            self.discriminator.r1_penalty(real_images)
            * self.r1_gamma
            / self.lazy_reg_interval
        )

    # ── Sampling ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        batch: int = 1,
        seed: int | None = None,
        truncation_psi: float = 1.0,
        noise_mode: str = "random",
        device: str = "cpu",
    ) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        z = torch.randn(batch, self.generator.z_dim, generator=gen).to(device)
        return self.sample_from_z(z, truncation_psi=truncation_psi, noise_mode=noise_mode)

    @torch.no_grad()
    def sample_from_z(
        self, z: torch.Tensor, truncation_psi: float = 1.0, noise_mode: str = "random"
    ) -> torch.Tensor:
        if truncation_psi != 1.0 and float(self.w_avg_n) > 0:
            # Truncation trick: pull w toward the running mean.
            w = self.generator.mapping(z)
            w = self.w_avg + truncation_psi * (w - self.w_avg)
            ws = w.unsqueeze(1).expand(-1, self.generator.num_ws, -1)
            # Reuse the generator forward but override ws via a small shim.
            saved = self.generator.mapping
            self.generator.mapping = lambda x: w  # type: ignore[assignment]
            try:
                img = self.generator(z, truncation_psi=1.0, noise_mode=noise_mode)
            finally:
                self.generator.mapping = saved  # type: ignore[assignment]
            return img
        return self.generator(z, truncation_psi=truncation_psi, noise_mode=noise_mode)

    @torch.no_grad()
    def update_w_avg(self, decay: float = 0.995) -> None:
        """Track the running mean of w (call on synthesis batches)."""
        z = torch.randn(8, self.generator.z_dim, device=self.w_avg.device)
        w = self.generator.mapping(z).mean(dim=0)
        n = float(self.w_avg_n)
        self.w_avg.copy_(self.w_avg * (n / (n + 1)) + w * (1 / (n + 1)))
        self.w_avg_n += 1

    @staticmethod
    def ema_copy(module: nn.Module, beta: float = 0.999) -> nn.Module:
        """Independent EMA shadow of a module (used for evaluation exports)."""
        shadow = copy.deepcopy(module).eval()
        for p in shadow.parameters():
            p.requires_grad_(False)
        return shadow

    @staticmethod
    def ema_update(shadow: nn.Module, live: nn.Module, beta: float = 0.999) -> None:
        with torch.no_grad():
            for ps, pl in zip(shadow.parameters(), live.parameters(), strict=False):
                ps.mul_(beta).add_(pl.detach(), alpha=1 - beta)

    @staticmethod
    def stabilize_images(img: torch.Tensor) -> torch.Tensor:
        """Map raw synthesis output to displayable [0, 1] range."""
        return img.clamp(-1, 1).add(1).div(2)
