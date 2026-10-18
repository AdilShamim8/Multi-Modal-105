"""GAN trainer for the StyleGAN2 family.

Fills the previously empty placeholder. Implements the production training
loop for adversarial models: alternating D/G steps, lazy R1 regularization,
WGAN-GP support, gradient clipping, EMA shadow of the generator, periodic
checkpointing and sample grids.

Consumes: `multimodal.config.gan_config.GANConfig`,
`multimodal.models.gans.stylegan.StyleGAN2`.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multimodal.config.gan_config import GANConfig
from multimodal.models.gans.stylegan import StyleGAN2
from multimodal.utils.checkpoint import CheckpointManager
from multimodal.utils.distributed import is_main_process
from multimodal.utils.logging_utils import MetricLogger
from multimodal.utils.seeds import seed_everything


class GANTrainer:
    """Production GAN training loop with alternating optimization."""

    def __init__(
        self,
        gan: StyleGAN2,
        config: GANConfig,
        train_loader: DataLoader,
        device: str | None = None,
        checkpoint_dir: str = "./checkpoints/gan",
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gan = gan.to(self.device)
        self.train_loader = train_loader
        self.step = 0

        self.optim_g = torch.optim.Adam(
            self.gan.generator.parameters(),
            lr=config.gen_lr,
            betas=(config.beta1, config.beta2),
        )
        self.optim_d = torch.optim.Adam(
            self.gan.discriminator.parameters(),
            lr=config.disc_lr,
            betas=(config.beta1, config.beta2),
        )

        self.ema_gen = StyleGAN2.ema_copy(self.gan.generator, config.ema_beta)
        self.checkpoints = CheckpointManager(
            output_dir=checkpoint_dir,
            max_checkpoints=3,
        )
        self.metrics = MetricLogger(
            project=config.wandb_project if config.use_wandb else "multimodal-agent",
            name=f"{config.experiment_name}-gan",
            config={"gan": vars(config)},
            use_wandb=config.use_wandb,
            output_dir=config.output_dir,
        )

    # ── Single optimization round ─────────────────────────────────────────
    def train_step(self, real_batch: torch.Tensor) -> dict[str, float]:
        cfg = self.config
        real = real_batch.to(self.device, non_blocking=True)
        B = real.shape[0]
        z = torch.randn(B, cfg.z_dim, device=self.device)

        # ── Discriminator step ────────────────────────────────────────────
        self.optim_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = self.gan.generator(z)
        real_logit = self.gan.discriminator(real)
        fake_logit = self.gan.discriminator(fake)
        d_loss = self.gan.d_loss(real_logit, fake_logit)

        r1 = self.gan.r1_lazy(real, step=self.step)
        if r1 is not None:
            d_loss = d_loss + r1
        if cfg.loss_type == "wgan-gp":
            d_loss = d_loss + self.gan.gradient_penalty(real, fake.detach(), cfg.gp_lambda)

        d_loss.backward()
        nn.utils.clip_grad_norm_(self.gan.discriminator.parameters(), cfg.max_grad_norm)
        self.optim_d.step()

        # ── Generator step ────────────────────────────────────────────────
        self.optim_g.zero_grad(set_to_none=True)
        z = torch.randn(B, cfg.z_dim, device=self.device)
        fake = self.gan.generator(z)
        g_loss = self.gan.g_loss(self.gan.discriminator(fake))
        g_loss.backward()
        nn.utils.clip_grad_norm_(self.gan.generator.parameters(), cfg.max_grad_norm)
        self.optim_g.step()

        StyleGAN2.ema_update(self.ema_gen, self.gan.generator, cfg.ema_beta)
        self.step += 1

        return {
            "gan/d_loss": float(d_loss.detach()),
            "gan/g_loss": float(g_loss.detach()),
            "gan/r1": float(r1.detach()) if r1 is not None else 0.0,
        }

    # ── Sampling for visual regression checks ─────────────────────────────
    @torch.no_grad()
    def generate_samples(self, n: int = 8, truncation_psi: float = 0.7) -> torch.Tensor:
        grid: list[torch.Tensor] = []
        for _ in range(0, n, 4):
            batch = min(4, n - len(grid) * 4)
            grid.append(self.gan.sample(batch, truncation_psi=truncation_psi, device=self.device))
        return StyleGAN2.stabilize_images(torch.cat(grid, dim=0))

    # ── Full loop ─────────────────────────────────────────────────────────
    def train(self) -> None:
        cfg = self.config
        seed_everything(cfg.seed)
        epoch = 0
        while self.step < cfg.num_train_steps:
            epoch += 1
            for batch in self.train_loader:
                images = batch[0] if isinstance(batch, list | tuple) else batch
                stats = self.train_step(images)
                if self.step % cfg.log_every_n_steps == 0 and is_main_process():
                    self.metrics.log(stats, step=self.step)
                if self.step % cfg.eval_every_n_steps == 0 and is_main_process():
                    self.gan.update_w_avg()
                    self.metrics.log_image("samples", self.generate_samples(), step=self.step)
                if self.step % cfg.save_every_n_steps == 0 and is_main_process():
                    self._save()
                if self.step >= cfg.num_train_steps:
                    break
        if is_main_process():
            self._save()
            self.metrics.finish()

    def _save(self) -> None:
        self.checkpoints.save(
            step=self.step,
            model=self.gan.generator,
            optimizer=self.optim_g,
            ema_model=self.ema_gen,
            metadata={"gan": asdict(self.config)},
            extra_states={
                "discriminator": self.gan.discriminator.state_dict(),
                "optim_d": self.optim_d.state_dict(),
            },
        )
