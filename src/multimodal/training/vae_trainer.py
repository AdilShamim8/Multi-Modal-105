
import lpips
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from multimodal.config.vae_config import VAEConfig
from multimodal.models.gans.discriminator import StyleGAN2Discriminator
from multimodal.models.vae.vae import AutoencoderKL
from multimodal.utils.checkpoint import CheckpointManager
from multimodal.utils.distributed import is_main_process
from multimodal.utils.logging_utils import MetricLogger


class VAETrainer:
    """
    Production VAE trainer with:
    - Perceptual (LPIPS) loss
    - KL regularization
    - Adversarial (patch discriminator) loss
    - Mixed precision
    - Distributed training
    """

    def __init__(self, config: VAEConfig, train_loader: DataLoader,
                 val_loader: DataLoader | None = None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)

        # Models
        self.vae = AutoencoderKL(
            in_channels=config.in_channels,
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout,
        ).to(self.device)

        self.discriminator = StyleGAN2Discriminator(
            img_resolution=config.image_size,
            img_channels=config.in_channels,
        ).to(self.device)

        # Perceptual loss
        self.perceptual = lpips.LPIPS(net="vgg").to(self.device)
        for p in self.perceptual.parameters():
            p.requires_grad_(False)

        # Optimizers
        self.opt_vae = torch.optim.AdamW(
            self.vae.parameters(),
            lr=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )
        self.opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )

        # Mixed precision
        self.scaler_vae = GradScaler(enabled=config.mixed_precision != "no")
        self.scaler_disc = GradScaler(enabled=config.mixed_precision != "no")

        # Utilities
        self.logger = MetricLogger(
            project=config.wandb_project,
            name=config.experiment_name,
            config=config.__dict__,
            use_wandb=config.use_wandb,
            output_dir=config.output_dir,
        )
        self.checkpoint_mgr = CheckpointManager(config.output_dir)
        self.global_step = 0

    def compute_vae_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        posterior,
        step: int,
    ) -> dict[str, torch.Tensor]:
        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(recon, x)

        # Perceptual loss
        # Normalize to [-1,1] → LPIPS expects this range
        perc_loss = self.perceptual(recon, x).mean()

        # KL loss
        kl_loss = posterior.kl()

        # Adversarial loss (generator side)
        adv_loss = torch.tensor(0.0, device=x.device)
        if step >= self.config.disc_start_step:
            fake_pred = self.discriminator(recon)
            adv_loss = F.softplus(-fake_pred).mean()

        total = (
            recon_loss
            + self.config.perceptual_weight * perc_loss
            + self.config.kl_weight * kl_loss
            + self.config.adversarial_weight * adv_loss
        )

        return {
            "total": total,
            "recon": recon_loss,
            "perceptual": perc_loss,
            "kl": kl_loss,
            "adv_g": adv_loss,
        }

    def compute_disc_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Hinge loss
        real_pred = self.discriminator(x.detach())
        fake_pred = self.discriminator(recon.detach())

        d_real = F.softplus(-real_pred).mean()
        d_fake = F.softplus(fake_pred).mean()

        # R1 penalty
        r1 = self.discriminator.r1_penalty(x)

        total = d_real + d_fake + 10.0 * r1

        return {
            "disc_total": total,
            "d_real": d_real,
            "d_fake": d_fake,
            "r1": r1,
        }

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        x = batch["image"].to(self.device)
        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        # ── VAE forward ──────────────────────────────────────────────────────
        self.opt_vae.zero_grad()
        with autocast(enabled=self.config.mixed_precision != "no", dtype=dtype):
            recon, posterior = self.vae(x)
            vae_losses = self.compute_vae_loss(x, recon, posterior, self.global_step)

        self.scaler_vae.scale(vae_losses["total"]).backward()
        self.scaler_vae.unscale_(self.opt_vae)
        torch.nn.utils.clip_grad_norm_(
            self.vae.parameters(), self.config.optimizer.max_grad_norm
        )
        self.scaler_vae.step(self.opt_vae)
        self.scaler_vae.update()

        # ── Discriminator forward ─────────────────────────────────────────────
        disc_losses = {}
        if self.global_step >= self.config.disc_start_step:
            self.opt_disc.zero_grad()
            with autocast(enabled=self.config.mixed_precision != "no", dtype=dtype):
                disc_losses = self.compute_disc_loss(x, recon)

            self.scaler_disc.scale(disc_losses["disc_total"]).backward()
            self.scaler_disc.unscale_(self.opt_disc)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.config.optimizer.max_grad_norm
            )
            self.scaler_disc.step(self.opt_disc)
            self.scaler_disc.update()

        self.global_step += 1
        return {k: v.item() for k, v in {**vae_losses, **disc_losses}.items()}

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.vae.train()
            self.discriminator.train()

            for batch in self.train_loader:
                metrics = self.train_step(batch)

                if self.global_step % self.config.log_every_n_steps == 0 and is_main_process():
                    self.logger.log(metrics, step=self.global_step)

                if self.global_step % self.config.save_every_n_steps == 0 and is_main_process():
                    self.checkpoint_mgr.save(
                        step=self.global_step,
                        model=self.vae,
                        optimizer=self.opt_vae,
                    )

            if is_main_process():
                print(f"Epoch {epoch} completed | step {self.global_step}")
