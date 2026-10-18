import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from multimodal.config.diffusion_config import DiffusionConfig
from multimodal.models.diffusion.dit import EMAModel
from multimodal.models.diffusion.forward_backward import DiffusionForwardProcess
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.models.vae.vae import AutoencoderKL
from multimodal.utils.checkpoint import CheckpointManager
from multimodal.utils.distributed import is_main_process


class DiffusionTrainer:
    """
    Production diffusion model trainer using HuggingFace Accelerate.
    Handles:
    - Mixed precision (bf16)
    - Gradient accumulation
    - EMA model tracking
    - min-SNR loss weighting
    - Distributed training
    - Periodic sampling for visual evaluation
    """

    def __init__(
        self,
        config: DiffusionConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        text_encoder: nn.Module | None = None,
        tokenizer=None,
        vae: AutoencoderKL | None = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Accelerate handles device placement + distributed + mixed precision
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir,
        )

        self.device = self.accelerator.device

        # Noise scheduler & forward process
        self.scheduler = NoiseScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        )
        self.forward_process = DiffusionForwardProcess(self.scheduler)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.epsilon,
        )

        # LR scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.optimizer.num_training_steps,
            eta_min=config.optimizer.learning_rate * config.optimizer.min_lr_ratio,
        )

        # Accelerate prepare
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(model, self.optimizer, train_loader, self.lr_scheduler)

        # EMA
        self.ema = EMAModel(
            self.accelerator.unwrap_model(self.model),
            decay=config.ema_decay,
        )

        # Frozen encoders (not wrapped in accelerate)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        if self.text_encoder:
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        if self.vae:
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)

        # Utilities
        self.checkpoint_mgr = CheckpointManager(config.output_dir, max_checkpoints=5)
        self.global_step = 0
        self.snr_gamma = config.snr_gamma

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text to embeddings."""
        with torch.no_grad():
            out = self.text_encoder(input_ids, output_hidden_states=True)
            return out.last_hidden_state

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents."""
        with torch.no_grad():
            return self.vae.encode_to_latent(images)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()

        with self.accelerator.accumulate(self.model):
            # ── Encode inputs ──────────────────────────────────────────────
            if self.vae is not None and "image" in batch:
                latents = self.encode_image(batch["image"])
            else:
                latents = batch["latent"]

            text_embeds = None
            if self.text_encoder is not None and "input_ids" in batch:
                text_embeds = self.encode_text(batch["input_ids"])

            # ── Sample timesteps + add noise ───────────────────────────────
            B = latents.shape[0]
            t = self.forward_process.sample_timesteps(B, latents.device)
            noise = torch.randn_like(latents)
            x_t, noise = self.forward_process.q_sample(latents, t, noise)

            # ── Model forward ──────────────────────────────────────────────
            model_kwargs = {}
            if text_embeds is not None:
                model_kwargs["text_embeds"] = text_embeds
            if "class_labels" in batch:
                model_kwargs["y"] = batch["class_labels"]

            model_output = self.model(x_t, t, **model_kwargs)

            # Handle learned sigma (DiT)
            if model_output.shape[1] == latents.shape[1] * 2:
                model_output, _ = model_output.chunk(2, dim=1)

            # ── Loss ───────────────────────────────────────────────────────
            loss = self.forward_process.compute_loss(
                model_output=model_output,
                noise=noise,
                x_start=latents,
                x_t=x_t,
                t=t,
                snr_gamma=self.snr_gamma,
            )

            # ── Backward ───────────────────────────────────────────────────
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.max_grad_norm,
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # EMA update (on main process)
                self.ema.update(self.accelerator.unwrap_model(self.model))
                self.global_step += 1

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.global_step,
        }

    def train(self):
        cfg = self.config
        self.accelerator.print(f"Starting diffusion training for {cfg.num_train_steps} steps")

        data_iter = iter(self.train_loader)

        while self.global_step < cfg.num_train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            if self.global_step % cfg.log_every_n_steps == 0 and is_main_process():
                self.accelerator.log(metrics, step=self.global_step)

            if self.global_step % cfg.save_every_n_steps == 0 and is_main_process():
                unwrapped = self.accelerator.unwrap_model(self.model)
                self.checkpoint_mgr.save(
                    step=self.global_step,
                    model=unwrapped,
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    metadata={"loss": metrics["loss"]},
                )

        self.accelerator.end_training()
        self.accelerator.print("Training complete!")
