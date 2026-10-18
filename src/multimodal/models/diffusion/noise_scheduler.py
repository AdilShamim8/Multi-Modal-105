import numpy as np
import torch


class NoiseScheduler:
    """
    Unified noise scheduler supporting:
    - Linear, cosine, scaled_linear beta schedules
    - DDPM / DDIM / DPM-Solver compatible precomputed buffers
    - v-prediction, epsilon, sample prediction types
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",  # "linear"|"cosine"|"scaled_linear"
        prediction_type: str = "v_prediction",  # "epsilon"|"sample"|"v_prediction"
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

        # ── Build beta schedule ───────────────────────────────────────────────
        self.betas = self._build_betas(num_train_timesteps, beta_start, beta_end, beta_schedule)

        # ── Precompute diffusion constants ────────────────────────────────────
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # sqrt(alpha_bar) — used to scale x0 in forward process
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        # sqrt(1 - alpha_bar) — used to scale noise in forward process
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        # log(1 - alpha_bar)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        # sqrt(1 / alpha_bar) — used in DDPM posterior
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        # sqrt(1 / alpha_bar - 1)
        self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1.0).sqrt()

        # ── DDPM posterior q(x_{t-1} | x_t, x_0) ────────────────────────────
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )

        # SNR (signal-to-noise ratio) for min-SNR weighting
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

        # Timesteps
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1).long()

    # ─────────────────────────────────────────────────────────────────────────
    # Beta schedule builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_betas(
        self,
        T: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        elif schedule == "scaled_linear":
            # SD-style: linear in sqrt space
            return torch.linspace(beta_start**0.5, beta_end**0.5, T, dtype=torch.float64) ** 2

        elif schedule == "cosine":
            # Nichol & Dhariwal (improved DDPM)
            steps = T + 1
            t = torch.linspace(0, T, steps, dtype=torch.float64)
            f = torch.cos((t / T + 0.008) / 1.008 * np.pi / 2.0) ** 2
            alphas_cumprod = f / f[0]
            betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, min=0.0, max=0.999)

        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    # ─────────────────────────────────────────────────────────────────────────
    # Forward process q(x_t | x_0)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values from 1-D tensor at timestep indices t, broadcast to shape."""
        device = t.device
        arr = arr.to(device)
        out = arr.gather(0, t.long())
        return out.view(*([t.shape[0]] + [1] * (len(shape) - 1))).float()

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(ā_t)*x0, (1-ā_t)*I)
        x_t = sqrt(ā_t)*x_0 + sqrt(1-ā_t)*ε
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def get_velocity(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """v = sqrt(ā_t)*ε - sqrt(1-ā_t)*x_0  (v-prediction target)."""
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha_bar * noise - sqrt_one_minus * x_start

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior q(x_{t-1} | x_t, x_0)
    # ─────────────────────────────────────────────────────────────────────────

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance:
        μ_q = coef1*x_0 + coef2*x_t
        """
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction conversion utilities
    # ─────────────────────────────────────────────────────────────────────────

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """x_0 = (x_t - sqrt(1-ā_t)*ε) / sqrt(ā_t)."""
        recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return recip * x_t - recipm1 * noise_pred

    def predict_x0_from_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> torch.Tensor:
        """x_0 = sqrt(ā_t)*x_t - sqrt(1-ā_t)*v."""
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_bar * x_t - sqrt_one_minus * v_pred

    def predict_noise_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_pred: torch.Tensor,
    ) -> torch.Tensor:
        """ε = (x_t - sqrt(ā_t)*x_0) / sqrt(1-ā_t)."""
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus

    def get_x0_prediction(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert model output to x_0 prediction based on prediction_type."""
        if self.prediction_type == "epsilon":
            return self.predict_x0_from_noise(x_t, t, model_output)
        elif self.prediction_type == "v_prediction":
            return self.predict_x0_from_velocity(x_t, t, model_output)
        elif self.prediction_type == "sample":
            return model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding (Imagen)."""
        dtype = sample.dtype
        batch = sample.shape[0]
        s = torch.quantile(
            sample.reshape(batch, -1).abs().float(),
            self.dynamic_thresholding_ratio,
            dim=1,
        )
        s = torch.clamp(s, min=self.sample_max_value)
        s = s.view(batch, *([1] * (sample.ndim - 1)))
        sample = sample.clamp(-s, s) / s
        return sample.to(dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # SNR-based loss weighting (min-SNR, Hang et al. 2023)
    # ─────────────────────────────────────────────────────────────────────────

    def get_loss_weight(
        self,
        timesteps: torch.Tensor,
        snr_gamma: float | None = 5.0,
    ) -> torch.Tensor:
        snr_t = self.snr.to(timesteps.device).gather(0, timesteps)
        if snr_gamma is None:
            return torch.ones_like(snr_t)
        # min-SNR weighting
        if self.prediction_type == "v_prediction":
            snr_t = snr_t + 1.0
        weights = torch.clamp(snr_t, max=snr_gamma) / snr_t
        return weights

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        """Set inference timesteps (subset of training timesteps)."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            (torch.arange(0, num_inference_steps) * step_ratio).round().flip(0).long().to(device)
        )
        self.num_inference_steps = num_inference_steps
