
import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class DiagonalGaussian(nn.Module):
    """Reparameterization trick distribution."""

    def __init__(self, parameters: torch.Tensor):
        super().__init__()
        mu, logvar = parameters.chunk(2, dim=1)
        self.mu = mu
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mu

    def kl(self) -> torch.Tensor:
        """KL divergence to standard normal N(0, I)."""
        return 0.5 * (
            self.mu.pow(2) + self.logvar.exp() - 1.0 - self.logvar
        ).sum(dim=[1, 2, 3]).mean()


class AutoencoderKL(nn.Module):
    """
    Latent Diffusion VAE (KL-regularized autoencoder).
    Encodes images to 4-channel latent space with 8x spatial downsampling.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()

        # Pull params from kwargs (config accepted for API compatibility)
        in_channels      = kwargs.get("in_channels", 3)
        latent_channels  = kwargs.get("latent_channels", 4)
        base_channels    = kwargs.get("base_channels", 128)
        ch_mult          = kwargs.get("channel_multipliers", (1, 2, 4, 8))
        num_res_blocks   = kwargs.get("num_res_blocks", 2)
        attn_res         = kwargs.get("attention_resolutions", (16,))
        dropout          = kwargs.get("dropout", 0.0)

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=ch_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attn_res,
            dropout=dropout,
            double_z=True,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=ch_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attn_res,
            dropout=dropout,
        )

        # Quant conv (refine distribution parameters)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        # Scaling factor (SD uses 0.18215)
        self.scaling_factor = 0.18215

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussian(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(z)
        return recon, posterior

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to scaled latent (ready for diffusion)."""
        posterior = self.encode(x)
        z = posterior.sample()
        return z * self.scaling_factor

    def decode_from_latent(self, z_scaled: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent back to image."""
        z = z_scaled / self.scaling_factor
        return self.decode(z)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Batch encode images (no grad, for inference)."""
        return self.encode_to_latent(images)
