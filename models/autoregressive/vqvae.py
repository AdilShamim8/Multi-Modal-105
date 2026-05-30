import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    VQ-VAE vector quantization layer with EMA updates.
    """

    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_ema: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())
            self.ema_decay = ema_decay

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, H, W) continuous embeddings
        Returns:
            z_q: quantized embeddings
            loss: VQ loss
            encoding_indices: (B, H*W) integer codes
        """
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")  # (N, D)

        # Distances to codebook
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )  # (N, K)

        encoding_indices = distances.argmin(dim=1)         # (N,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        z_q_flat = self.embedding(encoding_indices)         # (N, D)

        # EMA codebook update
        if self.use_ema and self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay
                )
                dw = encodings.T @ z_flat                  # (K, D)
                self.ema_w.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.num_embeddings * 1e-5) * n
                )
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        z_q = rearrange(z_q_flat, "(b h w) d -> b d h w", b=B, h=H, w=W)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Losses
        if self.use_ema:
            loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        else:
            loss = (
                F.mse_loss(z_q.detach(), z)
                + self.commitment_cost * F.mse_loss(z_q, z.detach())
            )

        perplexity = torch.exp(
            -torch.sum(
                encodings.mean(0) * torch.log(encodings.mean(0) + 1e-10)
            )
        )

        return z_q_st, loss, encoding_indices.view(B, H * W)


class VQVAE(nn.Module):
    """
    VQ-VAE: discrete autoencoder for auto-regressive image generation.
    Used as the tokenizer for models like VQ-GAN, DALL-E, etc.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        num_residual_blocks: int = 2,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        from ..vae.encoder import ResidualBlock

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)],
        )
        self.pre_quant = nn.Conv2d(hidden_channels, embedding_dim, 1)

        # VQ layer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder
        self.post_quant = nn.Conv2d(embedding_dim, hidden_channels, 1)
        self.decoder = nn.Sequential(
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)],
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 2, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pre_quant(self.encoder(x))
        z_q, vq_loss, indices = self.vq(z)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant(z_q))

    def forward(self, x: torch.Tensor):
        z_q, vq_loss, indices = self.encode(x)
        recon = self.decode(z_q)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + vq_loss
        return recon, total_loss, {"recon_loss": recon_loss, "vq_loss": vq_loss}

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, indices = self.encode(x)
        return indices

    def decode_from_indices(self, indices: torch.Tensor, h: int, w: int) -> torch.Tensor:
        z_q = self.vq.embedding(indices).view(-1, h, w, self.vq.embedding_dim)
        z_q = rearrange(z_q, "b h w d -> b d h w")
        return self.decode(z_q)