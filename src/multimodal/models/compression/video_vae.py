from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─────────────────────────────────────────────────────────────────────────────
# 3D Causal Convolution (for temporal causality)
# ─────────────────────────────────────────────────────────────────────────────


class CausalConv3d(nn.Module):
    """
    3D convolution with causal padding in time dimension.
    Ensures no future frame information leaks into past frames.
    Used in video VAEs (e.g., Open-Sora, CogVideo).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],  # (T, H, W)
        stride: tuple[int, int, int] = (1, 1, 1),
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kt, kh, kw = kernel_size
        self.temporal_pad = (kt - 1) * dilation  # causal: only pad past
        self.spatial_pad_h = kh // 2
        self.spatial_pad_w = kw // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(dilation, 1, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # Causal temporal padding (pad only left = past)
        x = F.pad(
            x,
            (
                self.spatial_pad_w,
                self.spatial_pad_w,  # W pad
                self.spatial_pad_h,
                self.spatial_pad_h,  # H pad
                self.temporal_pad,
                0,  # T causal pad
            ),
        )
        return self.conv(x)


class SpatialDownsample(nn.Module):
    """2x spatial downsampling using strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — apply spatial conv per frame
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv(x)
        _, _, H2, W2 = x.shape
        return rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)


class SpatialUpsample(nn.Module):
    """2x spatial upsampling (nearest + conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        _, _, H2, W2 = x.shape
        return rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)


class TemporalDownsample(nn.Module):
    """2x temporal downsampling using strided causal conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, (3, 1, 1), stride=(2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """2x temporal upsampling (interpolate + conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, (3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        # Temporal upsample via linear interpolation
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        x = F.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="nearest")
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3D ResBlock (spatial + temporal)
# ─────────────────────────────────────────────────────────────────────────────


class VideoResBlock(nn.Module):
    """
    3D residual block for video processing.
    Uses spatial 2D conv + temporal 1D conv (factorized).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Spatial path
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.spatial_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Temporal path (causal 1D conv along time)
        self.temporal_conv = CausalConv3d(out_channels, out_channels, (3, 1, 1))
        self.temporal_norm = nn.GroupNorm(32, out_channels)

        self.act = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial operations frame-by-frame."""
        B, C, T, H, W = x.shape
        x_flat = rearrange(x, "b c t h w -> (b t) c h w")

        h = self.act(self.norm1(x_flat))
        h = self.spatial_conv(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.spatial_conv2(h)
        skip = self.shortcut(x_flat)
        out = h + skip

        return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        h = self._apply_spatial(x)

        # Temporal refinement
        residual = h
        h_t = self.act(self.temporal_norm(h))
        h_t = self.temporal_conv(h_t)
        return h_t + residual


# ─────────────────────────────────────────────────────────────────────────────
# Video Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────


class VideoEncoder(nn.Module):
    """
    Video VAE encoder:
    - 4x spatial downsampling (H,W → H/8, W/8)
    - 4x temporal downsampling (T → T/4)
    - Outputs latent distribution parameters (mu, logvar)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample_times: int = 2,  # 2^2 = 4x temporal compression
        spatial_downsample_times: int = 3,  # 2^3 = 8x spatial compression
        dropout: float = 0.0,
        double_z: bool = True,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]

        # Input projection
        self.conv_in = CausalConv3d(in_channels, channels[0], (1, 3, 3))

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        spatial_ds_count = 0
        temporal_ds_count = 0

        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(VideoResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch

            # Spatial downsampling
            if i < len(channels) - 1 and spatial_ds_count < spatial_downsample_times:
                block.append(SpatialDownsample(in_ch))
                spatial_ds_count += 1

            # Temporal downsampling
            if i < len(channels) - 1 and temporal_ds_count < temporal_downsample_times:
                block.append(TemporalDownsample(in_ch))
                temporal_ds_count += 1

            self.down_blocks.append(block)

        # Middle
        self.mid_block = VideoResBlock(in_ch, in_ch, dropout)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        out_ch_final = 2 * latent_channels if double_z else latent_channels
        self.conv_out = CausalConv3d(in_ch, out_ch_final, (1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → (B, 2*latent_ch, T', H', W')"""
        h = self.conv_in(x)

        # ModuleList is non-generic in torch's stubs (Iterator[Module]); the
        # cast documents the per-scale ModuleList groups. Zero runtime cost.
        for block in cast("list[nn.ModuleList]", self.down_blocks):
            for layer in block:
                h = layer(h)

        h = self.mid_block(h)
        h = self.act(self.norm_out(h))
        return self.conv_out(h)


class VideoDecoder(nn.Module):
    """
    Video VAE decoder:
    Mirror of VideoEncoder — upsamples latent back to video.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample_times: int = 2,
        spatial_upsample_times: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [base_channels * m for m in reversed(channel_multipliers)]
        in_ch = channels[0]

        # Input from latent
        self.conv_in = CausalConv3d(latent_channels, in_ch, (1, 3, 3))

        # Middle
        self.mid_block = VideoResBlock(in_ch, in_ch, dropout)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        spatial_us_count = 0
        temporal_us_count = 0

        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(VideoResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch

            # Spatial upsampling
            if i < len(channels) - 1 and spatial_us_count < spatial_upsample_times:
                block.append(SpatialUpsample(in_ch))
                spatial_us_count += 1

            # Temporal upsampling
            if i < len(channels) - 1 and temporal_us_count < temporal_upsample_times:
                block.append(TemporalUpsample(in_ch))
                temporal_us_count += 1

            self.up_blocks.append(block)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv_out = CausalConv3d(in_ch, out_channels, (1, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_ch, T', H', W') → (B, 3, T, H, W)"""
        h = self.conv_in(z)
        h = self.mid_block(h)

        for block in cast("list[nn.ModuleList]", self.up_blocks):
            for layer in block:
                h = layer(h)

        h = self.act(self.norm_out(h))
        return torch.tanh(self.conv_out(h))


# ─────────────────────────────────────────────────────────────────────────────
# Video AutoencoderKL
# ─────────────────────────────────────────────────────────────────────────────


class VideoAutoencoderKL(nn.Module):
    """
    KL-regularized Video VAE.
    Compresses (B, 3, T, H, W) → (B, 16, T/4, H/8, W/8) latents.
    Used as the compression backbone for all T2V latent diffusion models.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_compression: int = 4,  # T / 4
        spatial_compression: int = 8,  # H,W / 8
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.scaling_factor = scaling_factor

        temporal_ds_times = int(np.log2(temporal_compression))
        spatial_ds_times = int(np.log2(spatial_compression))

        self.encoder = VideoEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_downsample_times=temporal_ds_times,
            spatial_downsample_times=spatial_ds_times,
            double_z=True,
        )

        self.decoder = VideoDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_upsample_times=temporal_ds_times,
            spatial_upsample_times=spatial_ds_times,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)

    def encode(self, x: torch.Tensor):
        """x: (B, 3, T, H, W) → DiagonalGaussian3D"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mu, logvar = moments.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=[1, 2, 3, 4]).mean()

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.sample_latent(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, self.kl_loss(mu, logvar)

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.sample_latent(mu, logvar)
        return z * self.scaling_factor

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z / self.scaling_factor)
