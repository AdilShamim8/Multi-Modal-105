import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        def reshape(t):
            return rearrange(t, "b (nh hd) h w -> b nh (h w) hd",
                             nh=self.num_heads, hd=self.head_dim)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention (Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = rearrange(attn_out, "b nh (h w) hd -> b (nh hd) h w",
                             h=H, w=W)
        return x + self.proj(attn_out)


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Strided conv instead of pooling (learnable downsampling)
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """
    VAE Encoder: image → latent distribution parameters (mu, logvar).
    Downsamples by factor of 2^len(channel_multipliers) total.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        double_z: bool = True,                # output mu + logvar
    ):
        super().__init__()
        self.latent_channels = latent_channels
        channels = [base_channels * m for m in channel_multipliers]

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Downsampling stages
        self.down_blocks = nn.ModuleList()
        current_res = None   # tracked symbolically; resolution checked at fwd
        in_ch = channels[0]

        for i, out_ch in enumerate(channels):
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if attention_resolutions and i in attention_resolutions:
                    stage.append(AttentionBlock(in_ch))
            self.down_blocks.append(stage)
            if i < len(channels) - 1:
                self.down_blocks.append(nn.ModuleList([DownsampleBlock(in_ch)]))

        # Middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, dropout)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        out_channels = 2 * latent_channels if double_z else latent_channels
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):
                for layer in block:
                    h = layer(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.act(self.norm_out(h))
        h = self.conv_out(h)
        return h                   # (B, 2*latent_channels, H/f, W/f)