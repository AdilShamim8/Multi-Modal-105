import torch
import torch.nn as nn
from typing import Tuple
from .encoder import ResidualBlock, AttentionBlock


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Decoder(nn.Module):
    """
    VAE Decoder: latent → image reconstruction.
    Mirror of Encoder with upsampling instead of downsampling.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]
        in_ch = channels[-1]

        # Input projection from latent
        self.conv_in = nn.Conv2d(latent_channels, in_ch, 3, padding=1)

        # Middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, dropout)

        # Upsampling stages (reversed channel list)
        self.up_blocks = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            stage = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if attention_resolutions and i in attention_resolutions:
                    stage.append(AttentionBlock(in_ch))
            self.up_blocks.append(stage)
            if i < len(channels) - 1:
                self.up_blocks.append(nn.ModuleList([UpsampleBlock(in_ch)]))

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            if isinstance(block, nn.ModuleList):
                for layer in block:
                    h = layer(h)

        h = self.act(self.norm_out(h))
        h = self.conv_out(h)
        return torch.tanh(h)       # (B, 3, H, W) in [-1, 1]