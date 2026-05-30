import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP projection."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = t.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(t)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention for text conditioning
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Multi-head cross-attention for text → spatial conditioning."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.norm_q = nn.LayerNorm(query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context is None:
            context = x          # self-attention fallback

        B, N, D = x.shape
        x_norm = self.norm_q(x)

        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head
        def split_heads(t):
            return rearrange(t, "b n (nh hd) -> b nh n hd",
                             nh=self.num_heads, hd=self.head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Flash attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0 if not self.training else 0.1,
        )
        attn_out = rearrange(attn_out, "b nh n hd -> b n (nh hd)")
        return x + self.to_out(attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner * 2),
            nn.GLU(),                    # gated linear unit
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block (spatial)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTransformerBlock(nn.Module):
    """
    Spatial transformer: flatten spatial → transformer → unflatten.
    Used inside U-Net at lower resolutions for cross-attention with text.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttention(channels, channels, num_heads, head_dim, dropout),  # self
                CrossAttention(channels, context_dim, num_heads, head_dim, dropout),  # cross
                FeedForward(channels, dropout=dropout),
            ])
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = self.proj_in(h)
        h = rearrange(h, "b c h w -> b (h w) c")

        for self_attn, cross_attn, ff in self.layers:
            h = self_attn(h)
            h = cross_attn(h, context=context)
            h = ff(h)

        h = rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return residual + self.proj_out(h)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with timestep embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        # Time projection — outputs 2*out_ch for scale-shift norm
        time_out = 2 * out_channels if use_scale_shift_norm else out_channels
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_out),
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Initialize last conv to zero (better initialization)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        t = self.time_proj(t_emb)[:, :, None, None]  # (B, C, 1, 1)

        if self.use_scale_shift_norm:
            scale, shift = t.chunk(2, dim=1)
            h = self.norm2(h) * (1.0 + scale) + shift
            h = self.act(h)
        else:
            h = self.act(self.norm2(h + t))

        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.op = (
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            if use_conv
            else nn.AvgPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet2DConditionModel(nn.Module):
    """
    Latent diffusion U-Net with:
    - Timestep embedding (sinusoidal + MLP)
    - Cross-attention text conditioning
    - Skip connections
    - Scale-shift norm
    - Flash attention
    Compatible with Stable Diffusion latent space (4ch, 64x64).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 320,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2, 1),
        num_heads: int = 8,
        head_dim: int = 64,
        context_dim: int = 768,
        dropout: float = 0.1,
        use_linear_proj: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        channels = [base_channels * m for m in channel_multipliers]
        time_emb_dim = base_channels * 4

        # ── Time embedding ─────────────────────────────────────────────────
        self.time_embedding = TimestepEmbedding(base_channels)
        # Output dim = base_channels*4
        self.time_proj = nn.Sequential(
            nn.Linear(base_channels * 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Input projection ───────────────────────────────────────────────
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ── Encoder (down) path ────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_channels = [channels[0]]
        in_ch = channels[0]

        for i, out_ch in enumerate(channels):
            res = 2 ** (len(channels) - 1 - i)          # relative resolution
            use_attn = res in attention_resolutions

            block = nn.ModuleList()
            for j in range(num_res_blocks):
                block.append(ResBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                if use_attn:
                    block.append(SpatialTransformerBlock(
                        in_ch, num_heads, head_dim, context_dim, dropout=dropout
                    ))
                skip_channels.append(in_ch)

            self.down_blocks.append(block)
            if i < len(channels) - 1:
                self.down_samples.append(Downsample(in_ch))
            else:
                self.down_samples.append(None)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attn = SpatialTransformerBlock(
            in_ch, num_heads, head_dim, context_dim, dropout=dropout
        )
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim, dropout)

        # ── Decoder (up) path ──────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, out_ch in enumerate(reversed(channels)):
            res = 2 ** i
            use_attn = res in attention_resolutions

            block = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                if use_attn:
                    block.append(SpatialTransformerBlock(
                        in_ch, num_heads, head_dim, context_dim, dropout=dropout
                    ))

            self.up_blocks.append(block)
            if i < len(channels) - 1:
                self.up_samples.append(Upsample(in_ch))
            else:
                self.up_samples.append(None)

        # ── Output ─────────────────────────────────────────────────────────
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                     (B, C, H, W) noisy latent
            timesteps:             (B,) integer timesteps
            encoder_hidden_states: (B, seq_len, context_dim) text embeddings
            attention_mask:        optional text attention mask
        Returns:
            (B, C, H, W) noise / v prediction
        """
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_proj(t_emb)          # (B, time_emb_dim)

        # Input
        h = self.conv_in(x)
        skips = [h]

        # Down path
        for block, downsample in zip(self.down_blocks, self.down_samples):
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, SpatialTransformerBlock):
                    h = layer(h, encoder_hidden_states)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, encoder_hidden_states)
        h = self.mid_block2(h, t_emb)

        # Up path
        for block, upsample in zip(self.up_blocks, self.up_samples):
            for layer in block:
                if isinstance(layer, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, t_emb)
                elif isinstance(layer, SpatialTransformerBlock):
                    h = layer(h, encoder_hidden_states)
            if upsample is not None:
                h = upsample(h)

        # Output
        h = self.act(self.norm_out(h))
        return self.conv_out(h)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())