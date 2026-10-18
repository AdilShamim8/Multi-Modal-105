import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from typing import Optional, Tuple

# TimestepEmbedding is shared with the U-Net implementation (same package;
# no cycle: unet.py does not import dit.py).
from .unet import TimestepEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """2D image → sequence of patch tokens."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 1152,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)
        return rearrange(x, "b d h w -> b (h w) d")


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    2D sinusoidal positional embedding.
    Returns: (grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)   # [2, grid_size, grid_size]
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, N)

    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (N, D)


def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Layer Norm (adaLN-Zero) — DiT's conditioning mechanism
# ─────────────────────────────────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero modulation.
    Conditions on timestep + class embeddings via learned scale/shift.
    """

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 6 outputs: scale1, shift1, gate1, scale2, shift2, gate2
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        # Zero-init for stable training
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        mods = self.modulation(c)[:, None].chunk(6, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods
        x_norm = self.norm(x) * (1 + scale1) + shift1
        return x_norm, gate1, shift2, scale2, gate2


class AdaLNSingle(nn.Module):
    """Simplified adaLN for single modulation (used in final layer)."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.modulation(c)[:, None].chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# DiT Block
# ─────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    DiT Transformer block with adaLN-Zero conditioning.
    Supports both self-attention and cross-attention for text.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1152,
        context_dim: Optional[int] = None,     # if None, no cross-attn
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        # adaLN
        self.adaLN = AdaLNZero(hidden_size, cond_dim)

        # Self-attention
        self.self_attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Cross-attention (optional, for text conditioning)
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cross_attn_k = nn.Linear(context_dim, hidden_size, bias=False)
            self.cross_attn_v = nn.Linear(context_dim, hidden_size, bias=False)
            self.cross_attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.head_dim = head_dim

    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = rearrange(self.self_attn_q(x), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        k = rearrange(self.self_attn_k(x), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        v = rearrange(self.self_attn_v(x), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh n hd -> b n (nh hd)")
        return self.self_attn_out(out)

    def _cross_attn(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        q = rearrange(self.cross_attn_q(x), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        k = rearrange(self.cross_attn_k(context), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        v = rearrange(self.cross_attn_v(context), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh n hd -> b n (nh hd)")
        return self.cross_attn_out(out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # adaLN conditioning
        x_norm, gate1, shift2, scale2, gate2 = self.adaLN(x, c)

        # Self-attention
        x = x + gate1 * self._self_attn(x_norm)

        # Cross-attention
        if self.has_cross_attn and context is not None:
            x = x + self.norm_cross(x)
            x = x + self._cross_attn(self.norm_cross(x), context)

        # MLP with adaLN
        x_norm2 = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(x_norm2)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Final layer
# ─────────────────────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int,
                 out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c)[:, None].chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# DiT — full model
# ─────────────────────────────────────────────────────────────────────────────

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for image generation.
    - Class-conditional: uses class label embedding
    - Text-conditional: uses cross-attention with text encoder output
    Supports both modes; class_cond and text_cond can be combined.
    Reference: Peebles & Xie 2023 (https://arxiv.org/abs/2212.09748)
    """

    def __init__(
        self,
        input_size: int = 32,         # latent spatial size (e.g., 64 for 512px images)
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,          # for class-conditional
        context_dim: Optional[int] = 768, # for text-conditional; None = class-only
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2

        # ── Patch embedding ─────────────────────────────────────────────────
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)

        # ── Timestep embedding ──────────────────────────────────────────────
        self.t_embedder = nn.Sequential(
            TimestepEmbedding(hidden_size),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ── Class label embedding (classifier-free guidance) ─────────────────
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null token
        self.null_class_id = num_classes

        # ── Text conditioning projection ─────────────────────────────────────
        self.has_text_cond = context_dim is not None
        if self.has_text_cond:
            self.text_proj = nn.Linear(context_dim, hidden_size)

        # ── Positional embedding (fixed sinusoidal) ─────────────────────────
        pos_embed = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=hidden_size,
                context_dim=hidden_size if self.has_text_cond else None,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ── Final layer ──────────────────────────────────────────────────────
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, hidden_size
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Patch embedder
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(
            self.x_embedder.proj.weight.shape[0], -1
        ))

        # Timestep MLP
        nn.init.normal_(self.t_embedder[0].mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[0].mlp[2].weight, std=0.02)

        # Class embedding
        nn.init.normal_(self.y_embedder.weight, std=0.02)

    def _get_conditioning(
        self,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        force_drop_ids: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build conditioning vector c from timestep + class label."""
        t_emb = self.t_embedder(t)     # (B, D)

        if y is not None:
            # Classifier-free guidance dropout
            if self.training and not force_drop_ids:
                drop_mask = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
                y = y.clone()
                y[drop_mask] = self.null_class_id
            y_emb = self.y_embedder(y)  # (B, D)
            c = t_emb + y_emb
        else:
            c = t_emb

        # Text embeddings
        ctx = None
        if self.has_text_cond and text_embeds is not None:
            ctx = self.text_proj(text_embeds)  # (B, seq_len, D)

        return c, ctx

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_size^2*C) → (B, C, H, W)."""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.out_channels
        x = x.view(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(-1, c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        force_drop_ids: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:           (B, C, H, W) noisy latent
            t:           (B,) timesteps
            y:           (B,) class labels (optional)
            text_embeds: (B, seq_len, context_dim) text embeddings (optional)
            encoder_hidden_states: diffusers-style alias for text_embeds —
                keeps DiT and U-Net interchangeable behind the pipelines
        Returns:
            (B, C, H, W) or (B, 2C, H, W) if learn_sigma
        """
        # encoder_hidden_states is the canonical pipeline kwarg; accept both.
        if text_embeds is None and encoder_hidden_states is not None:
            text_embeds = encoder_hidden_states

        # Patch embed + positional encoding
        x = self.x_embedder(x) + self.pos_embed         # (B, N, D)

        # Conditioning
        c, ctx = self._get_conditioning(t, y, text_embeds, force_drop_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, context=ctx)

        # Final layer → unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Classifier-free guidance forward pass.
        Runs both conditional and unconditional in one batch.
        """
        B = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)

        if y is not None:
            y_null = torch.full_like(y, self.null_class_id)
            y_double = torch.cat([y, y_null], dim=0)
        else:
            y_double = None

        if text_embeds is not None:
            null_text = torch.zeros_like(text_embeds)
            text_double = torch.cat([text_embeds, null_text], dim=0)
        else:
            text_double = None

        out = self.forward(x_double, t_double, y_double, text_double)
        cond, uncond = out[:B], out[B:]

        return uncond + cfg_scale * (cond - uncond)


# ─────────────────────────────────────────────────────────────────────────────
# EMA model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy updated with EMA during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Copy EMA weights into model for inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.decay = state["decay"]