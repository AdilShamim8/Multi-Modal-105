
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit import (
    AdaLNZero,
    FinalLayer,
    TimestepEmbedding,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3D RoPE (Rotary Position Embedding for video)
# ─────────────────────────────────────────────────────────────────────────────

def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    3D sinusoidal position embedding for (T, H, W) video tokens.
    Allocates embed_dim/3 to each axis.
    """
    assert embed_dim % 3 == 0, "embed_dim must be divisible by 3 for 3D pos embed"
    d = embed_dim // 3

    def _sincos_1d(length, dim):
        pos = np.arange(length, dtype=np.float32)
        omega = np.arange(dim // 2, dtype=np.float64)
        omega /= dim / 2.0
        omega = 1.0 / (10000 ** omega)
        out = np.outer(pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (L, dim)

    t_emb = _sincos_1d(grid_t, d)     # (T, d)
    h_emb = _sincos_1d(grid_h, d)     # (H, d)
    w_emb = _sincos_1d(grid_w, d)     # (W, d)

    # Combine: for each (t, h, w), concat all three
    T, H, W = grid_t, grid_h, grid_w
    pos = np.zeros((T * H * W, embed_dim), dtype=np.float32)

    for ti in range(T):
        for hi in range(H):
            for wi in range(W):
                idx = ti * H * W + hi * W + wi
                pos[idx, :d] = t_emb[ti]
                pos[idx, d:2*d] = h_emb[hi]
                pos[idx, 2*d:] = w_emb[wi]

    return torch.from_numpy(pos).float()   # (T*H*W, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionBlock(nn.Module):
    """
    Attention across the time dimension only.
    Applied after spatial attention to capture temporal dependencies.
    Tokens are grouped by spatial position: each (h,w) position
    attends to all T time steps independently.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Zero-init for residual learning
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        S: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T*S, D) flattened video tokens
            T: number of temporal frames
            S: number of spatial tokens per frame (H*W / patch_size^2)
        """
        B, N, D = x.shape
        assert N == T * S

        residual = x
        x_norm = self.norm(x)

        # Reshape: (B, T*S, D) → (B*S, T, D) to attend over time
        x_norm = rearrange(x_norm, "b (t s) d -> (b s) t d", t=T, s=S)

        q = rearrange(self.q(x_norm), "bs t (nh hd) -> bs nh t hd",
                      nh=self.num_heads, hd=self.head_dim)
        k = rearrange(self.k(x_norm), "bs t (nh hd) -> bs nh t hd",
                      nh=self.num_heads, hd=self.head_dim)
        v = rearrange(self.v(x_norm), "bs t (nh hd) -> bs nh t hd",
                      nh=self.num_heads, hd=self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "bs nh t hd -> bs t (nh hd)")
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Reshape back: (B*S, T, D) → (B, T*S, D)
        attn_out = rearrange(attn_out, "(b s) t d -> b (t s) d", b=B, s=S)
        return residual + attn_out


# ─────────────────────────────────────────────────────────────────────────────
# Video DiT Block
# ─────────────────────────────────────────────────────────────────────────────

class VideoDiTBlock(nn.Module):
    """
    Video Diffusion Transformer block.
    Architecture:
        1. adaLN-Zero spatial self-attention
        2. Temporal attention (across T dim)
        3. Cross-attention with text
        4. adaLN-Zero MLP (FFN)
    
    Supports full spatio-temporal modeling with factorized attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1152,
        context_dim: int | None = 768,
        temporal_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # ── Spatial self-attention with adaLN ─────────────────────────────
        self.adaLN = AdaLNZero(hidden_size, cond_dim)
        self.spatial_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spatial_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spatial_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spatial_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Temporal attention ────────────────────────────────────────────
        self.temporal_attn = TemporalAttentionBlock(
            hidden_size, temporal_heads, dropout
        )

        # ── Cross-attention (text conditioning) ───────────────────────────
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.cross_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_q = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cross_k = nn.Linear(context_dim, hidden_size, bias=False)
            self.cross_v = nn.Linear(context_dim, hidden_size, bias=False)
            self.cross_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── MLP with adaLN ────────────────────────────────────────────────
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def _spatial_attn(self, x_norm: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.spatial_q(x_norm), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        k = rearrange(self.spatial_k(x_norm), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        v = rearrange(self.spatial_v(x_norm), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh n hd -> b n (nh hd)")
        return self.spatial_out(out)

    def _cross_attn(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        x_norm = self.cross_norm(x)
        q = rearrange(self.cross_q(x_norm), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        k = rearrange(self.cross_k(context), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        v = rearrange(self.cross_v(context), "b n (nh hd) -> b nh n hd",
                      nh=self.num_heads, hd=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh n hd -> b n (nh hd)")
        return self.cross_out(out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        T: int,
        S: int,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T*S, D) video tokens
            c:       (B, D) conditioning vector (t_emb + text_pool)
            T:       number of temporal frames (latent)
            S:       spatial tokens per frame
            context: (B, seq_len, context_dim) text embeddings
        """
        # ── Spatial self-attention ─────────────────────────────────────────
        x_norm, gate1, shift2, scale2, gate2 = self.adaLN(x, c)
        x = x + gate1 * self._spatial_attn(x_norm)

        # ── Temporal attention ────────────────────────────────────────────
        x = self.temporal_attn(x, T, S)

        # ── Cross-attention ───────────────────────────────────────────────
        if self.has_cross_attn and context is not None:
            x = x + self._cross_attn(x, context)

        # ── MLP ───────────────────────────────────────────────────────────
        x_norm2 = self.norm_mlp(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(x_norm2)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Video Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class VideoPatchEmbed(nn.Module):
    """
    3D patch embedding for video latents.
    Embeds (B, C, T, H, W) → (B, T*H'*W', D) tokens.
    Temporal patch size = 1 (per-frame patches) or > 1 for tube embedding.
    """

    def __init__(
        self,
        spatial_patch_size: int = 2,
        temporal_patch_size: int = 1,
        in_channels: int = 16,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            tokens: (B, T'*H'*W', D)
            T_tokens: number of temporal tokens
            S_tokens: number of spatial tokens per frame
        """
        B, C, T, H, W = x.shape
        x = self.proj(x)          # (B, D, T', H', W')
        B, D, T_, H_, W_ = x.shape
        x = rearrange(x, "b d t h w -> b (t h w) d")
        T_tokens = T_
        S_tokens = H_ * W_
        return x, T_tokens, S_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Full Video DiT
# ─────────────────────────────────────────────────────────────────────────────

class VideoDiT(nn.Module):
    """
    Video Diffusion Transformer (Video DiT).
    
    Architecture based on:
    - Open-Sora (PKU-Yuan Lab)
    - CogVideo (Zhipu AI)
    - Sora (OpenAI) — as described in technical report
    
    Key design choices:
    1. Factorized spatial + temporal attention
    2. 3D sinusoidal positional embeddings
    3. adaLN-Zero conditioning from timestep + text pool
    4. Cross-attention with T5/CLIP text encoder
    5. Classifier-free guidance with null text
    """

    def __init__(
        self,
        # Input shape
        num_frames: int = 16,
        frame_size: int = 32,              # latent spatial size (after VAE)
        in_channels: int = 16,             # video VAE latent channels
        # Architecture
        spatial_patch_size: int = 2,
        temporal_patch_size: int = 1,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        temporal_heads: int = 8,
        mlp_ratio: float = 4.0,
        # Conditioning
        context_dim: int = 768,            # text encoder dim
        pooled_text_dim: int = 1024,       # pooled text for adaLN
        # Output
        learn_sigma: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.depth = depth

        # ── Patch embedding ──────────────────────────────────────────────
        self.patch_embed = VideoPatchEmbed(
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        T_tokens = num_frames // temporal_patch_size
        S_tokens = (frame_size // spatial_patch_size) ** 2
        self.num_tokens = T_tokens * S_tokens

        # ── Positional embedding (3D sinusoidal, fixed) ──────────────────
        pos_embed = get_3d_sincos_pos_embed(
            hidden_size,
            grid_t=T_tokens,
            grid_h=frame_size // spatial_patch_size,
            grid_w=frame_size // spatial_patch_size,
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        # ── Timestep embedding ───────────────────────────────────────────
        self.t_embedder = nn.Sequential(
            TimestepEmbedding(hidden_size),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ── Text conditioning ─────────────────────────────────────────────
        # Project text sequence embeddings for cross-attention
        self.text_proj = nn.Linear(context_dim, hidden_size)
        # Project pooled text for adaLN conditioning
        self.text_pool_proj = nn.Linear(pooled_text_dim, hidden_size)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            VideoDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=hidden_size,
                context_dim=hidden_size,   # projected text
                temporal_heads=temporal_heads,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ── Final layer ───────────────────────────────────────────────────
        self.final_layer = FinalLayer(
            hidden_size,
            spatial_patch_size,
            self.out_channels,
            hidden_size,
        )

        self._init_weights()
        self._T_tokens = T_tokens
        self._S_tokens = S_tokens

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)

    def unpatchify(
        self,
        x: torch.Tensor,
        T: int,
        H: int,
        W: int,
        p: int,
    ) -> torch.Tensor:
        """
        (B, T*H'*W', p*p*C) → (B, C, T, H, W)
        where H' = H//p, W' = W//p
        """
        Hp = H // p
        Wp = W // p
        C = self.out_channels

        x = rearrange(
            x, "b (t hp wp) (p1 p2 c) -> b c t (hp p1) (wp p2)",
            t=T, hp=Hp, wp=Wp, p1=p, p2=p, c=C
        )
        return x

    def forward(
        self,
        x: torch.Tensor,                              # (B, C, T, H, W)
        t: torch.Tensor,                              # (B,) timesteps
        text_embeds: torch.Tensor | None = None,   # (B, seq, context_dim)
        text_pool: torch.Tensor | None = None,     # (B, pooled_text_dim)
        force_drop_text: bool = False,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Patch embed
        tokens, T_tok, S_tok = self.patch_embed(x)   # (B, N, D)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]

        # Conditioning
        t_emb = self.t_embedder(t)                    # (B, D)
        if text_pool is not None:
            pool_emb = self.text_pool_proj(text_pool) # (B, D)
            c = t_emb + pool_emb
        else:
            c = t_emb

        # Project text sequence for cross-attention
        ctx = None
        if text_embeds is not None:
            ctx = self.text_proj(text_embeds)         # (B, seq, D)

        # Transformer
        for block in self.blocks:
            tokens = block(tokens, c, T_tok, S_tok, context=ctx)

        # Final layer
        tokens = self.final_layer(tokens, c)          # (B, N, p²C)

        # Unpatchify
        p = self.patch_embed.spatial_patch_size
        out = self.unpatchify(tokens, T_tok, H // self.patch_embed.temporal_patch_size, W, p)
        # If temporal_patch_size > 1, out T dim = T / temporal_patch_size
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
        text_pool: torch.Tensor | None = None,
        null_text_embeds: torch.Tensor | None = None,
        null_text_pool: torch.Tensor | None = None,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        """Classifier-free guidance forward pass."""
        B = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)

        # Stack conditional + unconditional
        null_te = null_text_embeds if null_text_embeds is not None \
                  else torch.zeros_like(text_embeds)
        null_tp = null_text_pool if null_text_pool is not None \
                  else torch.zeros_like(text_pool)

        text_double = torch.cat([text_embeds, null_te], dim=0)
        pool_double = torch.cat([text_pool, null_tp], dim=0)

        out = self.forward(x_double, t_double, text_double, pool_double)
        cond, uncond = out[:B], out[B:]
        return uncond + cfg_scale * (cond - uncond)
