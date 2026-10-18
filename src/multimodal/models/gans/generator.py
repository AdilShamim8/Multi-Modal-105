import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    """Equalized learning rate linear layer (StyleGAN)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, lr_mul: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1 / np.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x, self.weight * self.scale, self.bias * self.lr_mul if self.bias is not None else None
        )


class MappingNetwork(nn.Module):
    """StyleGAN2 mapping network: z → w."""

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8,
        lr_mul: float = 0.01,
    ):
        super().__init__()
        layers: list[nn.Module] = [PixelNorm()]
        for i in range(num_layers):
            layers.append(EqualLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        self.w_dim = w_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        w_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(w_dim, in_channels, bias=True)
        self.scale = 1 / np.sqrt(in_channels * kernel_size**2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Modulation
        style = self.modulation(w).view(B, 1, C, 1, 1) + 1.0
        weight = self.scale * self.weight * style  # (B, out, in, k, k)

        # Demodulation
        if self.demodulate:
            sigma = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).sqrt() + 1e-8
            weight = weight / sigma

        # Group convolution trick (batch as groups).
        # reshape (not view): the broadcast modulation makes `weight`
        # non-contiguous, which view() rejects at runtime.
        weight = weight.reshape(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        x = x.reshape(1, B * C, H, W)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            _, _, H, W = x.shape

        x = F.conv2d(x, weight, padding=self.padding, groups=B)
        return x.reshape(B, self.out_channels, x.shape[-2], x.shape[-1])


class StyleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        upsample: bool = False,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, w_dim, upsample=upsample)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv(x, w)
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_strength * noise
        x = x + self.bias.view(1, -1, 1, 1)
        return self.act(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, w_dim: int):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.conv(x, w) + self.bias.view(1, -1, 1, 1)
        if skip is not None:
            out = out + F.interpolate(skip, scale_factor=2.0, mode="bilinear", align_corners=False)
        return out


class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2-style generator.
    Generates high-quality images from latent vectors via style modulation.
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_resolution: int = 256,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_mapping_layers: int = 8,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.num_ws = int(np.log2(img_resolution)) * 2 - 2

        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping_layers)

        # Synthesis network
        self.log_res = int(np.log2(img_resolution))
        self.resolutions = [2**i for i in range(2, self.log_res + 1)]

        def nf(stage):
            return min(channel_base // (2**stage), channel_max)

        # Learned constant input
        self.const = nn.Parameter(torch.randn(1, nf(1), 4, 4))

        self.style_blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_ch = nf(1)
        for i, _res in enumerate(self.resolutions):
            out_ch = nf(i + 1)
            upsample = i > 0
            self.style_blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=upsample))
            self.style_blocks.append(StyleBlock(out_ch, out_ch, w_dim))
            self.to_rgbs.append(ToRGB(out_ch, w_dim))
            in_ch = out_ch

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        noise_mode: str = "random",  # "random" | "const" | "none"
    ) -> torch.Tensor:
        B = z.shape[0]
        w = self.mapping(z)  # (B, w_dim)
        ws = w.unsqueeze(1).expand(-1, self.num_ws, -1)  # (B, num_ws, w_dim)

        # W+ space: different w per layer
        x = self.const.expand(B, -1, -1, -1)
        img = None

        block_idx = 0
        for _, to_rgb in enumerate(self.to_rgbs):
            x = self.style_blocks[block_idx](x, ws[:, block_idx])
            block_idx += 1
            x = self.style_blocks[block_idx](x, ws[:, block_idx])
            block_idx += 1
            img = to_rgb(x, ws[:, block_idx - 1], img)

        return img  # (B, 3, H, W) in [-1, 1] (after tanh in ToRGB optionally)
