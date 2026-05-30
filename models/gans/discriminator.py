import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class EqualConv2d(nn.Module):
    """Equalized learning rate conv2d."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.scale = 1 / np.sqrt(in_ch * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale,
                        self.bias, self.stride, self.padding)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation for improved diversity."""

    def __init__(self, group_size: int = 4, num_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        F_ = self.num_features
        c = C // F_

        # Compute std across groups
        y = x.view(G, -1, F_, c, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.conv1 = EqualConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualConv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.skip = EqualConv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.downsample(h)
        return h + self.downsample(self.skip(x))


class StyleGAN2Discriminator(nn.Module):
    """
    StyleGAN2 discriminator with R1 gradient penalty support.
    """

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_max: int = 512,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        log_res = int(np.log2(img_resolution))

        def nf(stage):
            return min(channel_base // (2 ** stage), channel_max)

        # From-RGB
        self.from_rgb = EqualConv2d(img_channels, nf(log_res), 1)
        self.act = nn.LeakyReLU(0.2)

        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(log_res, 2, -1):
            self.blocks.append(DiscriminatorBlock(nf(i), nf(i - 1)))

        # Final block
        self.mb_stddev = MinibatchStdDev()
        self.final_conv = EqualConv2d(nf(2) + 1, nf(2), 3, padding=1)
        self.final_linear = nn.Sequential(
            nn.Linear(nf(2) * 4 * 4, nf(2)),
            nn.LeakyReLU(0.2),
            nn.Linear(nf(2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.from_rgb(x))
        for block in self.blocks:
            h = block(h)
        h = self.mb_stddev(h)
        h = self.act(self.final_conv(h))
        h = h.view(h.shape[0], -1)
        return self.final_linear(h)

    def r1_penalty(self, real_images: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty for training stability."""
        real_images = real_images.requires_grad_(True)
        real_pred = self(real_images)
        grad = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            create_graph=True,
        )[0]
        r1 = grad.pow(2).view(grad.shape[0], -1).sum(1).mean()
        return r1