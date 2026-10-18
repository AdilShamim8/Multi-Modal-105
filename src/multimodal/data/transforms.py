import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import random


class ImageTransform:
    """
    Production image transform pipeline for T2I training.
    Supports:
    - Center crop / random crop
    - Random horizontal flip
    - Resolution bucketing (variable aspect ratios)
    - Normalization to [-1, 1]
    """

    def __init__(
        self,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        normalize: bool = True,
    ):
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.normalize = normalize

        # Core transforms
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize smallest side to resolution
        w, h = image.size
        scale = self.resolution / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Crop
        if self.center_crop:
            image = TF.center_crop(image, self.resolution)
        else:
            i = random.randint(0, new_h - self.resolution)
            j = random.randint(0, new_w - self.resolution)
            image = TF.crop(image, i, j, self.resolution, self.resolution)

        # Flip
        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)

        # To tensor [0, 1] then normalize to [-1, 1]
        t = self.to_tensor(image)
        if self.normalize:
            t = self.norm(t)
        return t


class ResolutionBucketTransform:
    """
    Multi-resolution bucketing for efficient batch training.
    Groups images into fixed aspect-ratio buckets to minimize padding waste.
    Used in SDXL and other production T2I systems.
    """

    BUCKETS = [
        (512, 512),
        (512, 768),
        (768, 512),
        (512, 1024),
        (1024, 512),
        (768, 768),
        (768, 1024),
        (1024, 768),
        (1024, 1024),
    ]

    def __init__(
        self,
        target_buckets: Optional[List[Tuple[int, int]]] = None,
        random_flip: bool = True,
    ):
        self.buckets = target_buckets or self.BUCKETS
        self.random_flip = random_flip
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def find_bucket(self, w: int, h: int) -> Tuple[int, int]:
        """Find best matching bucket for given image dimensions."""
        aspect = w / h
        best_bucket = min(
            self.buckets,
            key=lambda b: abs(b[0] / b[1] - aspect)
        )
        return best_bucket

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        bw, bh = self.find_bucket(w, h)

        # Resize to fill bucket
        scale = max(bw / w, bh / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to bucket
        image = TF.center_crop(image, (bh, bw))

        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)

        t = self.norm(self.to_tensor(image))
        return t, (bw, bh)


class AugmentationPipeline:
    """
    Advanced augmentation for data diversity:
    - Color jitter
    - Gaussian blur
    - Random grayscale
    Applied with configurable probability.
    """

    def __init__(
        self,
        color_jitter_prob: float = 0.1,
        blur_prob: float = 0.05,
        grayscale_prob: float = 0.02,
    ):
        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        )
        self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.color_jitter_prob = color_jitter_prob
        self.blur_prob = blur_prob
        self.grayscale_prob = grayscale_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        if random.random() < self.blur_prob:
            image = self.blur(image)
        if random.random() < self.grayscale_prob:
            image = TF.to_grayscale(image, num_output_channels=3)
        return image