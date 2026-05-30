import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class InceptionScoreEvaluator:
    """
    Inception Score (IS) evaluator.
    IS = exp(E_x[KL(p(y|x) || p(y))])
    Higher IS = better quality + diversity.
    Reference: Salimans et al. 2016.
    """

    def __init__(
        self,
        device: str = "cuda",
        num_splits: int = 10,
    ):
        self.device = device
        self.num_splits = num_splits

        # Load InceptionV3 classifier (with logits)
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception = self.inception.to(device)
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_predictions(self, images: torch.Tensor) -> np.ndarray:
        """
        Get softmax class probabilities from InceptionV3.
        Args:
            images: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 1000) softmax probabilities
        """
        # Normalize to InceptionV3 range
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        logits = self.inception(images)
        return F.softmax(logits, dim=1).cpu().numpy()

    def compute_inception_score(
        self,
        preds: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute IS from precomputed predictions using split method.
        Returns:
            (is_mean, is_std) across splits
        """
        N = preds.shape[0]
        split_size = N // self.num_splits
        scores = []

        for i in range(self.num_splits):
            part = preds[i * split_size: (i + 1) * split_size]

            # p(y|x) for this split
            # p(y) = marginal distribution
            p_y = np.mean(part, axis=0, keepdims=True)     # (1, 1000)
            p_y_x = part                                    # (N/splits, 1000)

            # KL(p(y|x) || p(y)) for each x
            kl = p_y_x * (np.log(p_y_x + 1e-10) - np.log(p_y + 1e-10))
            kl_sum = kl.sum(axis=1)                         # (N/splits,)

            # IS for this split
            scores.append(np.exp(kl_sum.mean()))

        return float(np.mean(scores)), float(np.std(scores))

    def evaluate(
        self,
        generator_fn: Callable,
        num_samples: int = 50000,
        batch_size: int = 32,
    ) -> Tuple[float, float]:
        """
        Full IS evaluation pipeline.
        Args:
            generator_fn: callable(batch_size) → (B, 3, H, W) images
        Returns:
            (is_mean, is_std)
        """
        all_preds = []
        total = 0

        with tqdm(total=num_samples, desc="Computing IS") as pbar:
            while total < num_samples:
                cur_batch = min(batch_size, num_samples - total)
                images = generator_fn(cur_batch)
                if isinstance(images, tuple):
                    images = images[0]
                preds = self.get_predictions(images.to(self.device))
                all_preds.append(preds)
                total += cur_batch
                pbar.update(cur_batch)

        all_preds = np.concatenate(all_preds, axis=0)[:num_samples]
        is_mean, is_std = self.compute_inception_score(all_preds)

        logger.info(f"Inception Score: {is_mean:.3f} ± {is_std:.3f}")
        return is_mean, is_std