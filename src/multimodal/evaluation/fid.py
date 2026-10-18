import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InceptionV3Features(nn.Module):
    """
    InceptionV3 feature extractor for FID computation.
    Uses pool3 features (2048-d) as per Heusel et al. 2017.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device

        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # Remove final classification layers
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.features.to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Args:
            images: (B, 3, H, W) in [0, 1] or [-1, 1]
        Returns:
            (B, 2048) numpy features
        """
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Resize to InceptionV3 input size
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Normalize for InceptionV3
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return self.features(images).cpu().numpy()

    def compute_stats(
        self,
        dataloader: DataLoader,
        max_samples: int = 50000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of Inception features for a dataset.
        Returns:
            mu: (2048,) mean vector
            sigma: (2048, 2048) covariance matrix
        """
        all_features = []
        total = 0

        for batch in tqdm(dataloader, desc="Extracting Inception features"):
            if isinstance(batch, list | tuple):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("pixel_values"))
            else:
                images = batch

            images = images.to(self.device)
            features = self.extract_features(images)
            all_features.append(features)
            total += images.shape[0]

            if total >= max_samples:
                break

        all_features = np.concatenate(all_features, axis=0)[:max_samples]
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance between two Gaussian distributions.
    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2*(Σ1*Σ2)^(1/2))
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # Matrix square root via eigendecomposition
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        logger.warning("FID: non-finite covmean, adding epsilon to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical clean-up
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)


class FIDEvaluator:
    """
    Full FID evaluation pipeline:
    1. Compute real data statistics (once, cache to disk)
    2. Generate fake images
    3. Compute fake statistics
    4. Compute FID
    """

    def __init__(
        self,
        real_dataloader: DataLoader,
        cache_dir: str = "./fid_cache",
        device: str = "cuda",
        max_real_samples: int = 50000,
        max_fake_samples: int = 10000,
    ):
        self.real_dataloader = real_dataloader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_real_samples = max_real_samples
        self.max_fake_samples = max_fake_samples

        self.inception = InceptionV3Features(device)
        self._real_stats: tuple[np.ndarray, np.ndarray] | None = None

    def _load_or_compute_real_stats(self) -> tuple[np.ndarray, np.ndarray]:
        stats_file = self.cache_dir / "real_stats.npz"
        if stats_file.exists():
            data = np.load(stats_file)
            logger.info("Loaded real FID stats from cache")
            return data["mu"], data["sigma"]

        logger.info("Computing real data FID statistics...")
        mu, sigma = self.inception.compute_stats(self.real_dataloader, self.max_real_samples)
        np.savez(stats_file, mu=mu, sigma=sigma)
        logger.info("Real FID stats cached")
        return mu, sigma

    @torch.no_grad()
    def compute_fid_from_generator(
        self,
        generator_fn,  # callable: (batch_size) → torch.Tensor (B, 3, H, W)
        batch_size: int = 32,
        num_samples: int | None = None,
    ) -> float:
        """
        Args:
            generator_fn: function that generates a batch of images given batch_size
            batch_size:   generation batch size
            num_samples:  total fake samples to generate
        """
        num_samples = num_samples or self.max_fake_samples

        if self._real_stats is None:
            self._real_stats = self._load_or_compute_real_stats()
        mu_real, sigma_real = self._real_stats

        # Generate fake features
        fake_features = []
        total_generated = 0

        logger.info(f"Generating {num_samples} fake samples for FID...")
        with tqdm(total=num_samples, desc="FID: generating") as pbar:
            while total_generated < num_samples:
                cur_batch = min(batch_size, num_samples - total_generated)
                fake_images = generator_fn(cur_batch)
                if isinstance(fake_images, tuple):
                    fake_images = fake_images[0]

                feats = self.inception.extract_features(fake_images.to(self.device))
                fake_features.append(feats)
                total_generated += cur_batch
                pbar.update(cur_batch)

        fake_features = np.concatenate(fake_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        logger.info(f"FID Score: {fid:.3f}")
        return fid
