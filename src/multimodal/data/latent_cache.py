import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LatentCacheBuilder:
    """
    Pre-encodes all dataset images into VAE latents and caches them to disk.
    This massively speeds up diffusion training by avoiding repeated VAE encoding.

    Usage:
        builder = LatentCacheBuilder(vae, source_dataset, cache_dir)
        builder.build(batch_size=64, num_workers=8)
        cached_dataset = LatentCachedDataset(cache_dir, tokenizer)
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        source_dataset: Dataset,
        cache_dir: str,
        device: str = "cuda",
    ):
        self.vae = vae.eval().to(device)
        self.source_dataset = source_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def build(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        overwrite: bool = False,
    ):
        """Encode all images and cache latents + metadata."""
        loader = DataLoader(
            self.source_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        completed_file = self.cache_dir / "completed.txt"
        if completed_file.exists() and not overwrite:
            logger.info("Cache already built. Skipping.")
            return

        logger.info(f"Building latent cache → {self.cache_dir}")
        global_idx = 0

        for batch in tqdm(loader, desc="Encoding latents"):
            images = batch["image"].to(self.device)

            # VAE encode
            latents = self.vae.encode_to_latent(images)   # (B, 4, H/8, W/8)
            latents = latents.cpu().numpy().astype(np.float16)

            # Save per-sample
            for i in range(latents.shape[0]):
                sample_path = self.cache_dir / f"{global_idx:08d}"
                sample_path.mkdir(exist_ok=True)

                # Latent
                np.save(sample_path / "latent.npy", latents[i])

                # Metadata (caption, input_ids, etc.)
                meta = {}
                for k, v in batch.items():
                    if k == "image":
                        continue
                    if isinstance(v, torch.Tensor):
                        meta[k] = v[i].numpy()
                    elif isinstance(v, list | tuple):
                        meta[k] = v[i]
                    else:
                        meta[k] = v

                torch.save(meta, sample_path / "meta.pt")
                global_idx += 1

        # Write completion marker
        with open(completed_file, "w") as f:
            f.write(f"total_samples={global_idx}\n")

        logger.info(f"Latent cache complete: {global_idx} samples")


class LatentCachedDataset(Dataset):
    """
    Dataset that loads pre-encoded VAE latents from disk.
    Used during diffusion model training for maximum throughput.
    """

    def __init__(
        self,
        cache_dir: str,
        latent_scale: float = 0.18215,
        return_text: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.latent_scale = latent_scale
        self.return_text = return_text

        # Discover all samples
        self.samples: list[Path] = sorted(
            [p for p in self.cache_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name),
        )
        logger.info(f"LatentCachedDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_path = self.samples[idx]

        # Load latent
        latent = np.load(sample_path / "latent.npy").astype(np.float32)
        latent = torch.from_numpy(latent) * self.latent_scale

        # Load metadata
        meta = torch.load(sample_path / "meta.pt", weights_only=False)

        result = {"latent": latent}
        if self.return_text:
            if "input_ids" in meta:
                result["input_ids"] = torch.from_numpy(meta["input_ids"]).long()
            if "attention_mask" in meta:
                result["attention_mask"] = torch.from_numpy(meta["attention_mask"]).long()
            if "caption" in meta:
                result["caption"] = meta["caption"]

        return result


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """Build production dataloader with optional distributed sampler."""
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
