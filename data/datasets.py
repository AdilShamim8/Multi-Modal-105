import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import webdataset as wds
from pathlib import Path
from typing import Optional, Dict, List, Callable, Tuple, Any
import io
import logging

from .transforms import ImageTransform, ResolutionBucketTransform

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LAION-style local dataset
# ─────────────────────────────────────────────────────────────────────────────

class LAIONDataset(Dataset):
    """
    Local LAION-style image-text dataset.
    Expects a directory with:
        images/  → 000000.jpg, 000001.jpg, ...
        metadata.parquet → columns: [image_id, caption, width, height, ...]
    """

    def __init__(
        self,
        data_root: str,
        tokenizer,
        resolution: int = 512,
        max_token_length: int = 77,
        center_crop: bool = True,
        random_flip: bool = True,
        filter_min_size: int = 256,
        filter_aesthetic_score: Optional[float] = None,
        use_bucket_sampling: bool = False,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.filter_min_size = filter_min_size

        # Load metadata
        meta_path = self.data_root / "metadata.parquet"
        self.meta = pd.read_parquet(meta_path)

        # Apply filters
        self.meta = self._filter_metadata(filter_min_size, filter_aesthetic_score)
        self.meta = self.meta.reset_index(drop=True)
        logger.info(f"Dataset loaded: {len(self.meta)} samples after filtering")

        # Transforms
        if use_bucket_sampling:
            self.transform = ResolutionBucketTransform(random_flip=random_flip)
        else:
            self.transform = ImageTransform(
                resolution=resolution,
                center_crop=center_crop,
                random_flip=random_flip,
            )
        self.use_bucket_sampling = use_bucket_sampling

    def _filter_metadata(
        self,
        min_size: int,
        aesthetic_score: Optional[float],
    ) -> pd.DataFrame:
        df = self.meta.copy()

        # Size filter
        if "width" in df.columns and "height" in df.columns:
            df = df[
                (df["width"] >= min_size) & (df["height"] >= min_size)
            ]

        # Aesthetic score filter
        if aesthetic_score is not None and "aesthetic_score" in df.columns:
            df = df[df["aesthetic_score"] >= aesthetic_score]

        # Drop missing captions
        if "caption" in df.columns:
            df = df[df["caption"].notna()]
            df = df[df["caption"].str.strip() != ""]

        return df

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.meta.iloc[idx]

        # Load image
        image_id = str(row.get("image_id", idx)).zfill(8)
        image_path = self.data_root / "images" / f"{image_id}.jpg"

        try:
            image = Image.open(image_path)
            if self.use_bucket_sampling:
                pixel_values, bucket = self.transform(image)
            else:
                pixel_values = self.transform(image)
                bucket = None
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            pixel_values = torch.zeros(3, 512, 512)
            bucket = None

        # Tokenize caption
        caption = str(row.get("caption", ""))
        tokens = self.tokenizer(
            caption,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "image": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
        }
        if bucket is not None:
            result["bucket"] = bucket
        return result


# ─────────────────────────────────────────────────────────────────────────────
# WebDataset for streaming large-scale data (LAION-5B, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class WebDatasetT2I:
    """
    WebDataset-based streaming dataloader for internet-scale datasets.
    Supports local shards, S3, GCS, HTTP.
    Format: tar shards containing {jpg, txt, json} triplets.
    """

    def __init__(
        self,
        urls: List[str],
        tokenizer,
        resolution: int = 512,
        max_token_length: int = 77,
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle_buffer: int = 10000,
        center_crop: bool = True,
        random_flip: bool = True,
        filter_min_size: int = 256,
        resampled: bool = True,
    ):
        self.transform = ImageTransform(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
        )
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.batch_size = batch_size

        # Build WebDataset pipeline
        dataset = (
            wds.WebDataset(
                urls,
                resampled=resampled,
                shardshuffle=True,
                nodesplitter=wds.split_by_node,
            )
            .shuffle(shuffle_buffer)
            .decode("pil")
            .rename(image="jpg;png;jpeg", text="txt", meta="json")
            .map(self._preprocess)
            .select(lambda x: x is not None)
            .batched(batch_size, partial=False)
        )

        self.loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=True,
        ).unbatched().shuffle(1000).batched(batch_size)

    def _preprocess(self, sample: Dict) -> Optional[Dict]:
        try:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                return None
            w, h = image.size
            if w < 256 or h < 256:
                return None

            pixel_values = self.transform(image)
            caption = sample.get("text", "")
            if not caption or not isinstance(caption, str):
                caption = ""

            tokens = self.tokenizer(
                caption,
                max_length=self.max_token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "image": pixel_values,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "caption": caption,
            }
        except Exception:
            return None

    def __iter__(self):
        return iter(self.loader)


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-style class-conditional dataset (for DiT)
# ─────────────────────────────────────────────────────────────────────────────

class ImageNetDataset(Dataset):
    """
    Class-conditional image dataset (ImageNet format).
    Directory structure: root/{class_id}/{image}.JPEG
    """

    def __init__(
        self,
        root: str,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.root = Path(root)
        self.resolution = resolution
        self.transform = ImageTransform(resolution, center_crop, random_flip)

        # Build class mapping
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

        # Build sample list
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in classes:
            cls_dir = self.root / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for ext in ("*.JPEG", "*.jpg", "*.png"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, cls_idx))

        logger.info(
            f"ImageNetDataset: {len(self.samples)} images, "
            f"{self.num_classes} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, self.resolution, self.resolution)

        return {
            "image": pixel_values,
            "class_labels": torch.tensor(label, dtype=torch.long),
        }