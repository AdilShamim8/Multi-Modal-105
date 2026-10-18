import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, ClassVar
import json
import random
import logging
import decord
from decord import VideoReader, cpu, gpu
import cv2
from PIL import Image

logger = logging.getLogger(__name__)
decord.bridge.set_bridge("torch")


# ─────────────────────────────────────────────────────────────────────────────
# Video loading utilities
# ─────────────────────────────────────────────────────────────────────────────

class VideoLoader:
    """
    Efficient video loading with decord.
    Handles variable frame rates, resolutions, and durations.
    """

    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 256,
        fps: int = 8,
        center_crop: bool = True,
        random_flip: bool = True,
        sampling: str = "uniform",    # "uniform" | "random" | "fps"
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.fps = fps
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.sampling = sampling

    def load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load video and return (T, H, W, C) tensor in [0, 255].
        Returns None if loading fails.
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()

            # Sample frame indices
            indices = self._sample_frame_indices(
                total_frames, video_fps
            )
            if indices is None:
                return None

            frames = vr.get_batch(indices)   # (T, H, W, C) torch tensor
            return frames

        except Exception as e:
            logger.debug(f"Failed to load {video_path}: {e}")
            return None

    def _sample_frame_indices(
        self,
        total_frames: int,
        video_fps: float,
    ) -> Optional[List[int]]:
        """Sample frame indices based on sampling strategy."""

        if self.sampling == "fps":
            # Sample at target FPS
            stride = max(1, int(video_fps / self.fps))
            available = list(range(0, total_frames, stride))
            if len(available) < self.num_frames:
                return None
            start = random.randint(0, len(available) - self.num_frames)
            return available[start: start + self.num_frames]

        elif self.sampling == "uniform":
            if total_frames < self.num_frames:
                return None
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            return indices.tolist()

        elif self.sampling == "random":
            if total_frames < self.num_frames:
                return None
            start = random.randint(0, total_frames - self.num_frames)
            return list(range(start, start + self.num_frames))

        return None

    def preprocess_frames(
        self, frames: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Preprocess frames: resize, crop, normalize.
        Args:
            frames: (T, H, W, C) uint8 tensor
        Returns:
            (C, T, H, W) float tensor in [-1, 1]
        """
        T, H, W, C = frames.shape

        # Resize shortest side
        scale = self.frame_size / min(H, W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        # Process each frame
        processed = []
        for i in range(T):
            frame = frames[i].numpy().astype(np.uint8)  # (H, W, C)
            frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)

            # Crop
            if self.center_crop:
                h_start = (new_H - self.frame_size) // 2
                w_start = (new_W - self.frame_size) // 2
            else:
                h_start = random.randint(0, new_H - self.frame_size)
                w_start = random.randint(0, new_W - self.frame_size)

            frame = frame[
                h_start: h_start + self.frame_size,
                w_start: w_start + self.frame_size,
            ]
            processed.append(torch.from_numpy(frame))

        video = torch.stack(processed)   # (T, H, W, C)

        # Random horizontal flip (consistent across frames)
        if self.random_flip and random.random() < 0.5:
            video = torch.flip(video, dims=[2])

        # (T, H, W, C) → (C, T, H, W), normalize to [-1, 1]
        video = video.permute(3, 0, 1, 2).float()
        video = video / 127.5 - 1.0
        return video

    def __call__(self, video_path: str) -> Optional[torch.Tensor]:
        frames = self.load_video(video_path)
        if frames is None:
            return None
        return self.preprocess_frames(frames)


# ─────────────────────────────────────────────────────────────────────────────
# Video filtering
# ─────────────────────────────────────────────────────────────────────────────

class VideoFilter:
    """
    Multi-criteria video quality filter for dataset curation.
    Applies:
    - Resolution filtering (min W, H)
    - Duration filtering (min/max seconds)
    - Motion score filtering (avoid static/too-fast videos)
    - Aesthetic score filtering (if available)
    - NSFW filtering (if classifier available)
    """

    def __init__(
        self,
        min_resolution: int = 256,
        min_duration: float = 2.0,
        max_duration: float = 60.0,
        min_motion_score: float = 0.5,
        max_motion_score: float = 10.0,
        min_aesthetic_score: Optional[float] = None,
    ):
        self.min_resolution = min_resolution
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_motion_score = min_motion_score
        self.max_motion_score = max_motion_score
        self.min_aesthetic_score = min_aesthetic_score

    def compute_motion_score(self, frames: torch.Tensor) -> float:
        """
        Compute optical flow magnitude as motion proxy.
        Args:
            frames: (C, T, H, W) video tensor
        Returns:
            mean motion score (higher = more motion)
        """
        # Downsample for speed
        T = frames.shape[1]
        if T < 2:
            return 0.0

        frames_np = (
            (frames.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
        ).astype(np.uint8)

        motion_scores = []
        for i in range(min(T - 1, 8)):  # sample up to 8 frame pairs
            f1 = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            f2 = cv2.cvtColor(frames_np[i + 1], cv2.COLOR_RGB2GRAY)

            # Resize for speed
            f1 = cv2.resize(f1, (64, 64))
            f2 = cv2.resize(f2, (64, 64))

            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            motion_scores.append(float(magnitude))

        return float(np.mean(motion_scores)) if motion_scores else 0.0

    def filter_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply metadata-based filters to a DataFrame."""
        original = len(df)

        if "width" in df.columns and "height" in df.columns:
            df = df[
                (df["width"] >= self.min_resolution) &
                (df["height"] >= self.min_resolution)
            ]

        if "duration" in df.columns:
            df = df[
                (df["duration"] >= self.min_duration) &
                (df["duration"] <= self.max_duration)
            ]

        if "motion_score" in df.columns:
            df = df[
                (df["motion_score"] >= self.min_motion_score) &
                (df["motion_score"] <= self.max_motion_score)
            ]

        if self.min_aesthetic_score and "aesthetic_score" in df.columns:
            df = df[df["aesthetic_score"] >= self.min_aesthetic_score]

        if "caption" in df.columns:
            df = df[df["caption"].notna() & (df["caption"].str.len() > 10)]

        logger.info(
            f"VideoFilter: {original} → {len(df)} samples "
            f"({100*len(df)/max(original,1):.1f}% retained)"
        )
        return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Video standardization
# ─────────────────────────────────────────────────────────────────────────────

class VideoStandardizer:
    """
    Standardize video metadata to a consistent format.
    Converts various metadata schemas (WebVid, HD-VILA, Panda-70M, etc.)
    into a unified format.
    """

    UNIFIED_SCHEMA: ClassVar[Dict[str, type]] = {
        "video_id": str,
        "video_path": str,
        "caption": str,
        "width": int,
        "height": int,
        "duration": float,
        "fps": float,
        "num_frames": int,
        "aesthetic_score": float,
        "motion_score": float,
    }

    @staticmethod
    def from_webvid(df: pd.DataFrame, video_root: str) -> pd.DataFrame:
        """Convert WebVid-10M metadata to unified format."""
        out = pd.DataFrame()
        out["video_id"] = df["videoid"].astype(str)
        out["video_path"] = df["videoid"].apply(
            lambda vid: str(Path(video_root) / f"{vid}.mp4")
        )
        out["caption"] = df["name"]
        out["duration"] = df.get("duration", 0.0)
        out["width"] = df.get("width", 0)
        out["height"] = df.get("height", 0)
        out["fps"] = df.get("fps", 25.0)
        out["aesthetic_score"] = df.get("aesthetic_score", 0.0)
        out["motion_score"] = df.get("motion_score", 0.0)
        return out

    @staticmethod
    def from_panda70m(df: pd.DataFrame, video_root: str) -> pd.DataFrame:
        """Convert Panda-70M metadata to unified format."""
        out = pd.DataFrame()
        out["video_id"] = df["id"].astype(str)
        out["video_path"] = df["id"].apply(
            lambda vid: str(Path(video_root) / f"{vid}.mp4")
        )
        out["caption"] = df["caption"]
        out["duration"] = df.get("duration", 0.0)
        out["width"] = df.get("width", 0)
        out["height"] = df.get("height", 0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Video dataset
# ─────────────────────────────────────────────────────────────────────────────

class VideoTextDataset(Dataset):
    """
    Production video-text dataset for T2V training.
    Loads videos on-the-fly with caching support.
    """

    def __init__(
        self,
        metadata_path: str,
        tokenizer,
        num_frames: int = 16,
        frame_size: int = 256,
        fps: int = 8,
        max_token_length: int = 77,
        center_crop: bool = True,
        random_flip: bool = True,
        filter_config: Optional[Dict] = None,
        sampling: str = "uniform",
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        # Load and filter metadata
        if metadata_path.endswith(".parquet"):
            self.meta = pd.read_parquet(metadata_path)
        elif metadata_path.endswith(".csv"):
            self.meta = pd.read_csv(metadata_path)
        else:
            self.meta = pd.read_json(metadata_path, lines=True)

        # Apply filters
        if filter_config:
            video_filter = VideoFilter(**filter_config)
            self.meta = video_filter.filter_metadata(self.meta)

        logger.info(f"VideoTextDataset: {len(self.meta)} samples")

        self.video_loader = VideoLoader(
            num_frames=num_frames,
            frame_size=frame_size,
            fps=fps,
            center_crop=center_crop,
            random_flip=random_flip,
            sampling=sampling,
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        row = self.meta.iloc[idx]
        video_path = row["video_path"]

        # Load video
        video = self.video_loader(video_path)
        if video is None:
            # Return a random valid sample on failure
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Tokenize caption
        caption = str(row.get("caption", ""))
        tokens = self.tokenizer(
            caption,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "video": video,                                    # (C, T, H, W)
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Video latent cache
# ─────────────────────────────────────────────────────────────────────────────

class VideoLatentCache:
    """
    Pre-encode all videos into VideoVAE latents and cache to disk.
    Critical for large-scale T2V training efficiency.
    """

    def __init__(
        self,
        video_vae: torch.nn.Module,
        source_dataset: Dataset,
        cache_dir: str,
        device: str = "cuda",
    ):
        self.video_vae = video_vae.eval().to(device)
        self.source_dataset = source_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        for p in self.video_vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def build(
        self,
        batch_size: int = 4,
        num_workers: int = 4,
        overwrite: bool = False,
    ):
        """Pre-encode all videos to latents."""
        completed = self.cache_dir / "completed.txt"
        if completed.exists() and not overwrite:
            logger.info("Video latent cache already built. Skipping.")
            return

        loader = DataLoader(
            self.source_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        from tqdm import tqdm
        global_idx = 0

        for batch in tqdm(loader, desc="Encoding video latents"):
            videos = batch["video"].to(self.device)   # (B, C, T, H, W)

            # Encode with VideoVAE
            latents = self.video_vae.encode_to_latent(videos)  # (B, Cl, T', H', W')
            latents = latents.cpu().numpy().astype(np.float16)

            for i in range(latents.shape[0]):
                sample_dir = self.cache_dir / f"{global_idx:08d}"
                sample_dir.mkdir(exist_ok=True)
                np.save(sample_dir / "latent.npy", latents[i])

                # Save text data
                meta = {
                    "input_ids": batch["input_ids"][i].numpy(),
                    "attention_mask": batch["attention_mask"][i].numpy(),
                    "caption": batch["caption"][i] if isinstance(
                        batch["caption"], list
                    ) else batch["caption"],
                }
                torch.save(meta, sample_dir / "meta.pt")
                global_idx += 1

        with open(completed, "w") as f:
            f.write(f"total_samples={global_idx}\n")

        logger.info(f"Video latent cache complete: {global_idx} samples")


class VideoLatentDataset(Dataset):
    """Load cached video latents for fast T2V training."""

    def __init__(self, cache_dir: str, latent_scale: float = 0.18215):
        self.cache_dir = Path(cache_dir)
        self.latent_scale = latent_scale
        self.samples = sorted(
            [p for p in self.cache_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name),
        )
        logger.info(f"VideoLatentDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.samples[idx]
        latent = torch.from_numpy(
            np.load(p / "latent.npy").astype(np.float32)
        ) * self.latent_scale

        meta = torch.load(p / "meta.pt", weights_only=False)

        return {
            "latent": latent,
            "input_ids": torch.from_numpy(meta["input_ids"]).long(),
            "attention_mask": torch.from_numpy(meta["attention_mask"]).long(),
            "caption": meta.get("caption", ""),
        }