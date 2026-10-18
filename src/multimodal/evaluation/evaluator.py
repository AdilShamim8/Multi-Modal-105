import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .clip_score import CLIPScoreEvaluator
from .fid import FIDEvaluator
from .inception_score import InceptionScoreEvaluator

logger = logging.getLogger(__name__)


class T2IEvaluator:
    """
    Unified Text-to-Image evaluation suite.
    Computes:
    1. FID    — image quality + distribution fidelity
    2. IS     — image quality + diversity
    3. CLIP Score — image-text alignment
    4. Diversity  — intra-batch visual diversity
    """

    def __init__(
        self,
        real_dataloader,
        eval_prompts: list[str],
        cache_dir: str = "./eval_cache",
        device: str = "cuda",
        fid_num_real: int = 50000,
        fid_num_fake: int = 10000,
        is_num_samples: int = 50000,
        clip_batch_size: int = 32,
    ):
        self.device = device
        self.eval_prompts = eval_prompts
        self.clip_batch_size = clip_batch_size

        # Initialize evaluators
        self.fid_evaluator = FIDEvaluator(
            real_dataloader=real_dataloader,
            cache_dir=str(Path(cache_dir) / "fid"),
            device=device,
            max_real_samples=fid_num_real,
            max_fake_samples=fid_num_fake,
        )

        self.is_evaluator = InceptionScoreEvaluator(
            device=device,
            num_splits=10,
        )

        self.clip_evaluator = CLIPScoreEvaluator(device=device)

        self.results_dir = Path(cache_dir) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_all(
        self,
        generator_fn: Callable,
        step: int,
        compute_fid: bool = True,
        compute_is: bool = True,
        compute_clip: bool = True,
        compute_diversity: bool = True,
    ) -> dict[str, Any]:
        """
        Run full evaluation suite.
        Args:
            generator_fn: callable — either
                (batch_size: int) → (B, 3, H, W) for FID/IS
                or (prompts: List[str]) → (B, 3, H, W) for CLIP
        Returns:
            dict of metric_name → value
        """
        results: dict[str, Any] = {"step": step}

        # ── FID ──────────────────────────────────────────────────────────────
        if compute_fid:
            logger.info("Running FID evaluation...")
            try:
                # generator_fn already has the (batch_size) -> images signature
                # that the FID path expects; no adapter needed.
                fid_score = self.fid_evaluator.compute_fid_from_generator(
                    generator_fn=generator_fn,
                    batch_size=self.clip_batch_size,
                )
                results["eval/fid"] = fid_score
            except Exception as e:
                logger.error(f"FID failed: {e}")
                results["eval/fid"] = float("nan")

        # ── IS ───────────────────────────────────────────────────────────────
        if compute_is:
            logger.info("Running Inception Score evaluation...")
            try:
                is_mean, is_std = self.is_evaluator.evaluate(
                    generator_fn=generator_fn,
                    num_samples=10000,
                    batch_size=self.clip_batch_size,
                )
                results["eval/is_mean"] = is_mean
                results["eval/is_std"] = is_std
            except Exception as e:
                logger.error(f"IS failed: {e}")
                results["eval/is_mean"] = float("nan")

        # ── CLIP Score ────────────────────────────────────────────────────────
        if compute_clip:
            logger.info("Running CLIP Score evaluation...")
            try:
                def clip_gen(prompts: list[str]):
                    # CLIP wants (prompts) -> images; the underlying generator
                    # is batch_size-driven, so derive the batch from the prompts.
                    return generator_fn(len(prompts), prompts=prompts)

                clip_score, per_sample = self.clip_evaluator.evaluate(
                    generator_fn=clip_gen,
                    prompts=self.eval_prompts,
                    batch_size=self.clip_batch_size,
                    return_per_sample=True,
                )
                results["eval/clip_score"] = clip_score
                results["eval/clip_score_std"] = float(np.std(per_sample))
            except Exception as e:
                logger.error(f"CLIP Score failed: {e}")
                results["eval/clip_score"] = float("nan")

        # ── Diversity ─────────────────────────────────────────────────────────
        if compute_diversity:
            logger.info("Computing image diversity...")
            try:
                with torch.no_grad():
                    sample_images = generator_fn(min(64, len(self.eval_prompts)))
                    if isinstance(sample_images, tuple):
                        sample_images = sample_images[0]
                diversity = self.clip_evaluator.compute_image_diversity(sample_images)
                results["eval/diversity"] = diversity
            except Exception as e:
                logger.error(f"Diversity failed: {e}")
                results["eval/diversity"] = float("nan")

        # ── Save results ──────────────────────────────────────────────────────
        results_file = self.results_dir / f"eval_step{step:07d}.json"
        with open(results_file, "w") as f:
            json.dump(
                {k: float(v) if isinstance(v, np.floating | float) else v
                 for k, v in results.items()},
                f, indent=2
            )

        logger.info(f"Evaluation results @ step {step}:")
        for k, v in results.items():
            if k != "step":
                logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return results

    def evaluate_from_folder(
        self,
        fake_image_dir: str,
        step: int,
        prompts_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate from a folder of pre-generated images.
        Expects: fake_image_dir/{00000.png, 00001.png, ..., prompts.txt}
        """
        fake_dir = Path(fake_image_dir)
        image_paths = sorted(fake_dir.glob("*.png")) + sorted(fake_dir.glob("*.jpg"))

        # Load as tensor dataset
        images_list = []
        for p in image_paths[:10000]:
            try:
                img = Image.open(p).convert("RGB")
                t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                images_list.append(t.permute(2, 0, 1))
            except Exception:
                continue

        images_tensor = torch.stack(images_list)
        idx = [0]

        def generator_fn(n, prompts=None):
            start = idx[0]
            batch = images_tensor[start: start + n].to(self.device)
            idx[0] += n
            return batch

        return self.evaluate_all(generator_fn, step)
