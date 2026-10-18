import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CLIPScoreEvaluator:
    """
    CLIP Score evaluator for image-text alignment.

    CLIP Score = w * max(cos_sim(CLIP_img(I), CLIP_txt(T)), 0)
    where w = 2.5 (scaling factor from Hessel et al. 2021)

    Measures how well generated images align with their prompts.
    Reference: Hessel et al. "CLIPScore: A Reference-Free Evaluation Metric
    for Image Captioning" (2021).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        w: float = 2.5,
    ):
        self.device = device
        self.w = w

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        logger.info(f"CLIPScore evaluator loaded: {model_name}")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to CLIP image embeddings.
        Args:
            images: (B, 3, H, W) in [-1, 1] or [0, 1]
        Returns:
            (B, 768) normalized embeddings
        """
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # CLIP preprocessing
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)
        images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text prompts to CLIP text embeddings.
        Returns:
            (B, 768) normalized embeddings
        """
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def compute_clip_score(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP score for paired (image, text) embeddings.
        Returns:
            (B,) scores in [0, 2.5]
        """
        cos_sim = (image_feats * text_feats).sum(dim=-1)  # (B,)
        return self.w * torch.clamp(cos_sim, min=0.0)

    def evaluate(
        self,
        generator_fn,
        prompts: list[str],
        batch_size: int = 32,
        return_per_sample: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """
        Full CLIP score evaluation.
        Args:
            generator_fn: callable(prompts) → (B, 3, H, W) images
            prompts: list of text prompts
        Returns:
            mean CLIP score (and optionally per-sample scores)
        """
        all_scores = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Computing CLIP Score"):
            batch_prompts = prompts[i : i + batch_size]
            batch_images = generator_fn(batch_prompts)
            if isinstance(batch_images, tuple):
                batch_images = batch_images[0]
            batch_images = batch_images.to(self.device)

            img_feats = self.encode_images(batch_images)
            txt_feats = self.encode_texts(batch_prompts)
            scores = self.compute_clip_score(img_feats, txt_feats)
            all_scores.append(scores.cpu().numpy())

        all_scores = np.concatenate(all_scores)
        mean_score = float(np.mean(all_scores))
        logger.info(f"CLIP Score: {mean_score:.4f}")

        if return_per_sample:
            return mean_score, all_scores
        return mean_score

    @torch.no_grad()
    def compute_image_diversity(
        self,
        images: torch.Tensor,
    ) -> float:
        """
        Measure diversity of generated images using pairwise CLIP cosine distance.
        Higher = more diverse.
        """
        feats = self.encode_images(images.to(self.device))  # (N, D)
        # Pairwise cosine similarity matrix
        sim_matrix = feats @ feats.T  # (N, N)
        # Exclude diagonal
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=self.device)
        diversity = 1.0 - sim_matrix[mask].mean().item()
        logger.info(f"Image diversity (1 - avg CLIP sim): {diversity:.4f}")
        return diversity
