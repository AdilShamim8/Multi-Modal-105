import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    AutoTokenizer,
    AutoModel,
)
from typing import Optional, Dict, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder wrapper for text-to-image conditioning.
    Supports:
    - Standard CLIP (openai/clip-vit-large-patch14) — 768d
    - OpenCLIP variants — up to 1280d
    - Penultimate layer extraction (SD-style)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        use_penultimate_layer: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.use_penultimate_layer = use_penultimate_layer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

        if freeze:
            for p in self.text_model.parameters():
                p.requires_grad_(False)
            self.text_model.eval()

        self.embed_dim = self.text_model.config.hidden_size
        logger.info(
            f"CLIPTextEncoder loaded: {model_name}, "
            f"embed_dim={self.embed_dim}, "
            f"penultimate={use_penultimate_layer}"
        )

    def tokenize(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if self.use_penultimate_layer:
            # SD uses second-to-last hidden state
            return out.hidden_states[-2]      # (B, 77, 768)
        return out.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)

    def encode_texts(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convenience method: tokenize + encode in one call."""
        tokens = self.tokenize(texts, device=device)
        return self.encode(tokens["input_ids"], tokens.get("attention_mask"))


class DualTextEncoder(nn.Module):
    """
    Dual text encoder (SDXL-style): CLIP-L + OpenCLIP-bigG.
    Concatenates embeddings along feature dimension for richer conditioning.
    Output dim = 768 + 1280 = 2048 (pooled) or seq embeddings are cat.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        open_clip_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        max_length: int = 77,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder1 = CLIPTextEncoder(clip_model, max_length, freeze=freeze)
        self.encoder2 = CLIPTextEncoder(open_clip_model, max_length, freeze=freeze)
        self.embed_dim = self.encoder1.embed_dim + self.encoder2.embed_dim

    @torch.no_grad()
    def forward(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prompt_embeds: (B, 77, 2048) concatenated sequence embeddings
            pooled_embeds: (B, 1280) pooled from encoder2 for time conditioning
        """
        emb1 = self.encoder1(input_ids1, mask1)      # (B, 77, 768)
        emb2 = self.encoder2(input_ids2, mask2)      # (B, 77, 1280)
        prompt_embeds = torch.cat([emb1, emb2], dim=-1)  # (B, 77, 2048)

        # Pooled: mean of non-padding tokens from encoder2
        pooled = emb2.mean(dim=1)                    # (B, 1280)
        return prompt_embeds, pooled


class T5TextEncoder(nn.Module):
    """
    T5 text encoder for Imagen / Flux-style conditioning.
    Handles long captions better than CLIP (no 77-token limit).
    """

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-xl",
        max_length: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embed_dim = self.encoder.config.d_model

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state    # (B, seq_len, embed_dim)