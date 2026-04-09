"""
Question-aware segmentation: cross-attention from question to image features -> K evidence masks.
Uses frozen CLIP ViT as image encoder for pre-trained visual features.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel
except ImportError:
    CLIPVisionModel = None


class QuestionAwareSegmenter(nn.Module):
    """
    Frozen CLIP ViT image encoder + question-conditioned cross-attention -> K mask logits per spatial position.
    CLIP output: (B, N+1, D) patch embeddings (including [CLS]). We drop CLS token, reshape to spatial grid.
    """

    def __init__(
        self,
        image_feat_dim: int = 768,
        question_dim: int = 768,
        num_masks: int = 8,
        num_heads: int = 4,
        num_cross_layers: int = 1,
        clip_model: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__()
        self.num_masks = num_masks

        if CLIPVisionModel is None:
            raise ImportError(
                "transformers is required for CLIP-backed segmenter. Run: pip install transformers"
            )
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model)
        self.clip_encoder.requires_grad_(False)
        self.clip_hidden_dim = self.clip_encoder.config.hidden_size  # 768 for base
        self.patch_size = self.clip_encoder.config.patch_size  # 16
        num_patches = int(self.clip_encoder.config.image_size // self.patch_size)  # 14
        self.spatial_size = num_patches

        # Project CLIP features to cross-attention dim
        self.image_proj = nn.Linear(self.clip_hidden_dim, question_dim)

        cross_layer = nn.TransformerDecoderLayer(
            d_model=question_dim,
            nhead=num_heads,
            dim_feedforward=question_dim * 4,
            batch_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(cross_layer, num_layers=num_cross_layers)
        self.mask_head = nn.Linear(question_dim, num_masks)

    def forward(
        self,
        image: torch.Tensor,
        question_tokens: torch.Tensor,
        question_padding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image: (B, 3, H, W) — will be resized to CLIP's expected 224x224
        question_tokens: (B, L, D), question_padding: (B, L) 1=valid, 0=pad
        Returns:
            mask_logits_2d: (B, K, h, w) at CLIP's native 14x14 grid
            out: (B, N, D) cross-attention features
            image_feat_map: (B, D, h, w) at CLIP's native 14x14 grid (NOT upsampled)
        """
        B = image.shape[0]

        # Encode with CLIP (frozen) — always resize to 224 for CLIP
        clip_size = 224
        if image.shape[2] != clip_size or image.shape[3] != clip_size:
            clip_input = F.interpolate(image, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
        else:
            clip_input = image
        with torch.no_grad():
            clip_out = self.clip_encoder(pixel_values=clip_input)
            # Drop CLS token: (B, 197, 768) -> (B, 196, 768)
            patch_feats = clip_out.last_hidden_state[:, 1:]

        h = w = self.spatial_size  # 14x14 native CLIP grid

        # Project to cross-attention dim
        feat_flat = self.image_proj(patch_feats)  # (B, 196, question_dim)

        # Cross-attention: image patches attend to question tokens
        key_padding = (1 - question_padding).bool()
        out = self.cross_attn(feat_flat, question_tokens, memory_key_padding_mask=key_padding)  # (B, 196, D)

        # Mask logits at native 14x14 (no upsampling here — done by lightning_module for loss)
        mask_logits = self.mask_head(out)  # (B, 196, K)
        mask_logits_2d = mask_logits.permute(0, 2, 1).view(B, self.num_masks, h, w)  # (B, K, 14, 14)

        # Keep image_feat_map at native 14x14 — no giant 224x224 tensor!
        clip_dim = patch_feats.shape[2]  # 768
        image_feat_map = patch_feats.permute(0, 2, 1).view(B, clip_dim, h, w)  # (B, 768, 14, 14)

        return mask_logits_2d, out, image_feat_map
