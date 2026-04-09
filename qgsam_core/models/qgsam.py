"""
Question -> QuestionEncoder (BERT); Image + Question -> QuestionAwareSegmenter (CLIP ViT) -> masks + image_feat_map;
EvidencePool(mask_logits, image_feat_map, question_embed) -> evidence_embeddings;
EvidenceReasoner(evidence_embeddings) -> answer logits.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .question_encoder import QuestionEncoder
from .segmenter import QuestionAwareSegmenter
from .evidence_pool import EvidencePool
from .reasoner import EvidenceReasoner


class QGSAM(nn.Module):
    def __init__(
        self,
        num_answers: int = 1000,
        hidden_size: int = 512,
        image_feat_dim: int = 768,
        question_dim: int = 768,
        num_masks: int = 8,
        num_heads: int = 8,
        num_cross_layers: int = 1,
        num_reasoner_layers: int = 2,
        dropout: float = 0.1,
        clip_model: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__()
        self.num_masks = num_masks
        self.question_encoder = QuestionEncoder(
            question_dim=question_dim,
        )
        self.segmenter = QuestionAwareSegmenter(
            image_feat_dim=image_feat_dim,
            question_dim=question_dim,
            num_masks=num_masks,
            num_heads=num_heads,
            num_cross_layers=num_cross_layers,
            clip_model=clip_model,
        )
        self.evidence_pool = EvidencePool(
            image_feat_dim=image_feat_dim,
            question_dim=question_dim,
            out_dim=hidden_size,
        )
        self.reasoner = EvidenceReasoner(
            evidence_dim=hidden_size,
            num_answers=num_answers,
            num_heads=num_heads,
            num_layers=num_reasoner_layers,
            dropout=dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        image: (B, 3, H, W)
        input_ids: (B, L), attention_mask: (B, L)
        Returns: dict with answer_logits, mask_logits, evidence_embeddings
        """
        q_pooled, q_tokens = self.question_encoder(input_ids, attention_mask)
        mask_logits, mask_features, image_feat_map = self.segmenter(image, q_tokens, attention_mask)
        evidence_emb = self.evidence_pool(image_feat_map, mask_logits, q_pooled)
        answer_logits = self.reasoner(evidence_emb)
        return {
            "answer_logits": answer_logits,
            "mask_logits": mask_logits,
            "evidence_embeddings": evidence_emb,
        }
