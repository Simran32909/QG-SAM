"""
Evidence-only reasoning head: transformer over evidence embeddings -> answer logits.
No global image features; only evidence embeddings + optional question.
"""
from typing import Optional

import torch
import torch.nn as nn


class EvidenceReasoner(nn.Module):
    def __init__(
        self,
        evidence_dim: int,
        num_answers: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=evidence_dim,
            nhead=num_heads,
            dim_feedforward=evidence_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(evidence_dim, num_answers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, evidence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        evidence_embeddings: (B, K, D)
        Returns: (B, num_answers) logits
        """
        x = self.transformer(evidence_embeddings)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
