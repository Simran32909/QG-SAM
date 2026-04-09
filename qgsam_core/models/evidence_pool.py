import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidencePool(nn.Module):
    def __init__(self, image_feat_dim: int, question_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(image_feat_dim + question_dim, out_dim)
        self.act = nn.GELU()

    def forward(
        self,
        image_features: torch.Tensor,
        mask_logits: torch.Tensor,
        question_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_features: (B, D, H, W)
        mask_logits: (B, K, H, W)
        question_embedding: (B, question_dim)
        Returns: (B, K, out_dim)
        """
        B, D, H, W = image_features.shape
        K = mask_logits.shape[1]
        weights = torch.sigmoid(mask_logits)
        pooled = []
        for k in range(K):
            w = weights[:, k : k + 1]
            p = (image_features * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1.0)
            pooled.append(p)
        pooled = torch.stack(pooled, dim=1)
        q = question_embedding.unsqueeze(1).expand(-1, K, -1)
        fused = torch.cat([pooled, q], dim=-1)
        return self.act(self.proj(fused))
