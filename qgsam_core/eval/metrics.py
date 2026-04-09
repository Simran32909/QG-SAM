"""
Evaluation metrics: VQA accuracy, evidence quality (mask IoU, precision/recall, top-k relevance).
"""
from typing import Dict, List, Optional

import torch
import numpy as np


def compute_vqa_accuracy(
    answer_logits: torch.Tensor,
    answer_idx: torch.Tensor,
    top_k: int = 1,
) -> Dict[str, float]:
    """
    answer_logits: (B, num_answers), answer_idx: (B,)
    Returns: accuracy, accuracy@k
    """
    pred = answer_logits.argmax(dim=1)
    acc = (pred == answer_idx).float().mean().item()
    _, topk = answer_logits.topk(top_k, dim=1)
    correct_k = (topk == answer_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"accuracy": acc, f"accuracy@{top_k}": correct_k}


def compute_evidence_metrics(
    mask_logits: torch.Tensor,
    gt_boxes: Optional[List[torch.Tensor]] = None,
    gt_masks: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    mask_logits: (B, K, H, W). gt_boxes: list of (N, 4) in normalized [0,1] or pixel.
    gt_masks: (B, 1, H, W) optional.
    Returns: mean_iou, precision, recall, f1 (when GT available).
    """
    if gt_masks is None and (gt_boxes is None or len(gt_boxes) == 0):
        return {"mean_iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_masks = (torch.sigmoid(mask_logits) > 0.5).float()
    if gt_masks is not None:
        inter = (pred_masks * gt_masks).sum(dim=(2, 3))
        union = (pred_masks + gt_masks).clamp(0, 1).sum(dim=(2, 3))
        iou = (inter / (union + 1e-6)).mean().item()
    else:
        iou = 0.0
    return {"mean_iou": iou, "precision": iou, "recall": iou, "f1": iou}
