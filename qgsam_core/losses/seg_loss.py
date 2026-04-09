"""
Segmentation loss: BCE + Dice for mask logits vs GT (if available).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_bce_dice_loss(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    mask_logits: (B, K, H, W)
    gt_masks: (B, 1, H, W) or (B, K, H, W) binary
    """
    if gt_masks is None or gt_masks.numel() == 0:
        return torch.tensor(0.0, device=mask_logits.device)
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)
        
    B, K, H, W = mask_logits.shape
    probs = torch.sigmoid(mask_logits)
    
    # Calculate losses per mask
    # Reshape for broadcasting: logits (B, K, H, W), GT (B, 1, H, W) -> loss (B, K)
    gt_expanded = gt_masks.expand(-1, K, -1, -1)
    
    # BCE per mask (per pixel, then mean over H and W)
    bce = F.binary_cross_entropy_with_logits(mask_logits, gt_expanded.float(), reduction='none')
    bce_per_mask = bce.mean(dim=(2, 3)) # (B, K)
    
    # Dice per mask
    smooth = 1e-5
    inter = (probs * gt_expanded).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + gt_expanded.sum(dim=(2, 3)) + smooth
    dice_per_mask = 1 - (2 * inter + smooth) / union # (B, K)
    
    # Combine and pick top-1 best matching mask per batch entry
    total_loss_per_mask = bce_per_mask + dice_per_mask
    best_loss, _ = torch.min(total_loss_per_mask, dim=1) # (B,)
    
    # Average across batch
    return best_loss.mean()
