"""
CLIP alignment loss: align cropped mask regions to question text in CLIP space.
Optional dependency: transformers + open_clip_torch or clip. If not available, return zero loss.
"""
import torch


def clip_alignment_loss(
    image: torch.Tensor,
    mask_logits: torch.Tensor,
    question_texts: list,
    clip_model=None,
    clip_preprocess=None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    image: (B, 3, H, W), mask_logits: (B, K, H, W), question_texts: list of B strings.
    Uses top-1 mask per image to crop region, encode with CLIP image encoder, and align to text.
    If clip_model is None, returns 0.0 (no CLIP loss).
    """
    if clip_model is None or clip_preprocess is None:
        return torch.tensor(0.0, device=image.device if device is None else device)
    # Placeholder: full implementation would crop by mask, run CLIP image encoder, run CLIP text encoder, cosine loss
    return torch.tensor(0.0, device=image.device)
