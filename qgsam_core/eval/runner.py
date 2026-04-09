"""
Evaluation runner: load checkpoint, run on dataset, log metrics.
"""
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from ..models import QGSAM
from .metrics import compute_vqa_accuracy, compute_evidence_metrics


def run_evaluation(
    model: QGSAM,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer=None,
) -> Dict[str, float]:
    model.eval()
    all_acc = []
    all_evidence = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            answer_idx = batch.get("answer_id", batch.get("answer_idx")).to(device)
            if batch.get("input_ids") is not None:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif tokenizer is not None:
                tok = tokenizer(batch["question"], padding="longest", truncation=True, max_length=64, return_tensors="pt")
                input_ids = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
            else:
                continue
            out = model(image, input_ids, attention_mask)
            acc = compute_vqa_accuracy(out["answer_logits"], answer_idx)
            all_acc.append(acc["accuracy"])
            ev = compute_evidence_metrics(out["mask_logits"], batch.get("gt_boxes"), batch.get("gt_masks"))
            all_evidence.append(ev)
    mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    mean_iou = sum(e["mean_iou"] for e in all_evidence) / len(all_evidence) if all_evidence else 0.0
    return {"accuracy": mean_acc, "mean_iou": mean_iou}
