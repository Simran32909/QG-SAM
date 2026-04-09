"""
PyTorch Lightning module for QG-SAM training.
Multi-loss: VQA CE + optional segmentation + optional CLIP alignment.
"""
from typing import Any, Dict, Optional
from collections import Counter

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..models import QGSAM
from ..losses import seg_bce_dice_loss, clip_alignment_loss


class QGSAMLightning(pl.LightningModule):
    def __init__(
        self,
        model: QGSAM,
        num_answers: int,
        tokenizer=None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        use_seg_loss: bool = True,
        use_clip_loss: bool = False,
        seg_loss_weight: float = 1.0,
        clip_loss_weight: float = 0.1,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        idx_to_answer: Optional[list] = None,
        clip_model=None,
        clip_preprocess=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "clip_model", "clip_preprocess"])
        self.model = model
        self.tokenizer = tokenizer
        self.num_answers = num_answers
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_seg_loss = use_seg_loss
        self.use_clip_loss = use_clip_loss
        self.seg_loss_weight = seg_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.idx_to_answer = idx_to_answer or []
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.ce = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=self.label_smoothing,
            weight=self.class_weights,
        )
        self._train_pred_counter = Counter()
        self._val_pred_counter = Counter()
        self._bert_tokenizer = None

    def _get_bert_tokenizer(self):
        if self._bert_tokenizer is None:
            from transformers import AutoTokenizer
            self._bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return self._bert_tokenizer

    def forward(self, image, input_ids, attention_mask, gt_boxes=None):
        return self.model(image, input_ids, attention_mask, gt_boxes)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        answer_id = batch.get("answer_id", batch.get("answer_idx"))
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        gt_boxes = batch.get("gt_boxes")
        if input_ids is None and "question" in batch:
            tok = self._get_bert_tokenizer()(
                batch["question"],
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        answer_id = answer_id.to(self.device)
        out = self.model(image, input_ids, attention_mask, gt_boxes)
        loss_ce = self.ce(out["answer_logits"], answer_id)
        
        # NaN guard: skip corrupted batches entirely instead of exploding weights
        if not torch.isfinite(loss_ce):
            return None
            
        loss = loss_ce
        pred = out["answer_logits"].argmax(dim=1)
        top5 = torch.topk(out["answer_logits"], k=min(5, out["answer_logits"].size(1)), dim=1).indices
        top1_acc = (pred == answer_id).float().mean()
        top5_acc = (top5 == answer_id.unsqueeze(1)).any(dim=1).float().mean()
        self._train_pred_counter.update(pred.detach().cpu().tolist())
        self.log("train_loss_ce", loss_ce, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc_top1", top1_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_acc_top5", top5_acc, on_step=True, on_epoch=True, prog_bar=False)
        if self.use_seg_loss and gt_boxes is not None and len(gt_boxes) > 0:
            gt_masks = self._boxes_to_masks(gt_boxes, image.shape)
            
            # Only compute loss on images that actually have boxes annotated
            has_box_mask = torch.tensor([(b is not None and len(b) > 0) for b in gt_boxes], device=self.device)
            
            if has_box_mask.any():
                loss_seg = seg_bce_dice_loss(out["mask_logits"][has_box_mask], gt_masks[has_box_mask])
                if torch.isfinite(loss_seg):  # skip NaN seg loss too
                    loss = loss + self.seg_loss_weight * loss_seg
                    self.log("train_loss_seg", loss_seg, on_step=True, on_epoch=True)
        if self.use_clip_loss and self.clip_model is not None:
            questions = batch.get("question", [])
            loss_clip = clip_alignment_loss(
                image, out["mask_logits"], questions,
                self.clip_model, self.clip_preprocess, self.device,
            )
            loss = loss + self.clip_loss_weight * loss_clip
            self.log("train_loss_clip", loss_clip, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _boxes_to_masks(self, gt_boxes, image_shape):
        # Masks now live at native CLIP 14x14 grid (matches segmenter output)
        H = W = 14
        device = next(self.model.parameters()).device
        B = image_shape[0]
        masks = torch.zeros(B, 1, H, W, device=device)
        for b in range(B):
            if b < len(gt_boxes) and gt_boxes[b] is not None and len(gt_boxes[b]) > 0:
                boxes = gt_boxes[b]
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                for box in boxes:
                    x1, y1, x2, y2 = (box * torch.tensor([W, H, W, H], device=device)).long()
                    x1, x2 = max(0, x1.item()), min(W, x2.item())
                    y1, y2 = max(0, y1.item()), min(H, y2.item())
                    if x2 > x1 and y2 > y1:
                        masks[b, 0, y1:y2, x1:x2] = 1.0
        return masks

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        answer_id = batch.get("answer_id", batch.get("answer_idx"))
        if batch.get("input_ids") is not None:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
        elif "question" in batch:
            tok = self._get_bert_tokenizer()(batch["question"], padding="longest", truncation=True, max_length=64, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
        else:
            input_ids = batch.get("question_ids")
            attention_mask = batch.get("question_mask")
            if input_ids is None:
                return {}
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        answer_id = answer_id.to(self.device)
        out = self.model(image, input_ids, attention_mask)
        loss_ce = self.ce(out["answer_logits"], answer_id)
        pred = out["answer_logits"].argmax(dim=1)
        top5 = torch.topk(out["answer_logits"], k=min(5, out["answer_logits"].size(1)), dim=1).indices
        top1_acc = (pred == answer_id).float().mean()
        top5_acc = (top5 == answer_id.unsqueeze(1)).any(dim=1).float().mean()
        self._val_pred_counter.update(pred.detach().cpu().tolist())
        self.log("val_loss", loss_ce, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_top1", top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_top5", top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss_ce, "val_acc_top1": top1_acc, "val_acc_top5": top5_acc}

    def on_train_epoch_start(self) -> None:
        self._train_pred_counter = Counter()

    def on_validation_epoch_start(self) -> None:
        self._val_pred_counter = Counter()

    def _format_top_predictions(self, pred_counter: Counter, top_n: int = 5) -> str:
        top_items = pred_counter.most_common(top_n)
        if not top_items:
            return "none"
        pretty = []
        for idx, count in top_items:
            label = str(idx)
            if 0 <= idx < len(self.idx_to_answer):
                label = self.idx_to_answer[idx]
            pretty.append(f"{label}:{count}")
        return ", ".join(pretty)

    def on_train_epoch_end(self) -> None:
        self.print(f"[train] top predicted answers: {self._format_top_predictions(self._train_pred_counter)}")

    def on_validation_epoch_end(self) -> None:
        self.print(f"[val] top predicted answers: {self._format_top_predictions(self._val_pred_counter)}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Cosine annealing: smoothly decays LR from lr -> 1e-6 over 30 epochs
        # This eliminates the high-variance oscillations seen with fixed LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=30, eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
