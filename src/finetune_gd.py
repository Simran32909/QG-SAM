#script for finetuning groundingdino

import os
import sys
import argparse
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json
import wandb
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.console import Console
from torch.cuda.amp import autocast, GradScaler
import math
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

DINO_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/GroundingDINO"
sys.path.append(DINO_PATH)

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import clean_state_dict, NestedTensor, nested_tensor_from_tensor_list
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss

# Predefined captions for polyps with multiple variations
POLYP_CAPTIONS = [
    "a medical polyp in the intestine", 
    "intestinal polyp detected", 
    "polyp in medical imaging",
    "a polyp visible in the scan",
    "medical image showing a polyp",
    "endoscopic view of a polyp",
    "polyp detected in gastrointestinal tract"
]

def build_transform(is_train):
    if is_train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class KvasirDinoDataset(Dataset):
    def __init__(self, image_dir, ann_file, image_ids, tokenizer, is_train=False):
        self.image_dir = image_dir
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = image_ids
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.transform = build_transform(is_train)
        self.max_text_len = min(getattr(self.tokenizer, "model_max_length", 256), 256)
        self.positive_keyword = "polyp"
        
        # Special tokens for object delimiting
        self.obj_start_token = '<obj>'
        self.obj_end_token = '</obj>'

        # Use class-level predefined captions
        self.captions = POLYP_CAPTIONS

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        ann = self.annotations[image_id]
        width, height = ann['width'], ann['height']
        bboxes_info = ann.get("bbox", [])

        # Skip images without annotations
        if len(bboxes_info) == 0:
            return None

        bboxes = []
        for bbox in bboxes_info:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            # Convert to center, width, height
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            # Normalize
            bboxes.append([cx / width, cy / height, w / width, h / height])
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        
        # Ensure we always have a caption
        raw_caption = random.choice(self.captions)
        
        # Wrap caption with object tokens
        caption = f"{self.obj_start_token} {raw_caption} {self.obj_end_token}"
        
        # Apply transforms
        # The transform function from torchvision now returns the transformed image directly
        image_tensor = self.transform(image)
        
        # Prepare target
        target = {
            'boxes': bboxes,
            'labels': torch.zeros(len(bboxes), dtype=torch.long) # Placeholder for labels
        }
        
        # Tokenize caption with error handling
        tokenized = None
        try:
            tokenized = self.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            if tokenized['input_ids'].numel() == 0:
                raise ValueError("Tokenized caption is empty")
        except Exception as e:
            print(f"Tokenization error: {e}. Using fallback caption.")
            caption = f"{self.obj_start_token} polyp in medical image {self.obj_end_token}"
            tokenized = self.tokenizer(
                caption,
                padding="longest",
                return_tensors="pt",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )

        positive_map = self._build_positive_map(tokenized, len(bboxes), caption)
        tokenized.pop("offset_mapping", None)

        return {
            'image': image_tensor,
            'caption': caption,
            'boxes': bboxes,
            'tokenized': tokenized,
            'target': target,
            'positive_map': positive_map
        }

    def _build_positive_map(self, tokenized, num_boxes, caption):
        if num_boxes == 0:
            return torch.zeros((0, self.max_text_len), dtype=torch.float32)

        keyword = self.positive_keyword.lower()
        caption_lower = caption.lower()
        spans = []
        matches = [m for m in re.finditer(keyword, caption_lower)]
        if not matches:
            spans = [[[0, len(caption)]] for _ in range(num_boxes)]
        else:
            span_list = [[match.start(), match.end()] for match in matches]
            spans = [span_list.copy() for _ in range(num_boxes)]

        try:
            positive_map = create_positive_map_from_span(
                tokenized,
                spans,
                max_text_len=self.max_text_len
            )
        except Exception:
            seq_len = tokenized['attention_mask'].shape[1]
            valid_len = min(seq_len, self.max_text_len)
            base_map = torch.zeros((num_boxes, self.max_text_len), dtype=torch.float32)
            base_map[:, :valid_len] = 1.0 / max(valid_len, 1)
            positive_map = base_map

        return positive_map.float()

def collate_fn(batch, tokenizer):
    # Filter out None values (images without annotations)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # Return empty batch if no valid samples
        return torch.empty(0), [], [], {}, []
    
    images = torch.stack(
        [item['image'] for item in batch]
    )
    captions = [item['caption'] for item in batch]
    
    # Collect boxes
    targets = []
    for item in batch:
        targets.append({
            'boxes': item['boxes'],
            'positive_map': item['positive_map']
        })
    
    # Prepare tokenized inputs with padding
    max_len = max(item['tokenized']['input_ids'].shape[1] for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    
    for item in batch:
        curr_input_ids = item['tokenized']['input_ids']
        curr_attention_mask = item['tokenized']['attention_mask']
        
        # Pad input_ids and attention_mask to max_len
        pad_len = max_len - curr_input_ids.shape[1]
        
        padded_input_ids = F.pad(curr_input_ids, (0, pad_len), value=0)
        padded_attention_mask = F.pad(curr_attention_mask, (0, pad_len), value=0)
        
        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
    
    # Stack padded tensors
    tokenized_inputs = {
        'input_ids': torch.cat(input_ids_list, dim=0),
        'attention_mask': torch.cat(attention_mask_list, dim=0)
    }
    
    # Get special token ids
    obj_start_id = tokenizer.convert_tokens_to_ids('<obj>')
    obj_end_id = tokenizer.convert_tokens_to_ids('</obj>')
    
    # Default special tokens list
    special_tokens_list = [obj_start_id, obj_end_id]
    
    return images, captions, targets, tokenized_inputs, special_tokens_list


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for Hungarian matching. Please install scipy.")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_boxes = outputs.get("pred_boxes")
        pred_logits = outputs.get("pred_logits")

        if pred_boxes is None or pred_logits is None:
            device = pred_boxes.device if pred_boxes is not None else torch.device("cpu")
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return [(empty, empty) for _ in targets]

        pred_boxes = pred_boxes.float()
        pred_logits = pred_logits.float()

        bs, num_queries = pred_boxes.shape[:2]
        pred_probs = pred_logits.sigmoid()
        indices = []

        for b in range(bs):
            tgt_boxes = targets[b]['boxes']
            if tgt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_boxes.device)
                indices.append((empty, empty))
                continue

            tgt_boxes = tgt_boxes.to(pred_boxes.device).float()
            tgt_pos_map = targets[b]['positive_map'].to(pred_logits.device, dtype=pred_logits.dtype)

            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)
            out_bbox_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
            cost_class = -(pred_probs[b] @ tgt_pos_map.T)

            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class
            C = C.cpu()
            row_ind, col_ind = linear_sum_assignment(C)
            src_idx = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_boxes.device)
            tgt_idx = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_boxes.device)
            indices.append((src_idx, tgt_idx))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha=0.25, focal_gamma=2.0, max_text_len=256, label_smoothing=0.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_text_len = max_text_len
        self.label_smoothing = label_smoothing

    def _get_src_permutation_idx(self, indices, device):
        batch_idx = []
        src_idx = []
        for b, (src, _) in enumerate(indices):
            if src.numel() == 0:
                continue
            batch_idx.append(torch.full((len(src),), b, dtype=torch.int64, device=src.device))
            src_idx.append(src)
        if len(batch_idx) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty
        return torch.cat(batch_idx), torch.cat(src_idx)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.clamp(src_logits, min=-20.0, max=20.0)
        target_classes = torch.zeros_like(src_logits)

        batch_idx, src_idx = self._get_src_permutation_idx(indices, src_logits.device)
        if src_idx.numel() > 0:
            matched_pos = []
            for b, (_, tgt_idx) in enumerate(indices):
                if tgt_idx.numel() == 0:
                    continue
                matched_pos.append(
                    targets[b]['positive_map'][tgt_idx].to(
                        device=src_logits.device,
                        dtype=src_logits.dtype
                    )
                )
            if matched_pos:
                target_pos = torch.cat(matched_pos, dim=0)
                target_pos = torch.nan_to_num(target_pos, nan=0.0, posinf=0.0, neginf=0.0)
                target_classes[batch_idx, src_idx] = target_pos

        target_classes = torch.nan_to_num(target_classes, nan=0.0, posinf=0.0, neginf=0.0)
        if self.label_smoothing > 0:
            target_classes = target_classes * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss_ce = sigmoid_focal_loss(
            src_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            num_boxes=num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        pred_boxes = outputs["pred_boxes"].float()
        if pred_boxes.numel() == 0:
            zero = torch.tensor(0.0, device=pred_boxes.device)
            return {"loss_bbox": zero, "loss_giou": zero}

        batch_idx, src_idx = self._get_src_permutation_idx(indices, pred_boxes.device)
        if src_idx.numel() == 0:
            zero = torch.tensor(0.0, device=pred_boxes.device)
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = pred_boxes[batch_idx, src_idx]
        target_boxes = torch.cat(
            [targets[b]["boxes"][tgt_idx] for b, (_, tgt_idx) in enumerate(indices) if tgt_idx.numel() > 0],
            dim=0,
        ).to(src_boxes.device).float()

        loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        if len(targets) == 0:
            zero = torch.tensor(0.0, device=outputs["pred_boxes"].device)
            return {"loss_ce": zero, "loss_bbox": zero, "loss_giou": zero}

        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.tensor(float(max(num_boxes, 1)), device=outputs["pred_boxes"].device)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if pred_logits is not None and pred_boxes is not None:
            reg_logits = pred_logits.float().pow(2).mean()
            reg_boxes = pred_boxes.float().pow(2).mean()
            losses["loss_reg"] = reg_logits + reg_boxes
        return losses

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes in [cx, cy, w, h] format.
    """
    # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    # Compute IoU
    iou = inter_area / (union_area + 1e-6)
    return iou.item()

def compute_validation_metrics(outputs, targets, matcher, iou_threshold=0.5, score_threshold=0.3):
    """
    Compute detailed validation metrics for a batch.

    - Filters predictions by a score threshold derived from classification logits
    - Uses IoU threshold for deciding matches between GT and predicted boxes
    """
    pred_boxes = outputs.get('pred_boxes', None)
    pred_logits = outputs.get('pred_logits', None)

    if pred_boxes is None or pred_logits is None:
        return {
            'iou': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'matches': 0,
            'num_preds': 0,
            'num_gt': 0,
            'num_matched_pairs': 0,
            'avg_matched_score': 0.0,
            'avg_matched_iou': 0.0,
            'num_passed_score': 0,
            'num_passed_iou': 0
        }

    total_matches = 0
    total_preds = 0
    total_gt = 0
    total_iou = 0
    
    # Diagnostic counters
    total_matched_pairs = 0
    matched_scores = []
    matched_ious = []
    num_passed_score = 0
    num_passed_iou = 0

    indices = matcher(outputs, targets)

    for i, target in enumerate(targets):
        gt_boxes = target['boxes']
        sample_pred_boxes = pred_boxes[i]
        sample_pred_logits = pred_logits[i]

        # Classification scores for diagnostics
        scores = sample_pred_logits.sigmoid().max(dim=1)[0]
        keep = scores > score_threshold

        num_gt = len(gt_boxes)
        total_gt += num_gt
        # Diagnostics: how many predictions pass the score threshold (not used for main metrics)
        num_passed_score += int(keep.sum().item())

        if num_gt == 0:
            continue

        src_idx, tgt_idx = indices[i]
        if src_idx.numel() == 0:
            continue

        # For main detection metrics, treat all matched predictions as "predictions"
        num_pred = len(src_idx)
        total_preds += num_pred

        # Collect diagnostics for matched pairs and compute IoU-based matches
        for s_idx, t_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
            total_matched_pairs += 1
            matched_score = scores[s_idx].item()
            matched_scores.append(matched_score)

            pred_box = sample_pred_boxes[s_idx]
            gt_box = gt_boxes[t_idx]
            iou = compute_iou(gt_box, pred_box)
            matched_ious.append(iou)

            if iou >= iou_threshold:
                num_passed_iou += 1
                # Count this as a successful detection for precision/recall/IoU
                matches = 1
                total_matches += matches
                total_iou += iou

    precision = total_matches / (total_preds + 1e-6) if total_preds > 0 else 0.0
    recall = total_matches / (total_gt + 1e-6) if total_gt > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / (total_matches + 1e-6) if total_matches > 0 else 0.0
    
    # Diagnostic averages
    avg_matched_score = np.mean(matched_scores) if matched_scores else 0.0
    avg_matched_iou = np.mean(matched_ious) if matched_ious else 0.0

    return {
        'iou': avg_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'matches': total_matches,
        'num_preds': total_preds,
        'num_gt': total_gt,
        'num_matched_pairs': total_matched_pairs,
        'avg_matched_score': avg_matched_score,
        'avg_matched_iou': avg_matched_iou,
        'num_passed_score': num_passed_score,
        'num_passed_iou': num_passed_iou
    }

def sanity_check_dataset(dataset):
    """
    Perform comprehensive sanity checks on the dataset using a progress bar
    """
    print("\n--- Dataset Sanity Check ---")
    
    # Check dataset length
    total_samples = len(dataset)
    print(f"Total number of samples: {total_samples}")
    
    # Use rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn()
    ) as progress:
        # Create a task for dataset inspection
        task = progress.add_task("[green]Inspecting dataset samples...", total=total_samples)
        
        # Sample a few items for detailed inspection
        num_samples_to_check = min(5, total_samples)
        for i in range(num_samples_to_check):
            try:
                sample = dataset[i]
                
                # Basic validation checks
                assert 'image' in sample, "Missing image tensor"
                assert 'caption' in sample, "Missing caption"
                assert 'boxes' in sample, "Missing bounding boxes"
                assert 'tokenized' in sample, "Missing tokenization"
                
                # Update progress
                progress.update(task, advance=1, description=f"[green]Checked sample {i+1}/{num_samples_to_check}")
            
            except Exception as e:
                progress.console.print(f"[red]Error processing sample {i}: {e}")
                progress.update(task, advance=1, description=f"[red]Error in sample {i}")
    
    print("\n--- Dataset Sanity Check Complete ---")

def sanity_check_dataloader(dataloader, tokenizer):
    """
    Perform comprehensive sanity checks on the dataloader using a progress bar
    """
    print("\n--- DataLoader Sanity Check ---")
    
    # Use rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn()
    ) as progress:
        # Create tasks for different checks
        batch_task = progress.add_task("[green]Checking batches...", total=len(dataloader))
        
        # Check batch generation
        for batch_idx, (images, captions, targets, tokenized_inputs, special_tokens_list) in enumerate(dataloader):
            try:
                # Batch size and shapes validation
                assert images is not None, "Images tensor is None"
                assert len(captions) > 0, "No captions in batch"
                assert len(targets) > 0, "No targets in batch"
                
                # Tokenized inputs validation
                assert 'input_ids' in tokenized_inputs, "Missing input_ids"
                assert 'attention_mask' in tokenized_inputs, "Missing attention_mask"
                
                # Update progress
                progress.update(batch_task, advance=1, description=f"[green]Checked batch {batch_idx+1}/{len(dataloader)}")
                
                # Limit to first few batches for brevity
                if batch_idx >= 4:
                    break
            
            except Exception as e:
                progress.console.print(f"[red]Error in batch {batch_idx}: {e}")
                progress.update(batch_task, advance=1, description=f"[red]Error in batch {batch_idx}")
    
    print("\n--- DataLoader Sanity Check Complete ---")

def main(args):
    # --- Console and GPU Setup ---
    console = Console()
    if torch.cuda.is_available() and args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = list(range(torch.cuda.device_count()))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        multi_gpu = len(gpu_ids) > 1
        print(f"--- Setting up experiment on device: {device} ---")
        if multi_gpu:
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    else:
        device = torch.device("cpu")
        multi_gpu = False
        print("--- Setting up experiment on CPU ---")

    # Load model configuration
    cfg = SLConfig.fromfile(args.config_file)
    model = build_model(cfg)
    
    # Disable gradient checkpointing to resolve backward pass errors
    print("Disabling gradient checkpointing for training stability.")
    model.transformer.encoder.gradient_checkpointing = False
    model.transformer.decoder.gradient_checkpointing = False

    # Load pre-trained weights
    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    
    # Add special tokens to tokenizer
    special_tokens = ['<obj>', '</obj>']
    num_added = model.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added} additional special tokens: {special_tokens}")

    # Define special token IDs globally
    global OBJ_START_ID, OBJ_END_ID
    OBJ_START_ID = model.tokenizer.convert_tokens_to_ids('<obj>')
    OBJ_END_ID = model.tokenizer.convert_tokens_to_ids('</obj>')
    print(f"Special token IDs: <obj>={OBJ_START_ID}, </obj>={OBJ_END_ID}")

    # Verify special token handling
    print("\n--- Tokenization Sanity Check ---")
    # Predefined captions for verification
    sanity_captions = [
        "a medical polyp in the intestine", 
        "polyp in medical imaging",
        f"<obj> a medical polyp in the intestine </obj>",
        f"<obj> polyp in medical imaging </obj>"
    ]
    
    for caption in sanity_captions:
        print(f"\nCaption: '{caption}'")
        try:
            # Tokenize without special tokens
            tok = model.tokenizer(caption, return_tensors="pt")
            tokens = model.tokenizer.convert_ids_to_tokens(tok['input_ids'][0].tolist())
            print("Tokens:", tokens)
            
            # Print token IDs
            print("Token IDs:", tok['input_ids'][0].tolist())
            
            # Check if special tokens are in the tokenized output
            special_tokens_present = any(tid in [OBJ_START_ID, OBJ_END_ID] for tid in tok['input_ids'][0].tolist())
            print("Special tokens present:", special_tokens_present)
        except Exception as e:
            print(f"Tokenization error: {e}")
    
    print("\n--- Sanity Check Complete ---")

    # Resize model text embeddings with more robust approach
    try:
        # Try multiple methods to resize embeddings
        if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
            # Some models have a nested bert
            model.text_encoder.bert.embeddings.word_embeddings = nn.Embedding(
                len(model.tokenizer), 
                model.text_encoder.bert.embeddings.word_embeddings.embedding_dim
            )
            print("Resized embeddings via text_encoder.bert")
        elif hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'embeddings'):
            # Direct embeddings access
            model.text_encoder.embeddings.word_embeddings = nn.Embedding(
                len(model.tokenizer), 
                model.text_encoder.embeddings.word_embeddings.embedding_dim
            )
            print("Resized embeddings via text_encoder.embeddings")
        elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
            # Direct bert embeddings
            model.bert.embeddings.word_embeddings = nn.Embedding(
                len(model.tokenizer), 
                model.bert.embeddings.word_embeddings.embedding_dim
            )
            print("Resized embeddings via bert.embeddings")
        else:
            # Fallback: print model structure for debugging
            print("Could not resize embeddings. Model structure:")
            print(model)
            raise AttributeError("No known method to resize token embeddings")
    except Exception as e:
        print(f"Error resizing embeddings: {e}")
        print("Continuing without resizing. This may cause issues.")

    model.to(device)
    if multi_gpu:
        model = nn.Parallel(model, device_ids=gpu_ids)
    print("Model loaded successfully with pre-trained weights.")

    # Initialize W&B
    wandb.init(project=args.wandb_project, config=args)
    # Watch the model parameters, but NOT the gradients, to prevent conflicts.
    wandb.watch(model, log="parameters", log_freq=100)

    # --- Data Loading ---
    print("\n--- Loading Data ---")
    # Get image IDs for train/val splits from the directory structure
    train_img_dir = Path(args.data_path) / "train" / "images"
    val_img_dir = Path(args.data_path) / "val" / "images"
    
    train_image_ids = sorted([p.stem for p in train_img_dir.glob("*.jpg")])
    val_image_ids = sorted([p.stem for p in val_img_dir.glob("*.jpg")])
    
    # Verify we have training data
    if len(train_image_ids) == 0 or len(val_image_ids) == 0:
        raise ValueError("No training or validation images found. Check your dataset paths.")
    
    print(f"Found {len(train_image_ids)} training images and {len(val_image_ids)} validation images.")

    # Create datasets
    train_dataset = KvasirDinoDataset(train_img_dir, args.ann_path, train_image_ids, model.tokenizer, is_train=True)
    val_dataset = KvasirDinoDataset(val_img_dir, args.ann_path, val_image_ids, model.tokenizer, is_train=False)

    if args.sanity_check:
        print("Running sanity checks...")
        sanity_check_dataset(train_dataset)
        sanity_check_dataset(val_dataset)
        sanity_check_dataloader(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, model.tokenizer), num_workers=4), model.tokenizer)
        sanity_check_dataloader(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, model.tokenizer), num_workers=4), model.tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, model.tokenizer), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, model.tokenizer), num_workers=4)
    
    # --- Optimizer and Scheduler ---
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine scheduler stepped every optimizer update
    total_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.grad_accumulation_steps))
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

    def lr_lambda(current_step):
        current_step = max(0, current_step)
        if current_step < args.warmup_steps:
            warmup_progress = current_step / max(1, args.warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * warmup_progress
        progress = (current_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_reg": args.reg_weight}
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        focal_gamma=2.0,
        max_text_len=getattr(model, "max_text_len", 256),
        label_smoothing=args.label_smoothing
    ).to(device)
    
    # --- Training Loop ---
    scaler = GradScaler(enabled=not args.disable_amp)
    best_val_iou = -1.0
    
    with Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss]}"),
        transient=True,
    ) as progress:
        train_task = progress.add_task(f"[green]Training...", total=len(train_loader) * args.epochs, loss="")
        val_task = progress.add_task("[cyan]Validation", total=len(val_loader) * args.epochs, loss="", visible=False)

        for epoch in range(args.epochs):
            # --- Training Phase ---
            progress.update(train_task, description=f"[green]Epoch {epoch+1}/{args.epochs} Training...")
            progress.update(val_task, visible=False)
            model.train()
            train_loss_sum = 0
            
            # Reset optimizer gradients at the start of the epoch
            optimizer.zero_grad()

            for i, (images, captions, targets, tokenized_inputs, special_tokens_list) in enumerate(train_loader):

                if len(images) == 0:
                    continue
                    
                images = images.to(device)
                for t in targets:
                    t['boxes'] = t['boxes'].to(device)
                    if 'positive_map' in t:
                        t['positive_map'] = t['positive_map'].to(device)

                tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
                samples = NestedTensor(images, mask="auto")
                
                # --- AMP: Forward pass with autocasting ---
                with autocast():
                    outputs = model(
                        samples,
                        captions=captions,
                        tokenized_inputs=tokenized_inputs,
                        special_tokens_list=[OBJ_START_ID, OBJ_END_ID],
                    )
                    loss_components = criterion(outputs, targets)

                nonfinite_terms = {
                    k: not torch.isfinite(v).item()
                    for k, v in loss_components.items()
                }
                if any(nonfinite_terms.values()):
                    console.print(
                        "[red]Non-finite loss component detected:[/red] "
                        + ", ".join(f"{k}={loss_components[k].item():.4f}" for k in loss_components)
                    )
                    wandb.log({f"nonfinite_{k}": float(flag) for k, flag in nonfinite_terms.items()})
                    optimizer.zero_grad()
                    progress.update(train_task, advance=1, loss="Loss: NaN")
                    continue

                weighted_components = [
                    loss_components[k] * weight_dict.get(k, 1.0) for k in loss_components
                ]
                loss = torch.stack(weighted_components).sum() if weighted_components else torch.tensor(0.0, device=device)
                
                # Normalize loss for gradient accumulation
                if loss is not None:
                    loss = loss / args.grad_accumulation_steps

                # Ensure loss is a valid tensor with requires_grad
                if loss is not None and torch.is_tensor(loss) and loss.requires_grad:
                    # --- AMP: Backward pass with scaler ---
                    scaler.scale(loss).backward()

                    # --- Gradient Accumulation: Step optimizer only after N steps ---
                    if (i + 1) % args.grad_accumulation_steps == 0:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Scheduler step (per optimizer update) and zero gradients
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        
                    # Log detailed batch metrics to wandb
                    current_lr = optimizer.param_groups[0]['lr']
                    batch_log = {
                        'train_batch_loss': loss.item() * args.grad_accumulation_steps,
                        'learning_rate': current_lr,
                        'optimizer_step': lr_scheduler.last_epoch
                    }
                    for k, v in loss_components.items():
                        batch_log[f'train_{k}'] = v.detach().item()
                    wandb.log(batch_log)

                else:
                    # This happens when a batch has no valid samples for loss calculation
                    print("No valid samples in batch, skipping backpropagation.")
                    wandb.log({'train_batch_skipped': 1})
                
                current_loss_item = loss.item() * args.grad_accumulation_steps if torch.is_tensor(loss) else 0.0
                train_loss_sum += current_loss_item
                current_lr_display = optimizer.param_groups[0]['lr']
                progress.update(train_task, advance=1, loss=f"Loss: {current_loss_item:.4f} | LR: {current_lr_display:.6f}")

            # Compute average training loss for the epoch
            avg_train_loss = train_loss_sum / len(train_loader)
            
            # ---------------------
            #   VALIDATION
            # ---------------------
            progress.update(val_task, description=f"[cyan]Epoch {epoch+1}/{args.epochs} Validation...", visible=True)
            model.eval()
            val_loss_sum = 0
            all_val_metrics = {
                'iou': [], 'precision': [], 'recall': [], 'f1': [],
                'num_matched_pairs': [], 'avg_matched_score': [], 'avg_matched_iou': [],
                'num_passed_score': [], 'num_passed_iou': []
            }

            with torch.no_grad():
                for images, captions, targets, tokenized_inputs, special_tokens_list in val_loader:
                    images_gpu = [img.to(device) for img in images]
                    
                    batch_size_val = len(images_gpu)
                    if batch_size_val == 0:
                        continue
                        
                    image_sizes = torch.tensor([[img.shape[-2], img.shape[-1]] for img in images_gpu], device=device)

                    nested_images = nested_tensor_from_tensor_list(images_gpu)
                    
                    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
                    
                    outputs = model(
                        nested_images,
                        captions=captions,
                        tokenized_inputs=tokenized_inputs,
                        special_tokens_list=[OBJ_START_ID, OBJ_END_ID],
                    )
                    
                    # Compute and log validation metrics
                    # Use fixed IoU threshold (0.5) and box score threshold from args.box_threshold
                    val_metrics = compute_validation_metrics(
                        outputs,
                        targets,
                        matcher=matcher,
                        iou_threshold=0.5,
                        score_threshold=args.box_threshold,
                    )
                    if val_metrics:
                        for k in all_val_metrics.keys():
                            if k in val_metrics:
                                all_val_metrics[k].append(val_metrics[k])

                    # Note: Loss calculation for validation is optional but good for monitoring
                    # We will focus on IoU as the primary metric for improvement
                    progress.update(val_task, advance=1)
            
            progress.stop_task(val_task)

            # --- Epoch Summary ---
            avg_train_loss = train_loss_sum / max(1, len(train_loader))
            avg_val_iou = np.mean(all_val_metrics['iou']) if all_val_metrics['iou'] else 0
            avg_val_precision = np.mean(all_val_metrics['precision']) if all_val_metrics['precision'] else 0
            avg_val_recall = np.mean(all_val_metrics['recall']) if all_val_metrics['recall'] else 0
            avg_val_f1 = np.mean(all_val_metrics['f1']) if all_val_metrics['f1'] else 0
            
            # Diagnostic metrics
            total_matched_pairs = int(np.sum(all_val_metrics['num_matched_pairs'])) if all_val_metrics['num_matched_pairs'] else 0
            avg_matched_score = np.mean(all_val_metrics['avg_matched_score']) if all_val_metrics['avg_matched_score'] else 0.0
            avg_matched_iou = np.mean(all_val_metrics['avg_matched_iou']) if all_val_metrics['avg_matched_iou'] else 0.0
            total_passed_score = int(np.sum(all_val_metrics['num_passed_score'])) if all_val_metrics['num_passed_score'] else 0
            total_passed_iou = int(np.sum(all_val_metrics['num_passed_iou'])) if all_val_metrics['num_passed_iou'] else 0
            
            current_lr = optimizer.param_groups[0]['lr']
            
            summary_table = Table(title=f"Epoch {epoch+1}/{args.epochs} Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")
            summary_table.add_row("Average Training Loss", f"{avg_train_loss:.4f}")
            summary_table.add_row("Validation IoU", f"{avg_val_iou:.4f}")
            summary_table.add_row("Validation Precision", f"{avg_val_precision:.4f}")
            summary_table.add_row("Validation Recall", f"{avg_val_recall:.4f}")
            summary_table.add_row("Validation F1 Score", f"{avg_val_f1:.4f}")
            summary_table.add_row("", "")  # Separator
            summary_table.add_row("[yellow]Diagnostics:[/yellow]", "")
            summary_table.add_row("Matched Pairs (Hungarian)", f"{total_matched_pairs}")
            summary_table.add_row("Avg Matched Score", f"{avg_matched_score:.4f}")
            summary_table.add_row("Avg Matched IoU", f"{avg_matched_iou:.4f}")
            summary_table.add_row(f"Passed Score >{args.box_threshold}", f"{total_passed_score}")
            summary_table.add_row("Passed IoU >=0.5", f"{total_passed_iou}")
            summary_table.add_row("", "")  # Separator
            summary_table.add_row("Current Learning Rate", f"{current_lr:.6f}")
            console.print(summary_table)
            
            # Warning if classification scores are too low
            if avg_matched_score > 0 and avg_matched_score < 0.1:
                console.print(f"[yellow]⚠️  Warning: Classification scores are very low (avg: {avg_matched_score:.4f}). Consider increasing loss_ce weight or training longer.[/yellow]")
            if avg_matched_iou > 0 and avg_matched_iou < 0.3:
                console.print(f"[yellow]⚠️  Warning: Box regression needs improvement (avg IoU: {avg_matched_iou:.4f}). Consider increasing loss_bbox/giou weights.[/yellow]")

            # --- W&B Logging ---
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "val_iou": avg_val_iou,
                "val_precision": avg_val_precision,
                "val_recall": avg_val_recall,
                "val_f1": avg_val_f1,
                "learning_rate": current_lr,
                # Diagnostic metrics
                "val_matched_pairs": total_matched_pairs,
                "val_avg_matched_score": avg_matched_score,
                "val_avg_matched_iou": avg_matched_iou,
                "val_passed_score": total_passed_score,
                "val_passed_iou": total_passed_iou
            })

            # --- Save Best Model (Early Stopping) ---
            if avg_val_iou > best_val_iou:
                # Before saving a new best model, remove the old one if it exists
                if best_val_iou > -1.0:
                    old_save_path = os.path.join(args.output_dir, f"best_model_iou_{best_val_iou:.4f}.pth")
                    if os.path.exists(old_save_path):
                        os.remove(old_save_path)

                best_val_iou = avg_val_iou
                save_path = os.path.join(args.output_dir, f"best_model_iou_{best_val_iou:.4f}.pth")
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Save model, handling DataParallel if used
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), save_path)
                console.print(f"[bold green]🚀 New best model saved to {os.path.basename(save_path)} with validation IoU: {best_val_iou:.4f}[/bold green]")
        
        progress.stop()

    print("--- Training Complete ---")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GroundingDINO Finetuning Script")
    
    # Paths
    parser.add_argument("--data_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG-split", help="Path to the split dataset directory")
    parser.add_argument("--ann_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG/kavsir_bboxes.json", help="Path to the bounding box annotations file")
    parser.add_argument("--config_file", type=str, default=f"{DINO_PATH}/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to GroundingDINO model config")
    parser.add_argument("--pretrained_model_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/groundingdino_swint_ogc.pth", help="Path to pre-trained GroundingDINO weights")
    parser.add_argument("--output_dir", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/finetuned_model", help="Directory to save best model checkpoint")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=400)
    # Slightly lower default LR for more stable fine-tuning
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate for the transformer heads.")
    parser.add_argument("--lr_backbone", type=float, default=1e-6, help="Learning rate for the backbone.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate after cosine decay.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=800, help="Number of warmup steps for the learning rate.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--box_threshold", type=float, default=0.05, help="Box threshold for validation metrics (lowered from 0.3 to see early training progress)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for classification targets.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization weight for logits and box coords.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument("--wandb_project", type=str, default="Finetuning-GroundingDINO")
    parser.add_argument("--sanity_check", action="store_true", help="Run sanity checks on dataset and dataloader.")
    parser.add_argument("--disable_amp", action="store_true", help="Disable Automatic Mixed Precision.")
    args = parser.parse_args()
    main(args)