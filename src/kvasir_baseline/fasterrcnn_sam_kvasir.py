import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F

# --- Add SAM to Python path (reuse same setup as other baselines) ---
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")
from segment_anything import sam_model_registry, SamPredictor


class KvasirFRCNNDataset(Dataset):
    """
    Kvasir-SEG dataset for Faster R-CNN training/eval.

    Expects a split directory structure:
        base_dir/
          train/
            images/*.jpg
            masks/*.png  (not required for detection training)
          val/
            images/*.jpg
          test/
            images/*.jpg

    And a single JSON annotation file with structure:
        {
          "<image_id>": {
             "width": int,
             "height": int,
             "bbox": [
                {"xmin": int, "ymin": int, "xmax": int, "ymax": int},
                ...
             ]
          },
          ...
        }
    """

    def __init__(self, image_dir: Path, ann_path: Path, split: str):
        self.image_dir = Path(image_dir)
        self.split = split
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))

        with open(ann_path, "r") as f:
            self.annotations = json.load(f)

        # Filter to images that have entries in the annotation file
        self.image_ids = [p.stem for p in self.image_paths if p.stem in self.annotations]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_dir / f"{image_id}.jpg"

        # Load RGB image
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img)  # [3, H, W], float32 in [0,1]

        ann = self.annotations[image_id]
        bboxes_info = ann.get("bbox", [])

        boxes = []
        labels = []

        for bbox in bboxes_info:
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # single foreground class: polyp

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target, image_id


def detection_collate_fn(batch):
    """
    Standard collate function for torchvision detection models.

    Input: list of (image_tensor, target_dict, image_id)
    Output: list[image_tensor], list[target_dict], list[image_id]
    """
    images, targets, image_ids = zip(*batch)
    return list(images), list(targets), list(image_ids)


def get_fasterrcnn_model(num_classes: int = 2, pretrained: bool = True, device: str = "cuda"):
    """
    Build a Faster R-CNN model with a ResNet-50 FPN backbone.

    num_classes includes background (0) + foreground classes.
    """
    if hasattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn"):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )
    else:
        # Fallback for older torchvision versions
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head to match our number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.to(device)
    return model


def load_sam_predictor(sam_checkpoint: str, device: str = "cuda") -> SamPredictor:
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def calculate_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute IoU and Dice for binary masks.
    """
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    iou = intersection / (union + 1e-6)
    dice = (2.0 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-6)
    return iou, dice


def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Simple detection metrics: IoU (avg over GT boxes), precision, recall, F1.

    pred_boxes, gt_boxes: list of [x1, y1, x2, y2], in pixel coordinates.
    """

    def box_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union = areaA + areaB - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    if num_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, num_gt, num_pred, 0

    total_iou = 0.0
    matches = 0

    for gt in gt_boxes:
        best_iou = 0.0
        for pb in pred_boxes:
            iou = box_iou(gt, pb)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou
        if best_iou >= iou_threshold:
            matches += 1

    avg_iou = total_iou / max(1, num_gt)
    precision = matches / max(1, num_pred)
    recall = matches / max(1, num_gt)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return avg_iou, precision, recall, f1, num_gt, num_pred, matches


def run_fasterrcnn_sam_eval(
    base_dir: str,
    ann_path: str,
    sam_checkpoint: str,
    output_dir: str,
    score_threshold: float = 0.5,
    det_iou_threshold: float = 0.5,
    batch_size: int = 2,
):
    """
    Evaluation pipeline:
      1. Run Faster R-CNN on test images.
      2. Use high-score boxes to prompt SAM.
      3. Compute segmentation IoU/Dice and detection metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_dir = Path(base_dir)
    ann_path = Path(ann_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = base_dir / "test" / "images"
    test_mask_dir = base_dir / "test" / "masks"

    # --- Datasets & DataLoader ---
    test_dataset = KvasirFRCNNDataset(
        image_dir=test_img_dir,
        ann_path=ann_path,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )

    print(f"Test images: {len(test_dataset)}")

    # --- Models ---
    model = get_fasterrcnn_model(num_classes=2, pretrained=True, device=device)
    model.eval()

    sam_predictor = load_sam_predictor(str(sam_checkpoint), device=device)

    # --- Metrics accumulators ---
    total_mask_iou = 0.0
    total_dice = 0.0
    images_with_masks = 0

    det_total_iou = 0.0
    det_total_gt = 0
    det_total_pred = 0
    det_total_matches = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Evaluating Faster R-CNN + SAM on test set...", total=len(test_loader))

        with torch.no_grad():
            for images, targets, image_ids in test_loader:
                images = [img.to(device) for img in images]

                outputs = model(images)

                for img_tensor, target, out, image_id in zip(images, targets, outputs, image_ids):
                    # Convert image to numpy for SAM & mask loading
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)

                    # Get GT boxes from annotation dict
                    gt_boxes = target["boxes"].cpu().tolist()

                    # Detection metrics
                    pred_boxes_all = out["boxes"].detach().cpu().tolist()
                    pred_scores_all = out["scores"].detach().cpu().tolist()

                    # Filter by score threshold
                    pred_boxes = [
                        box for box, score in zip(pred_boxes_all, pred_scores_all) if score >= score_threshold
                    ]

                    avg_iou, prec, rec, f1, num_gt, num_pred, matches = calculate_detection_metrics(
                        pred_boxes, gt_boxes, iou_threshold=det_iou_threshold
                    )

                    det_total_iou += avg_iou * max(1, num_gt)  # accumulate sum over GTs
                    det_total_gt += num_gt
                    det_total_pred += num_pred
                    det_total_matches += matches

                    # Segmentation metrics via SAM (if we have any boxes)
                    if pred_boxes:
                        # Load GT mask
                        mask_path = test_mask_dir / f"{image_id}.png"
                        if not mask_path.exists():
                            mask_path = test_mask_dir / f"{image_id}.jpg"
                        if not mask_path.exists():
                            continue

                        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is None:
                            continue

                        sam_predictor.set_image(img_np)
                        boxes_tensor = torch.tensor(pred_boxes, device=sam_predictor.device, dtype=torch.float32)
                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                            boxes_tensor, img_np.shape[:2]
                        )

                        masks, sam_scores, _ = sam_predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,
                        )

                        best_idx = torch.argmax(sam_scores)
                        best_mask = masks[best_idx, 0].cpu().numpy()

                        seg_iou, seg_dice = calculate_segmentation_metrics(best_mask, gt_mask)
                        total_mask_iou += seg_iou
                        total_dice += seg_dice
                        images_with_masks += 1

                progress.advance(task)

    # --- Final metrics ---
    avg_mask_iou = total_mask_iou / max(1, images_with_masks)
    avg_dice = total_dice / max(1, images_with_masks)

    det_avg_iou = det_total_iou / max(1, det_total_gt)
    det_precision = det_total_matches / max(1, det_total_pred)
    det_recall = det_total_matches / max(1, det_total_gt)
    if det_precision + det_recall > 0:
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall)
    else:
        det_f1 = 0.0

    print("\n--- Faster R-CNN + SAM: Final Segmentation Metrics ---")
    print(f"Average Mask IoU (over images with at least one mask): {avg_mask_iou:.4f}")
    print(f"Average Dice Score (over images with at least one mask): {avg_dice:.4f}")
    print("----------------------------------")

    print("\n--- Faster R-CNN: Detection-only Metrics (test set) ---")
    print(f"Detection IoU (avg over GT boxes): {det_avg_iou:.4f}")
    print(f"Detection Precision: {det_precision:.4f}")
    print(f"Detection Recall: {det_recall:.4f}")
    print(f"Detection F1: {det_f1:.4f}")
    print(f"Total GT boxes: {det_total_gt}")
    print(f"Total predicted boxes: {det_total_pred}")
    print(f"Matches (IoU > {det_iou_threshold:.2f}): {det_total_matches}")
    print("----------------------------------")

    # Optionally, save metrics to JSON for later plotting
    metrics = {
        "segmentation": {
            "avg_mask_iou": float(avg_mask_iou),
            "avg_dice": float(avg_dice),
            "images_with_masks": images_with_masks,
        },
        "detection": {
            "avg_iou": float(det_avg_iou),
            "precision": float(det_precision),
            "recall": float(det_recall),
            "f1": float(det_f1),
            "total_gt": int(det_total_gt),
            "total_pred": int(det_total_pred),
            "matches": int(det_total_matches),
            "iou_threshold": float(det_iou_threshold),
            "score_threshold": float(score_threshold),
        },
    }

    metrics_path = output_dir / "fasterrcnn_sam_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Faster R-CNN (pretrained) + SAM baseline on Kvasir-SEG."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG-split",
        help="Path to the split Kvasir-SEG dataset directory.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG/kavsir_bboxes.json",
        help="Path to the bounding box annotations JSON.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth",
        help="Path to SAM checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/fasterrcnn_sam_kvasir",
        help="Directory to save metrics and optional outputs.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for Faster R-CNN detections.",
    )
    parser.add_argument(
        "--det_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for counting a detection as correct.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for evaluation.",
    )

    args = parser.parse_args()

    run_fasterrcnn_sam_eval(
        base_dir=args.base_dir,
        ann_path=args.ann_path,
        sam_checkpoint=args.sam_checkpoint,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        det_iou_threshold=args.det_iou_threshold,
        batch_size=args.batch_size,
    )


