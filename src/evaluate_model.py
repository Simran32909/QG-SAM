import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import json
import cv2

# Add GroundingDINO to path
DINO_PATH = Path("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/GroundingDINO")
sys.path.append(str(DINO_PATH))

# Add SAM to path
SAM_PATH = Path("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")
sys.path.append(str(SAM_PATH))

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
from groundingdino.util.utils import clean_state_dict
from groundingdino.datasets import transforms as T
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.misc import nested_tensor_from_tensor_list

from segment_anything import sam_model_registry, SamPredictor

# A collate function for the test dataloader to handle
# (image_name, image_tensor, original_image_np, gt_mask) tuples
def collate_fn(batch):
    """
    Batch a list of samples from KvasirTestDataset.

    Each sample is:
        (image_name: str,
         image_tensor: torch.Tensor [3, H, W],
         original_image: np.ndarray [H, W, 3],
         gt_mask: np.ndarray [H, W])

    We keep image tensors as a list (not stacked) because widths can differ after
    resizing; later we convert them into a NestedTensor via
    nested_tensor_from_tensor_list.
    """
    image_names, image_tensors, original_images, gt_masks = zip(*batch)
    return list(image_names), list(image_tensors), list(original_images), list(gt_masks)

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_segmentation_metrics(pred_mask, gt_mask):
    """Calculates IoU and Dice Score for segmentation masks."""
    if pred_mask.shape != gt_mask.shape:
        # Resize pred_mask to match gt_mask shape
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure binary masks
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    
    return iou, dice


def visualize_example(
    image_name,
    orig_img,
    gt_boxes,
    pred_boxes,
    final_mask,
    save_dir,
):
    """
    Save a visualization image with GT boxes, predicted boxes, and optional mask overlay.
    """
    os.makedirs(save_dir, exist_ok=True)

    vis_img = orig_img.copy()  # numpy array HxWx3, RGB

    # Convert RGB to BGR for OpenCV drawing
    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Draw GT boxes in green
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Optional: overlay SAM mask in blue
    if final_mask is not None:
        mask = (final_mask > 0).astype(np.uint8)
        colored_mask = np.zeros_like(vis_img_bgr)
        colored_mask[:, :, 0] = 255  # Blue channel
        alpha = 0.4
        vis_img_bgr = np.where(
            mask[:, :, None].astype(bool),
            (alpha * colored_mask + (1 - alpha) * vis_img_bgr).astype(vis_img_bgr.dtype),
            vis_img_bgr,
        )

    out_path = os.path.join(save_dir, f"{image_name}_vis.png")
    cv2.imwrite(out_path, vis_img_bgr)

class KvasirTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_name = image_path.stem
        
        # Also load the ground truth mask
        mask_path = self.mask_dir / f"{image_name}.png"
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image_tensor, _ = self.transform(image, None)
        else:
            image_tensor = ToTensor()(image)

        # The GT mask is not transformed, we need it in its original size
        gt_mask = np.array(Image.open(mask_path).convert("L"))

        return image_name, image_tensor, np.array(image), gt_mask

def load_model(model_config_path, model_checkpoint_path, device):
    """
    Load a GroundingDINO model for evaluation.

    Supports:
    - Official checkpoints with structure {"model": state_dict, ...}
    - Fine-tuned checkpoints saved as a raw state_dict from finetune_gd.py
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")

    # Official checkpoints from GroundingDINO repo
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = clean_state_dict(checkpoint["model"])
        model.load_state_dict(state_dict, strict=False)
    else:
        # Fine-tuned checkpoint saved as raw state_dict in finetune_gd.py
        # Mirror the special-token and embedding handling from the training script
        special_tokens = ['<obj>', '</obj>']
        num_added = model.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print(f"Added {num_added} additional special tokens for evaluation: {special_tokens}")

        # Resize text embeddings to match extended tokenizer vocabulary
        try:
            if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
                model.text_encoder.bert.embeddings.word_embeddings = torch.nn.Embedding(
                    len(model.tokenizer),
                    model.text_encoder.bert.embeddings.word_embeddings.embedding_dim,
                )
                print("Resized embeddings via text_encoder.bert for evaluation")
            elif hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'embeddings'):
                model.text_encoder.embeddings.word_embeddings = torch.nn.Embedding(
                    len(model.tokenizer),
                    model.text_encoder.embeddings.word_embeddings.embedding_dim,
                )
                print("Resized embeddings via text_encoder.embeddings for evaluation")
            elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                model.bert.embeddings.word_embeddings = torch.nn.Embedding(
                    len(model.tokenizer),
                    model.bert.embeddings.word_embeddings.embedding_dim,
                )
                print("Resized embeddings via bert.embeddings for evaluation")
            else:
                print("Warning: could not find known embeddings module to resize for evaluation.")
        except Exception as e:
            print(f"Error resizing embeddings during evaluation: {e}")

        # Load fine-tuned weights
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model

def load_sam_model(sam_checkpoint, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = load_model(args.model_config, args.model_checkpoint, device=device)
    model = model.to(device)
    
    # Load SAM
    sam_predictor = load_sam_model(args.sam_checkpoint, device=device)

    # Load annotations
    with open(args.ann_path, 'r') as f:
        gt_annotations = json.load(f)

    # Dataset and DataLoader
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_dataset = KvasirTestDataset(
        image_dir=args.test_data_path, 
        mask_dir=args.mask_data_path,
        transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Starting evaluation on the test set...")
    
    all_predictions = {}
    total_mask_iou = 0.0
    total_dice_score = 0.0
    images_with_boxes = 0

    # Detection-only sanity metrics
    det_total_iou = 0.0
    det_total_matches = 0
    det_total_gt = 0
    det_total_preds = 0

    vis_count = 0

    with torch.no_grad():
        for image_names, image_tensors, original_images, gt_masks in tqdm(test_dataloader, desc="Evaluating"):
            # image_tensors is a list of tensors with potentially different widths
            images_gpu = [img.to(device) for img in image_tensors]
            nested_images = nested_tensor_from_tensor_list(images_gpu)
            
            # Build captions with the same special tokens used during fine-tuning
            caption = args.text_prompt
            captions = [f"<obj> {caption} </obj>"] * len(image_names)

            # Tokenize captions and prepare special token IDs
            tokenized = model.tokenizer(
                captions,
                padding="longest",
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            obj_start_id = model.tokenizer.convert_tokens_to_ids("<obj>")
            obj_end_id = model.tokenizer.convert_tokens_to_ids("</obj>")

            outputs = model(
                nested_images,
                captions=captions,
                tokenized_inputs=tokenized,
                special_tokens_list=[obj_start_id, obj_end_id],
            )
            
            logits = outputs["pred_logits"].cpu().sigmoid()
            boxes = outputs["pred_boxes"].cpu()

            for i, (image_name, logit, box_pred, orig_img, gt_mask) in enumerate(zip(image_names, logits, boxes, original_images, gt_masks)):
                
                # --- Box filtering with fallback to "best box per image" ---
                # scores: max class score per query
                scores = logit.max(dim=1)[0]
                keep = scores > args.box_threshold

                if keep.sum() == 0:
                    # No box exceeds threshold: fall back to single best-scoring query
                    best_idx = torch.argmax(scores)
                    keep = torch.zeros_like(scores, dtype=torch.bool)
                    keep[best_idx] = True

                box_pred = box_pred[keep]
                scores = scores[keep]
                logit = logit[keep]
                
                # Convert from normalized cxcywh (relative to resized input) to
                # absolute xyxy in ORIGINAL image coordinates.
                box_pred_cxcywh = box_pred.clone()
                box_pred_cxcywh[:, 0] = box_pred[:, 0] - box_pred[:, 2] / 2
                box_pred_cxcywh[:, 1] = box_pred[:, 1] - box_pred[:, 3] / 2
                box_pred_cxcywh[:, 2] = box_pred[:, 0] + box_pred[:, 2] / 2
                box_pred_cxcywh[:, 3] = box_pred[:, 1] + box_pred[:, 3] / 2

                # Scale to ORIGINAL image size (not the resized tensor size)
                orig_h, orig_w = orig_img.shape[:2]
                box_pred_cxcywh[:, 0::2] *= orig_w   # x coordinates
                box_pred_cxcywh[:, 1::2] *= orig_h   # y coordinates

                box_pred_xyxy = box_pred_cxcywh.round().int().tolist()
                
                all_predictions[image_name] = {
                    "boxes": box_pred_xyxy,
                    "scores": scores.tolist()
                }

                # --- Detection-only sanity metrics (IoU, precision, recall, F1) ---
                ann = gt_annotations.get(image_name, None)
                if ann is not None:
                    bboxes_info = ann.get("bbox", [])
                    gt_boxes = [
                        [b['xmin'], b['ymin'], b['xmax'], b['ymax']]
                        for b in bboxes_info
                    ]
                else:
                    gt_boxes = []

                num_gt = len(gt_boxes)
                num_preds = len(box_pred_xyxy)
                det_total_gt += num_gt
                det_total_preds += num_preds

                if num_gt > 0 and num_preds > 0:
                    matches = 0
                    for gt_box in gt_boxes:
                        best_iou = 0.0
                        for pred_box in box_pred_xyxy:
                            iou = calculate_iou(gt_box, pred_box)
                            if iou > best_iou:
                                best_iou = iou
                        det_total_iou += best_iou
                        if best_iou > args.det_iou_threshold:
                            matches += 1
                    det_total_matches += matches

                # --- SAM Integration ---
                if len(box_pred_xyxy) > 0:
                    images_with_boxes += 1

                    sam_predictor.set_image(orig_img)
                    
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                        torch.tensor(box_pred_xyxy, device=sam_predictor.device),
                        orig_img.shape[:2],
                    )
                    
                    masks, scores, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    
                    # Select the best mask based on SAM's confidence score
                    best_mask_idx = torch.argmax(scores)
                    final_mask = masks[best_mask_idx, 0].cpu().numpy()
                    
                    # Calculate segmentation metrics
                    mask_iou, dice_score = calculate_segmentation_metrics(final_mask, gt_mask)
                    total_mask_iou += mask_iou
                    total_dice_score += dice_score

                    # Visualization: save a few examples
                    if vis_count < args.vis_n:
                        visualize_example(
                            image_name=image_name,
                            orig_img=orig_img,
                            gt_boxes=gt_boxes,
                            pred_boxes=box_pred_xyxy,
                            final_mask=final_mask,
                            save_dir=args.vis_output_dir,
                        )
                        vis_count += 1

    # --- Final Metric Calculation ---
    num_images = len(test_dataset)
    avg_mask_iou = total_mask_iou / max(1, images_with_boxes)
    avg_dice_score = total_dice_score / max(1, images_with_boxes)

    # Detection metrics
    det_avg_iou = det_total_iou / max(1, det_total_gt)
    det_precision = det_total_matches / max(1, det_total_preds)
    det_recall = det_total_matches / max(1, det_total_gt)
    det_f1 = (
        2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)
        if det_total_preds > 0 and det_total_gt > 0
        else 0.0
    )
    
    print("\n--- Final Segmentation Metrics ---")
    print(f"Average Mask IoU (over images with at least one box): {avg_mask_iou:.4f}")
    print(f"Average Dice Score (over images with at least one box): {avg_dice_score:.4f}")
    print("----------------------------------")

    print("\n--- Detection-only Sanity Metrics (test set) ---")
    print(f"Detection IoU (avg over GT boxes): {det_avg_iou:.4f}")
    print(f"Detection Precision: {det_precision:.4f}")
    print(f"Detection Recall: {det_recall:.4f}")
    print(f"Detection F1: {det_f1:.4f}")
    print(f"Total GT boxes: {det_total_gt}")
    print(f"Total predicted boxes: {det_total_preds}")
    print(f"Matches (IoU > {args.det_iou_threshold:.2f}): {det_total_matches}")
    print("----------------------------------")
    
    print("Evaluation finished.")
    print(f"Total images in dataset: {num_images}")
    print(f"Images with at least one predicted box: {images_with_boxes}")
    print(f"Generated predictions (including empty) for {len(all_predictions)} images.")

    # --- Optional: save metrics to JSON for reproducibility ---
    if getattr(args, "metrics_output", None):
        metrics = {
            "segmentation": {
                "avg_mask_iou": float(avg_mask_iou),
                "avg_dice": float(avg_dice_score),
                "images_with_boxes": int(images_with_boxes),
                "num_images": int(num_images),
            },
            "detection": {
                "avg_iou": float(det_avg_iou),
                "precision": float(det_precision),
                "recall": float(det_recall),
                "f1": float(det_f1),
                "total_gt": int(det_total_gt),
                "total_pred": int(det_total_preds),
                "matches": int(det_total_matches),
                "iou_threshold": float(args.det_iou_threshold),
                "box_threshold": float(args.box_threshold),
            },
            "config": {
                "model_config": args.model_config,
                "model_checkpoint": args.model_checkpoint,
                "sam_checkpoint": args.sam_checkpoint,
                "text_prompt": args.text_prompt,
                "test_data_path": args.test_data_path,
                "mask_data_path": args.mask_data_path,
                "ann_path": args.ann_path,
                "batch_size": int(args.batch_size),
            },
        }
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics JSON to: {args.metrics_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GroundingDINO Finetuned Model Evaluation", add_help=True)
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test images directory")
    parser.add_argument("--mask_data_path", type=str, required=True, help="Path to the test masks directory")
    parser.add_argument("--ann_path", type=str, required=True, help="Path to the ground truth bounding box annotations (JSON file)")
    parser.add_argument("--text_prompt", type=str, default="polyp", help="Text prompt for the model")
    parser.add_argument("--box_threshold", type=float, default=0.5, help="Box score threshold for filtering detections (used before fallback-best box)")
    parser.add_argument("--det_iou_threshold", type=float, default=0.5, help="IoU threshold for detection sanity metrics")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--vis_output_dir", type=str, default="eval_vis", help="Directory to save visualization images")
    parser.add_argument("--vis_n", type=int, default=5, help="Number of visualization examples to save")
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="",
        help="Optional path to save metrics JSON (for reproducible baselines)",
    )
    
    args = parser.parse_args()
    main(args)