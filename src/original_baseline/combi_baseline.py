import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import torchvision.transforms as T
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

# --- Add helper: convert cxcywh (normalized) -> xyxy pixel coords
def cxcywh_to_xyxy_pixel(boxes, image_shape):
    """
    boxes: torch.Tensor shape (N,4) in normalized (cx, cy, w, h) [0..1]
    image_shape: (H, W, C) or tuple (H, W)
    returns: numpy array (N,4) in pixel coordinates [x1, y1, x2, y2]
    """
    if isinstance(image_shape, (tuple, list)) and len(image_shape) >= 2:
        H = image_shape[0]
        W = image_shape[1]
    else:
        raise ValueError("image_shape must be (H,W,...)")
    if boxes.numel() == 0:
        return np.zeros((0,4), dtype=np.float32)
    # boxes: cx, cy, w, h (normalized)
    cxcy = boxes[:, :2]
    wh = boxes[:, 2:4]
    cx = cxcy[:, 0] * W
    cy = cxcy[:, 1] * H
    w = wh[:, 0] * W
    h = wh[:, 1] * H
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy = torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()
    # clip
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W-1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H-1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W-1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H-1)
    return xyxy

sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything")
sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/GroundingDINO")
sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/segment_anything")

from segment_anything import sam_model_registry, SamPredictor

from groundingdino.util.inference import load_model as dino_load_model, load_image as dino_load_image
from groundingdino.util import box_ops
# from groundingdino.util.vl_utils import preprocess_caption # This import is failing


# Debug Step: Re-add prompt mapping for robust detection
PROMPT_MAPPING = {
    "helmet": ["hard hat", "construction helmet", "safety helmet", "helmet"],
    "head": ["human head", "a person's head", "head"],
    "person": ["person", "human", "construction worker", "a worker"]
}

def preprocess_caption(caption: str) -> str:
    """
    Simple helper function to clean up text prompts for the model.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    return caption

def load_models(groundingdino_config_path, groundingdino_checkpoint_path, sam_checkpoint_path, device):
    # Debug Step 2: Confirm model weights are actually loaded
    print("\n--- Loading Models ---")
    print(f"GroundingDINO Config: {groundingdino_config_path}")
    print(f"GroundingDINO Checkpoint: {groundingdino_checkpoint_path}")
    groundingdino_model = dino_load_model(groundingdino_config_path, groundingdino_checkpoint_path)
    print("GroundingDINO Model Loaded Successfully.")
    # print(groundingdino_model) # Optional: uncomment for full model summary

    print(f"SAM Checkpoint: {sam_checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam)
    print("SAM Model Loaded Successfully.")
    print("----------------------\n")
    return groundingdino_model, sam_predictor

def grounding_outputs(model, image, caption, box_threshold, text_threshold, device="cuda"):
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = image.to(device)
    
    # Manually handle text encoding, which was causing the import error
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption, padding="longest", return_tensors="pt")
    
    with torch.no_grad():
        # The model expects a batched tensor. Add a batch dimension with image[None].
        outputs = model(image[None], captions=[caption])

    # Get raw logits and boxes from the output
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # Shape: (num_queries, num_classes)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # Shape: (num_queries, 4)

    # --- Start of New Debugging Block ---
    # Find the max score for each proposal
    max_scores_per_proposal = prediction_logits.max(dim=1)[0]
    
    # Find the absolute max score across all proposals
    absolute_max_score = max_scores_per_proposal.max().item()
    
    print(f"  - DEBUG: Absolute max confidence score found: {absolute_max_score:.4f}")
    # --- End of New Debugging Block ---

    # Filter based on box_threshold
    mask = max_scores_per_proposal > box_threshold
    logits = prediction_logits[mask]  # Filtered logits
    boxes = prediction_boxes[mask]   # Filtered boxes

    # Don't proceed if no boxes passed the threshold
    if logits.shape[0] == 0:
        return torch.empty(0, 4), torch.empty(0), []

    # From here, we use the logic from inference.py to get phrases
    tokenized_caption = tokenized # Use the tensorized token object
    phrases = []
    for logit_row in logits:
        # For each box, find the phrase that corresponds to the highest logit score
        phrase = get_phrases_from_posmap(
            logit_row > text_threshold, tokenized_caption, tokenizer
        ).replace('.', '')
        phrases.append(phrase)

    # Get the final scores for the detected boxes
    final_scores = logits.max(dim=1)[0]
    
    return boxes, final_scores, phrases


def get_phrases_from_posmap(posmap, tokenized, tokenizer):
    """
    Safer version that handles empty posmaps and unusual shapes.
    posmap: torch.BoolTensor or similar.
    tokenized: tokenizer(...) output where tokenized.input_ids exists
    tokenizer: huggingface tokenizer with decode()
    """
    try:
        # Defensive checks
        if posmap is None:
            return ""
        # Ensure posmap is a torch.Tensor
        if not isinstance(posmap, torch.Tensor):
            posmap = torch.tensor(posmap, dtype=torch.bool)
        if posmap.numel() == 0:
            return ""
        # Ensure 2D: (num_tokens, something)
        if posmap.dim() == 1:
            # single-row case: treat as one token-row
            posmap = posmap.unsqueeze(0)

        phrases = []
        # iterate rows (if rows represent different maps)
        for row in posmap:
            if row.numel() == 0:
                continue
            ids = (row > 0).nonzero(as_tuple=False).squeeze(1)
            if ids.numel() == 0:
                continue
            # merge contiguous ids into spans
            spans = []
            ids_list = ids.tolist()
            start = ids_list[0]
            end = ids_list[0]
            for idx in ids_list[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    spans.append((start, end))
                    start = idx
                    end = idx
            spans.append((start, end))
            for (s, e) in spans:
                token_ids = tokenized.input_ids.squeeze(0)[s:e+1].tolist()
                phrase = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if phrase:
                    phrases.append(phrase)
        # dedupe and join
        phrases = list(dict.fromkeys(phrases))
        return " . ".join(phrases)
    except Exception as e:
        print(f"  - WARNING: Caught Exception in get_phrases_from_posmap: {e}. Returning empty phrase.")
        return ""


def process_image(image_path, cat_name, groundingdino_model, sam_predictor, box_threshold, text_threshold):
    print(f"\nProcessing image: {image_path} for category: '{cat_name}'")

    # Debug Step 1: Confirm the image actually contains the object and path is correct
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        return None
    try:
        # dino_load_image also implicitly checks if the image is readable
        image, image_tensor = dino_load_image(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image, it might be corrupted: {image_path}")
            return None
        print(f"Successfully loaded image. Shape: {image.shape}")
    except Exception as e:
        print(f"[ERROR] Exception while loading image {image_path}: {e}")
        return None

    # Use prompt mapping for robust detection
    prompts_to_try = PROMPT_MAPPING.get(cat_name, [cat_name])
    boxes, scores, phrases = [], [], []

    for prompt in prompts_to_try:
        print(f"  Trying prompt: '{prompt}'")
        # Lowered thresholds for better recall during debugging
        temp_boxes, temp_scores, temp_phrases = grounding_outputs(
            model=groundingdino_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        if len(temp_boxes) > 0:
            print(f"  SUCCESS: Found {len(temp_boxes)} boxes with prompt: '{prompt}'")
            boxes, scores, phrases = temp_boxes, temp_scores, temp_phrases
            break # Stop on first successful prompt

    if len(boxes) == 0:
        print(f"  FAILURE: No boxes found for category '{cat_name}' with any prompt.")
        # Return an empty result instead of None, so the main loop can still visualize the failure
        return {
            'image': image,
            'boxes': np.array([]),
            'masks': [],
            'phrases': [],
            'scores': torch.tensor([])
        }

    H, W, _ = image.shape
    # boxes is the tensor from groundingdino (normalized cxcywh)
    # Convert to pixel xyxy for visualization and for SAM transform
    boxes_pixel_xyxy = cxcywh_to_xyxy_pixel(boxes, image.shape)

    # For SAM, create a torch tensor in image coordinates (xyxy) on the sam device:
    boxes_for_sam = torch.as_tensor(boxes_pixel_xyxy, device=sam_predictor.device, dtype=torch.float32)

    sam_predictor.set_image(image)
    # Now apply transform (SAM expects boxes in xyxy image coordinates)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_for_sam, image.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    print(f"Generated {len(masks)} masks from {len(boxes)} boxes.")

    return {
        'image': image,
        'boxes': boxes_pixel_xyxy,
        'masks': masks.cpu().numpy(),
        'phrases': phrases,
        'scores': scores
    }

def load_ground_truth_masks(annotation_file, image_id, image_shape):

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    image_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    print(f"Found {len(image_anns)} annotations for image ID {image_id}")
    
    gt_masks = {}
    for ann in image_anns:
        cat_id = ann['category_id']
        if cat_id not in gt_masks:
            gt_masks[cat_id] = np.zeros(image_shape[:2], dtype=bool)
        
        # If no segmentation -> use BBOX
        if 'bbox' in ann:
            print(f"  - Using bbox for category {cat_id}")

            x, y, w, h = [int(c) for c in ann['bbox']]
            x = max(0, min(x, image_shape[1] - 1))

            y = max(0, min(y, image_shape[0] - 1))
            w = min(w, image_shape[1] - x)

            h = min(h, image_shape[0] - y)
            gt_masks[cat_id][y:y+h, x:x+w] = True
    
    for cat_id, mask in gt_masks.items():
        if mask.sum() == 0:
            print(f"  - WARNING: Mask for category {cat_id} is empty!")
        else:
            print(f"  - Mask for category {cat_id} has {mask.sum()} positive pixels")
    
    return gt_masks

def visualize_results(image, boxes, masks, gt_mask, output_path):
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r', linewidth=2)
    plt.title("GroundingDINO Boxes")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(image)
    for mask in masks:
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask * 255  # Blue for predicted mask
        plt.imshow(colored_mask, alpha=0.5)
    plt.title("SAM Masks")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    gt_colored = np.zeros_like(image)
    gt_colored[:, :, 1] = gt_mask * 255  # Green for ground truth
    plt.imshow(gt_colored, alpha=0.5)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_comb_baseline(args):
    test_dir = "/scratch/tathagata.ghosh/qgsam/hat_dataset/test"
    output_dir = "/scratch/tathagata.ghosh/qgsam/results/true_baseline"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Models...")
    groundingdino_model, sam_predictor = load_models(
        args.groundingdino_config_path,
        args.groundingdino_checkpoint_path,
        args.sam_checkpoint_path,
        args.device
    )

    sys.path.append("/scratch/tathagata.ghosh/qgsam/utils")
    from common_test_set import get_common_test_set
    test_images = get_common_test_set(args.test_dir, num_samples=150)

    # Debug Step 3: Run inference on a known-good test image
    print("\n--- Running Sanity Check on Known Image ---")
    if not test_images:
        print("Warning: No test images found, skipping sanity check.")
    else:
        known_image_path = os.path.join(args.test_dir, test_images[0]) # Use the first test image
        sanity_check_result = process_image(
            known_image_path, 
            "helmet", 
            groundingdino_model, 
            sam_predictor,
            args.box_threshold,
            args.text_threshold
        )
        if sanity_check_result and len(sanity_check_result['boxes']) > 0:
            print("Sanity Check PASSED: Model found objects in a known image.")
        else:
            print("Sanity Check FAILED: Model could not find objects. Check model loading and paths.")
    print("-------------------------------------------\n")


    annotation_file = os.path.join(test_dir, "_annotations.coco.json")
    print(f"Loading annotations from: {annotation_file}")

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    file_to_image_id = {img['file_name']: img['id'] for img in annotations['images']}
    
    print(f"Will process {len(test_images)} test images")
    results = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:

        image_task = progress.add_task(
            "[cyan]Processing images...", 
            total=len(test_images)
        )
        
        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            img_id = file_to_image_id.get(img_name)

            if img_id is None:
                progress.print(f"[yellow]Warning: No annotations found for {img_name}, skipping...")
                progress.advance(image_task)
                continue
            
            progress.print(f"[blue]Processing {img_name} (ID: {img_id})")

            # Get GT Masks
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_masks = load_ground_truth_masks(
                annotation_file,
                img_id,
                image.shape
            )
            
            # Nested task for categories
            category_task = progress.add_task(
                f"[green]Categories for {img_name}...",
                total=len(categories)
            )
            
            for cat_id, cat_name in categories.items():
                if cat_id not in gt_masks:
                    progress.print(f"[yellow]  - No ground truth for {cat_name} (ID: {cat_id})")
                    progress.advance(category_task)
                    continue
                    
                progress.print(f"[blue]  - Processing category {cat_name} (ID: {cat_id})")
                gt_mask = gt_masks[cat_id]
        
                outputs = process_image(
                    img_path,
                    cat_name,
                    groundingdino_model,
                    sam_predictor,
                    args.box_threshold,
                    args.text_threshold
                )
                
                if outputs is None:
                    progress.print(f"[red]Failed to process image {img_name} for category {cat_name}")
                    progress.advance(category_task)
                    continue
                
                best_iou = 0
                if len(outputs['masks']) > 0:
                    for mask in outputs['masks']:
                        iou = calculate_iou(mask, gt_mask)
                        best_iou = max(best_iou, iou)
                        progress.print(f"[blue]    Mask IoU: {iou:.4f}")
                else:
                    progress.print(f"[yellow]No masks generated for {cat_name}")
                
                progress.print(f"[green]  - Best IoU for {cat_name}: {best_iou:.4f}")

                vis_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_{cat_name}.jpg")
                visualize_results(outputs['image'], outputs['boxes'], outputs['masks'], gt_mask, vis_path)
                
                # Record results
                results.append({
                    'image_name': img_name,
                    'category_id': cat_id,
                    'category_name': cat_name,
                    'iou': best_iou,
                    'num_boxes': len(outputs['boxes']),
                    'num_masks': len(outputs['masks']),
                    'box_scores': outputs['scores'].tolist() if len(outputs['scores']) > 0 else []
                })

                # Advance category progress
                progress.advance(category_task)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Advance image progress
            progress.advance(image_task)
    
        avg_iou = sum(r['iou'] for r in results) / len(results) if results else 0
    
    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics = {
        'avg_iou': avg_iou,
        'num_samples': len(results),
        'categories': list(categories.values())
    }
    
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Overall metrics: IoU={avg_iou:.3f}")

def calculate_iou(pred_mask, gt_mask):

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
        
    return intersection / union

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("End-to-end GroundingDINO + SAM baseline")
    parser.add_argument(
        "--test_dir", 
        type=str, 
        default="/scratch/tathagata.ghosh/qgsam/hat_dataset/test",
        help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/tathagata.ghosh/qgsam/results/true_baseline",
        help="Path to save results and visualizations"
    )
    parser.add_argument(
        "--groundingdino_config_path", 
        type=str, 
        default="/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to GroundingDINO config file"
    )
    parser.add_argument(
        "--groundingdino_checkpoint_path",
        type=str,
        default="/scratch/tathagata.ghosh/qgsam/weights/groundingdino_swint_ogc.pth",
        help="Path to GroundingDINO checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        type=str,
        default="/scratch/tathagata.ghosh/qgsam/weights/sam_vit_h_4b8939.pth",
        help="Path to SAM checkpoint file"
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.05,
        help="Box confidence threshold for GroundingDINO"
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.05,
        help="Text confidence threshold for GroundingDINO"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on"
    )
    args = parser.parse_args()
    run_comb_baseline(args)

