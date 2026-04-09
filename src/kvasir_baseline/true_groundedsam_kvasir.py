import os
import sys
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn

# --- Add Grounded-SAM to Python path ---
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything")
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/GroundingDINO")
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")

from groundingdino.util.inference import load_model as dino_load_model, load_image as dino_load_image
from segment_anything import sam_model_registry, SamPredictor

# --- Helper Functions from previous script ---

def cxcywh_to_xyxy_pixel(boxes, image_shape):
    if boxes.numel() == 0:
        return np.empty((0, 4), dtype=np.float32)
    H, W = image_shape[:2]
    boxes_scaled = boxes * torch.tensor([W, H, W, H], device=boxes.device)
    cx, cy, w, h = boxes_scaled.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

def preprocess_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    return caption

def calculate_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1.0

# --- Prompt Variations for Robustness ---
PROMPT_MAPPING = {
    "polyp": ["polyp", "a polyp", "intestinal polyp", "bowel polyp", "lesion"]
}

# --- Core Pipeline Functions ---

def load_all_models(dino_config, dino_ckpt, sam_ckpt, device):
    print("--- Loading All Models ---")
    groundingdino_model = dino_load_model(dino_config, dino_ckpt)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("All models loaded successfully.")
    print("--------------------------\n")
    return groundingdino_model, sam_predictor

def get_dino_predictions(model, image_tensor, caption, box_thresh, text_thresh, device):
    caption = preprocess_caption(caption)
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    # Filter by threshold
    mask = logits.max(dim=1)[0] > box_thresh
    
    if mask.sum() == 0: # Check if any boxes passed the threshold
        # Find the single best box if none pass the threshold, to handle low-confidence cases
        max_score, max_idx = logits.max(dim=1)[0].max(dim=0)
        if max_score > 0.05: # Use a very low fallback threshold
             return boxes[max_idx:max_idx+1], logits[max_idx:max_idx+1].max(dim=1)[0]
        else:
             return torch.empty(0, 4), torch.empty(0)

    boxes_filt = boxes[mask]
    scores_filt = logits[mask].max(dim=1)[0]
    
    return boxes_filt, scores_filt


def run_dino_with_prompt_variations(model, image_tensor, category, box_thresh, text_thresh, device, progress):
    """Tries multiple prompts for a category until a detection is found."""
    prompts_to_try = PROMPT_MAPPING.get(category, [category])
    
    for prompt in prompts_to_try:
        progress.console.print(f"  > Trying prompt: '[bold cyan]{prompt}[/bold cyan]'")
        boxes, scores = get_dino_predictions(
            model, image_tensor, prompt, box_thresh, text_thresh, device
        )
        if boxes.shape[0] > 0:
            progress.console.print(f"  > [green]Success![/green] Found {boxes.shape[0]} box(es) with prompt.")
            return boxes, scores
            
    progress.console.print("  > [yellow]Warning:[/yellow] No boxes found with any prompt variant.")
    return torch.empty(0, 4), torch.empty(0)


def run_true_groundedsam_test(test_img_dir, test_mask_dir, output_dir, args):
    """
    Runs the true, end-to-end Grounded-SAM test on the Kvasir-SEG test set.
    """
    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    dino_model, sam_predictor = load_all_models(
        args.groundingdino_config_path,
        args.groundingdino_checkpoint_path,
        args.sam_checkpoint_path,
        args.device
    )
    
    test_image_paths = sorted(list(Path(test_img_dir).glob("*.jpg")))
    print(f"Found {len(test_image_paths)} images in the test set.")
    
    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("[cyan]Processing test images...", total=len(test_image_paths))

        for img_path in test_image_paths:
            image_id = img_path.stem
            progress.update(task, description=f"[cyan]Processing {image_id}...")

            # --- Load Data ---
            try:
                image_bgr = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                _, image_tensor = dino_load_image(str(img_path))

                gt_mask_path = Path(test_mask_dir) / (image_id + ".png")
                gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
                gt_mask = gt_mask.astype(bool)
            except Exception as e:
                progress.console.print(f"[red]Error loading data for {image_id}: {e}[/red]")
                progress.advance(task)
                continue
            
            # --- GroundingDINO Prediction ---
            boxes_filt, scores = run_dino_with_prompt_variations(
                dino_model, image_tensor, "polyp", args.box_threshold, args.text_threshold, args.device, progress
            )
            
            # --- SAM Prediction ---
            if boxes_filt.shape[0] > 0:
                boxes_pixel = cxcywh_to_xyxy_pixel(boxes_filt, image_rgb.shape)
                
                sam_predictor.set_image(image_rgb)
                transformed_boxes = sam_predictor.transform.apply_boxes(boxes_pixel, image_rgb.shape[:2])
                transformed_boxes = torch.as_tensor(transformed_boxes, device=sam_predictor.device)

                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=transformed_boxes, multimask_output=False
                )
                pred_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
            else:
                boxes_pixel = np.empty((0, 4))
                pred_mask = np.zeros_like(gt_mask, dtype=bool)

            # --- Evaluation & Visualization ---
            iou = calculate_iou(pred_mask, gt_mask)
            results.append({'image_id': image_id, 'iou': iou, 'boxes_found': boxes_filt.shape[0]})
            
            plt.figure(figsize=(20, 5))
            # ... (visualization code as before)
            plt.subplot(1, 4, 1); plt.imshow(image_rgb); plt.title("Original"); plt.axis('off')
            plt.subplot(1, 4, 2); plt.imshow(image_rgb)
            for box in boxes_pixel:
                x0, y0, x1, y1 = box
                plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, ec='r', fc='none', lw=2))
            plt.title(f"GroundingDINO ({boxes_pixel.shape[0]} boxes)")
            plt.axis('off')

            plt.subplot(1, 4, 3); plt.imshow(image_rgb); plt.imshow(pred_mask, cmap='viridis', alpha=0.6)
            plt.title(f"SAM Mask\nIoU: {iou:.4f}"); plt.axis('off')
            
            plt.subplot(1, 4, 4); plt.imshow(image_rgb); plt.imshow(gt_mask, cmap='hot', alpha=0.6)
            plt.title("Ground Truth"); plt.axis('off')

            vis_path = os.path.join(output_dir, f"{image_id}_true_baseline.jpg")
            plt.tight_layout(); plt.savefig(vis_path); plt.close()

            progress.advance(task)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Save Final Results ---
    avg_iou = sum(r['iou'] for r in results) / len(results) if results else 0
    print(f"\n[bold green]True End-to-End Grounded-SAM Test Complete![/bold green]")
    print(f"Overall Average IoU: {avg_iou:.4f}")

    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump({"average_iou": avg_iou}, f, indent=4)
    print(f"Results and visualizations saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run True End-to-End Grounded-SAM test.")
    # Paths
    parser.add_argument("--base_dir", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG-split")
    parser.add_argument("--output_dir", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/true_groundedsam_kvasir")
    parser.add_argument("--groundingdino_config_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino_checkpoint_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint_path", type=str, default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth")
    # Model params
    parser.add_argument("--box_threshold", type=float, default=0.15)
    parser.add_argument("--text_threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    test_img_dir = os.path.join(args.base_dir, "test", "images")
    test_mask_dir = os.path.join(args.base_dir, "test", "masks")
    
    run_true_groundedsam_test(test_img_dir, test_mask_dir, args.output_dir, args)
