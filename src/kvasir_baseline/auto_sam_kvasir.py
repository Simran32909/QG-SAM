import os
import sys
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn

# --- Add SAM to Python path ---
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def get_sam_mask_generator(sam_checkpoint_path, device="cuda"):
    """
    Loads and returns the SAM automatic mask generator.
    """
    print("--- Loading SAM Model for Automatic Mask Generation ---")
    print(f"SAM Checkpoint: {sam_checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    
    # SAM's automatic generator has many parameters; these are good defaults.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Don't find tiny specs
    )
    print("SAM Automatic Mask Generator Loaded Successfully.")
    print("-------------------------------------------------\n")
    return mask_generator

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union between two binary masks.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def run_auto_sam_test(test_img_dir, test_mask_dir, output_dir):
    """
    Runs the Automatic SAM test on the Kvasir-SEG test set.
    """
    # --- Configuration ---
    SAM_CHECKPOINT_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    mask_generator = get_sam_mask_generator(SAM_CHECKPOINT_PATH, DEVICE)
    
    test_image_paths = sorted(list(Path(test_img_dir).glob("*.jpg")))
    print(f"Found {len(test_image_paths)} images in the test set.")
    
    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing test images...", total=len(test_image_paths))

        for img_path in test_image_paths:
            image_id = img_path.stem
            progress.update(task, description=f"[cyan]Processing {image_id}...")

            # --- Load Image and Ground Truth Mask ---
            try:
                image = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                gt_mask_path = Path(test_mask_dir) / (image_id + ".png")
                if not gt_mask_path.exists():
                    gt_mask_path = Path(test_mask_dir) / (image_id + ".jpg")
                
                gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
                gt_mask = gt_mask.astype(bool)

            except Exception as e:
                progress.console.print(f"[red]Error loading data for {image_id}: {e}[/red]")
                progress.advance(task)
                continue
            
            # --- Generate all masks automatically ---
            generated_masks = mask_generator.generate(image_rgb)
            
            if not generated_masks:
                progress.console.print(f"[yellow]Warning: SAM generated 0 masks for {image_id}.[/yellow]")
                iou = 0
                best_mask = np.zeros_like(gt_mask, dtype=bool)
            else:
                # --- Find the best matching mask ---
                best_iou = 0
                best_mask = None
                for mask_data in generated_masks:
                    pred_mask = mask_data['segmentation']
                    iou = calculate_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = pred_mask
                
                iou = best_iou # Final IoU for this image

            results.append({'image_id': image_id, 'iou': iou, 'masks_generated': len(generated_masks)})

            # --- Visualize Results ---
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(image_rgb)
            if best_mask is not None:
                plt.imshow(best_mask, cmap='viridis', alpha=0.6)
            plt.title(f"Best SAM Mask (of {len(generated_masks)})\nIoU: {iou:.4f}")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(image_rgb)
            plt.imshow(gt_mask, cmap='hot', alpha=0.6)
            plt.title("Ground Truth Mask")
            plt.axis('off')

            vis_path = os.path.join(output_dir, f"{image_id}_auto_result.jpg")
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close()

            progress.advance(task)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Save Results ---
    avg_iou = sum(r['iou'] for r in results) / len(results) if results else 0

    print(f"\n[bold green]Automatic SAM Test Complete![/bold green]")
    print(f"Overall Average IoU: {avg_iou:.4f}")

    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump({"average_iou": avg_iou}, f, indent=4)
        
    print(f"Results and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Automatic SAM test on Kvasir-SEG dataset.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG-split",
        help="Path to the split Kvasir-SEG dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/auto_sam_kvasir",
        help="Directory to save results and visualizations."
    )
    args = parser.parse_args()

    test_img_dir = os.path.join(args.base_dir, "test", "images")
    test_mask_dir = os.path.join(args.base_dir, "test", "masks")

    run_auto_sam_test(test_img_dir, test_mask_dir, args.output_dir)
