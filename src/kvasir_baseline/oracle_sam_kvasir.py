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
from segment_anything import sam_model_registry, SamPredictor

def load_oracle_boxes(annotation_file):
    """
    Loads bounding box annotations from the Kvasir-SEG JSON file.
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # The format is {image_id: {"bbox": [{"xmin":..., "ymin":..., "xmax":..., "ymax":...}]}}
    # We just need to return the whole dictionary.
    return annotations

def get_sam_predictor(sam_checkpoint_path, device="cuda"):
    """
    Loads and returns the SAM predictor model.
    """
    print("--- Loading SAM Model ---")
    print(f"SAM Checkpoint: {sam_checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM Model Loaded Successfully.")
    print("-------------------------\n")
    return predictor

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

def run_oracle_sam_test(test_img_dir, test_mask_dir, annotation_file, output_dir):
    """
    Runs the Oracle-Guided SAM test on the Kvasir-SEG test set.
    """
    # --- Configuration ---
    SAM_CHECKPOINT_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    predictor = get_sam_predictor(SAM_CHECKPOINT_PATH, DEVICE)
    oracle_boxes_data = load_oracle_boxes(annotation_file)
    
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                gt_mask_path = Path(test_mask_dir) / (image_id + ".png") # Assuming masks are PNG
                if not gt_mask_path.exists():
                     gt_mask_path = Path(test_mask_dir) / (image_id + ".jpg") # Fallback to JPG
                
                gt_mask_full = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                # Binarize the mask to be safe
                _, gt_mask_full = cv2.threshold(gt_mask_full, 127, 255, cv2.THRESH_BINARY)
                gt_mask_full = gt_mask_full.astype(bool)

            except Exception as e:
                progress.console.print(f"[red]Error loading data for {image_id}: {e}[/red]")
                progress.advance(task)
                continue
            
            # --- Get Oracle Boxes ---
            if image_id not in oracle_boxes_data:
                progress.console.print(f"[yellow]Warning: No annotations found for {image_id}. Skipping.[/yellow]")
                progress.advance(task)
                continue

            bboxes_info = oracle_boxes_data[image_id].get("bbox", [])
            if not bboxes_info:
                progress.console.print(f"[yellow]Warning: No bounding boxes for {image_id}. Skipping.[/yellow]")
                progress.advance(task)
                continue
            
            # Convert boxes to xyxy format for SAM
            input_boxes = np.array([[b['xmin'], b['ymin'], b['xmax'], b['ymax']] for b in bboxes_info])
            
            # --- Run SAM Prediction ---
            predictor.set_image(image)
            
            transformed_boxes = predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            transformed_boxes_torch = torch.as_tensor(transformed_boxes, device=predictor.device)

            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes_torch,
                multimask_output=False,
            )
            
            # Combine all predicted masks for the image into one mask
            if masks.shape[0] > 0:
                combined_pred_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
            else:
                combined_pred_mask = np.zeros_like(gt_mask_full, dtype=bool)

            # --- Calculate IoU ---
            iou = calculate_iou(combined_pred_mask, gt_mask_full)

            results.append({
                'image_id': image_id,
                'iou': iou,
                'num_boxes': len(input_boxes)
            })

            # --- Visualize Results ---
            plt.figure(figsize=(15, 5))
            
            # Original Image + Oracle Boxes
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            for box in input_boxes:
                x0, y0, x1, y1 = box
                plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor=(0,0,0,0), lw=2))
            plt.title(f"Original + Oracle Boxes ({len(input_boxes)})")
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            plt.imshow(combined_pred_mask, cmap='viridis', alpha=0.6)
            plt.title(f"SAM Predicted Mask\nIoU: {iou:.4f}")
            plt.axis('off')

            # Ground Truth Mask
            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.imshow(gt_mask_full, cmap='hot', alpha=0.6)
            plt.title("Ground Truth Mask")
            plt.axis('off')

            vis_path = os.path.join(output_dir, f"{image_id}_oracle_result.jpg")
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close()

            progress.advance(task)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Save Results ---
    if results:
        avg_iou = sum(r['iou'] for r in results) / len(results)
    else:
        avg_iou = 0

    print(f"\n[bold green]Oracle-Guided SAM Test Complete![/bold green]")
    print(f"Overall Average IoU: {avg_iou:.4f}")

    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump({"average_iou": avg_iou}, f, indent=4)
        
    print(f"Results and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Oracle-Guided SAM test on Kvasir-SEG dataset.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG-split",
        help="Path to the split Kvasir-SEG dataset directory."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/dataset/Kvasir-SEG/kavsir_bboxes.json",
        help="Path to the ground truth bounding box JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/oracle_sam_kvasir",
        help="Directory to save results and visualizations."
    )
    args = parser.parse_args()

    test_img_dir = os.path.join(args.base_dir, "test", "images")
    test_mask_dir = os.path.join(args.base_dir, "test", "masks") # Points to the binarized PNG masks

    run_oracle_sam_test(test_img_dir, test_mask_dir, args.annotation_file, args.output_dir)
