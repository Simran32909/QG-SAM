import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json

try:
    from pycocotools import mask as mask_utils
except ImportError:
    print("Warning: pycocotools not found. RLE mask decoding will not work.")
    class DummyMaskUtils:
        @staticmethod
        def decode(rle):
            return np.zeros((100, 100), dtype=bool)
    mask_utils = DummyMaskUtils()

sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything")
sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/GroundingDINO")
sys.path.append("/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/segment_anything")

from groundingdino.util.inference import load_image, load_model, predict
from segment_anything import sam_model_registry, SamPredictor

def run_grounded_sam(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    """Run GroundedSAM on a single image with a text prompt"""
    # Load models
    GROUNDING_DINO_CONFIG_PATH = "/scratch/tathagata.ghosh/qgsam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "/scratch/tathagata.ghosh/qgsam/weights/groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT_PATH = "/scratch/tathagata.ghosh/qgsam/weights/sam_vit_h_4b8939.pth"
    
    # Load GroundingDINO model
    groundingdino_model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # Load image
    image_source, image = load_image(image_path)
    
    # GroundingDINO prediction
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # Convert to numpy
    h, w, _ = image_source.shape
    boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
    
    # SAM prediction
    sam_predictor.set_image(image_source)
    masks = []
    for box in boxes_xyxy:
        mask, _, _ = sam_predictor.predict(
            box=box.cpu().numpy(),
            multimask_output=False
        )
        masks.append(mask[0])
    
    return image_source, boxes_xyxy.cpu().numpy(), masks, phrases

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union between two binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def load_ground_truth_masks(annotation_file, image_id, image_shape):
    """Load ground truth masks from COCO annotations"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    image_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    print(f"Found {len(image_anns)} annotations for image ID {image_id}")
    
    gt_masks = {}
    for ann in image_anns:
        cat_id = ann['category_id']
        if cat_id not in gt_masks:
            gt_masks[cat_id] = np.zeros(image_shape[:2], dtype=bool)
        
        # If no segmentation, use bounding box
        if 'bbox' in ann:
            print(f"  - Using bbox for category {cat_id}")
            x, y, w, h = [int(c) for c in ann['bbox']]
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            w = min(w, image_shape[1] - x)
            h = min(h, image_shape[0] - y)
            # Create mask from bounding box
            gt_masks[cat_id][y:y+h, x:x+w] = True
    
    # Check if masks are empty
    for cat_id, mask in gt_masks.items():
        if mask.sum() == 0:
            print(f"  - WARNING: Mask for category {cat_id} is empty!")
        else:
            print(f"  - Mask for category {cat_id} has {mask.sum()} positive pixels")
    
    return gt_masks

def run_groundedsam_test():
    """Run GroundedSAM test with evaluation metrics"""
    test_dir = "/scratch/tathagata.ghosh/qgsam/hat_dataset/test"
    output_dir = "/scratch/tathagata.ghosh/qgsam/results/groundedsam_baseline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load test annotations
    annotation_file = os.path.join(test_dir, "_annotations.coco.json")
    print(f"Loading annotations from: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations['categories'])} categories, {len(annotations['images'])} images, {len(annotations['annotations'])} annotations")
    
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    print(f"Categories: {categories}")
    
    image_id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
    file_to_image_id = {img['file_name']: img['id'] for img in annotations['images']}
    
    # Get a few sample images
    image_files = os.listdir(test_dir)
    image_files = [f for f in image_files if f.endswith('.jpg')]
    sample_images = image_files[:5]  # Process 5 images for testing
    
    print(f"Found {len(image_files)} images in {test_dir}")
    print(f"Sample images: {sample_images}")
    
    results = []
    
    for img_name in sample_images:
        img_path = os.path.join(test_dir, img_name)
        img_id = file_to_image_id.get(img_name)
        
        if img_id is None:
            print(f"Warning: No annotations found for {img_name}, skipping...")
            continue
        
        print(f"Processing {img_name} (ID: {img_id})")
        
        # Load ground truth masks
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_masks = load_ground_truth_masks(annotation_file, img_id, image.shape)
        
        # Run GroundedSAM for each category
        for cat_id, cat_name in categories.items():
            if cat_id not in gt_masks:
                print(f"  - No ground truth for category {cat_name} (ID: {cat_id})")
                continue
                
            gt_mask = gt_masks[cat_id]
            print(f"  - Processing category {cat_name} (ID: {cat_id})")
            
            # Run GroundedSAM with the category name as text prompt
            try:
                image, boxes, masks, phrases = run_grounded_sam(img_path, cat_name)
                
                if not masks:
                    print(f"  - No masks found for {cat_name}")
                    continue
                
                # Calculate IoU for each mask
                best_iou = 0
                best_mask_idx = -1
                for i, mask in enumerate(masks):
                    iou = calculate_iou(mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask_idx = i
                
                print(f"  - Best IoU for {cat_name}: {best_iou:.4f}")
                
                # Visualize results
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                # GroundedSAM result
                plt.subplot(1, 3, 2)
                plt.imshow(image)
                
                # Draw bounding box
                if best_mask_idx >= 0 and best_mask_idx < len(boxes):
                    box = boxes[best_mask_idx]
                    x0, y0, x1, y1 = box
                    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r', linewidth=2)
                
                # Draw mask
                if best_mask_idx >= 0 and best_mask_idx < len(masks):
                    mask = masks[best_mask_idx]
                    colored_mask = np.zeros_like(image)
                    colored_mask[:, :, 0] = mask * 255
                    plt.imshow(colored_mask, alpha=0.3)
                
                plt.title(f"GroundedSAM: {cat_name}\nIoU: {best_iou:.2f}")
                plt.axis('off')
                
                # Ground truth mask
                plt.subplot(1, 3, 3)
                plt.imshow(image)
                gt_colored = np.zeros_like(image)
                gt_colored[:, :, 1] = gt_mask * 255  # Green for ground truth
                plt.imshow(gt_colored, alpha=0.3)
                plt.title(f"Ground Truth: {cat_name}")
                plt.axis('off')
                
                # Save visualization
                output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_{cat_name}.jpg")
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                # Save results
                results.append({
                    'image_name': img_name,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'category_name': cat_name,
                    'iou': best_iou,
                    'num_boxes': len(boxes),
                    'num_masks': len(masks)
                })
            except Exception as e:
                print(f"  - Error processing {cat_name}: {e}")
    
    # Calculate overall metrics
    avg_iou = sum(r['iou'] for r in results) / len(results) if results else 0
    
    # Save detailed results
    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save overall metrics
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump({'avg_iou': avg_iou}, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Overall metrics: IoU={avg_iou:.3f}")

if __name__ == "__main__":
    run_groundedsam_test()
