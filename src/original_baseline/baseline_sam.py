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

sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything")
sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def generate_masks(image_path):
    SAM_CHECKPOINT_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks = mask_generator.generate(image)
    
    binary_masks = [mask["segmentation"] for mask in masks]
    scores = [mask["predicted_iou"] for mask in masks]
    
    return image, binary_masks, scores

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

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
        
        # Check if we have segmentation
        if 'segmentation' in ann and ann['segmentation']:
            print(f"  - Processing segmentation for category {cat_id}")
            
            if isinstance(ann['segmentation'], list):
                print(f"    - Polygon segmentation with {len(ann['segmentation'])} parts")
                for seg in ann['segmentation']:
                    if isinstance(seg, list):
                        points = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt_masks[cat_id], [points], 1)
            elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                print(f"    - RLE segmentation")
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                
                # Check if mask dimensions match image dimensions
                if mask.shape != image_shape[:2]:
                    print(f"    - WARNING: Mask shape {mask.shape} doesn't match image shape {image_shape[:2]}")
                    # Resize mask to match image dimensions
                    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                
                gt_masks[cat_id] = np.logical_or(gt_masks[cat_id], mask.astype(bool))
            else:
                print(f"    - Unknown segmentation format: {type(ann['segmentation'])}")
        # If no segmentation, use bounding box
        elif 'bbox' in ann:
            print(f"  - Using bbox for category {cat_id} (no segmentation found)")
            x, y, w, h = [int(c) for c in ann['bbox']]
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            w = min(w, image_shape[1] - x)
            h = min(h, image_shape[0] - y)
            # Create mask from bounding box
            gt_masks[cat_id][y:y+h, x:x+w] = True
        else:
            print(f"  - No segmentation or bbox found for annotation {ann['id']}")
    
    # Check if masks are empty
    for cat_id, mask in gt_masks.items():
        if mask.sum() == 0:
            print(f"  - WARNING: Mask for category {cat_id} is empty!")
        else:
            print(f"  - Mask for category {cat_id} has {mask.sum()} positive pixels")
    
    return gt_masks

def run_baseline_test():
    test_dir = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/hat_dataset/test"
    output_dir = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/simple_baseline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotation_file = os.path.join(test_dir, "_annotations.coco.json")
    print(f"Loading annotations from: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations['categories'])} categories, {len(annotations['images'])} images, {len(annotations['annotations'])} annotations")
    
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    print(f"Categories: {categories}")
    
    image_id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
    file_to_image_id = {img['file_name']: img['id'] for img in annotations['images']}
    
    # Use exactly 150 samples for testing
    sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/utils")
    from common_test_set import get_common_test_set
    sample_images = get_common_test_set(test_dir, num_samples=150)  # Fixed test size
    
    print(f"Using {len(sample_images)} test images")
    # Don't print all 500 image names to keep output clean
    print(f"First few test images: {sample_images[:5]}...")
    
    results = []
    
    for img_name in sample_images:
        img_path = os.path.join(test_dir, img_name)
        img_id = file_to_image_id.get(img_name)
        
        if img_id is None:
            print(f"Warning: No annotations found for {img_name}, skipping...")
            continue
        
        print(f"Processing {img_name} (ID: {img_id})")
        
        image, masks, scores = generate_masks(img_path)
        print(f"Generated {len(masks)} masks")
        
        gt_masks = load_ground_truth_masks(annotation_file, img_id, image.shape)
        print(f"Found ground truth masks for categories: {list(gt_masks.keys())}")
        
        for cat_id, cat_name in categories.items():
            if cat_id not in gt_masks:
                print(f"  - No ground truth for category {cat_name} (ID: {cat_id})")
                continue
                
            gt_mask = gt_masks[cat_id]
            print(f"  - Processing category {cat_name} (ID: {cat_id})")
            
            best_iou = 0
            best_mask = None
            best_idx = -1
            for i, mask in enumerate(masks):
                iou = calculate_iou(mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask
                    best_idx = i
            
            print(f"  - Best IoU for {cat_name}: {best_iou:.4f}")
            
            # Save debug visualization regardless of IoU
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            gt_colored = np.zeros_like(image)
            gt_colored[:, :, 1] = gt_mask * 255  # Green for ground truth
            plt.imshow(gt_colored, alpha=0.5)
            plt.title(f"Ground Truth: {cat_name}")
            plt.axis('off')
            
            # All SAM masks overlaid
            plt.subplot(1, 3, 3)
            plt.imshow(image)
            
            # Show all masks with random colors
            for i, mask in enumerate(masks[:20]):  # Show first 20 masks
                color = np.random.rand(3)
                colored_mask = np.zeros_like(image, dtype=float)
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c]
                plt.imshow(colored_mask, alpha=0.2)
            
            plt.title(f"SAM Masks (showing 20/{len(masks)})")
            plt.axis('off')
            
            debug_path = os.path.join(debug_dir, f"{os.path.splitext(img_name)[0]}_{cat_name}_debug.jpg")
            plt.tight_layout()
            plt.savefig(debug_path)
            plt.close()
            
            # Use a very low threshold to accept masks
            if best_iou < 0.01:
                print(f"  - No good mask found for {cat_name} (IoU too low)")
                continue
                
            # If we get here, we have a mask with at least some overlap
            best_mask = masks[best_idx]
                
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            colored_mask = np.zeros_like(image)
            colored_mask[:, :, 0] = best_mask * 255
            plt.imshow(colored_mask, alpha=0.3)
            plt.title(f"Best Mask for {cat_name}\nIoU: {best_iou:.2f}")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(image)
            gt_colored = np.zeros_like(image)
            gt_colored[:, :, 1] = gt_mask * 255
            plt.imshow(gt_colored, alpha=0.3)
            plt.title(f"Ground Truth: {cat_name}")
            plt.axis('off')
            
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_{cat_name}.jpg")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            # Free GPU memory after processing each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results.append({
                'image_name': img_name,
                'image_id': img_id,
                'category_id': cat_id,
                'category_name': cat_name,
                'iou': best_iou
            })
    
    avg_iou = sum(r['iou'] for r in results) / len(results) if results else 0
    
    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
        json.dump({'avg_iou': avg_iou}, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Overall metrics: IoU={avg_iou:.3f}")

if __name__ == "__main__":
    run_baseline_test()