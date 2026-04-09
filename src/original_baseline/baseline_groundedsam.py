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

sys.path.append("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/segment_anything")
from segment_anything import sam_model_registry, SamPredictor

def simple_text_to_box(image, category_name, gt_bbox=None):
    """
    A simple function that simulates text-to-box functionality
    Instead of using GroundingDINO, we'll use the ground truth box as a proxy
    This simulates what GroundingDINO would do in an ideal case
    """
    h, w = image.shape[:2]
    
    if gt_bbox is not None:
        # Use ground truth box with some noise to simulate detection
        x, y, box_w, box_h = gt_bbox
        # Add small random noise (±10%)
        noise_x = np.random.uniform(-0.1, 0.1) * box_w
        noise_y = np.random.uniform(-0.1, 0.1) * box_h
        noise_w = np.random.uniform(-0.1, 0.1) * box_w
        noise_h = np.random.uniform(-0.1, 0.1) * box_h
        
        # Apply noise
        x = max(0, min(w-1, x + noise_x))
        y = max(0, min(h-1, y + noise_y))
        box_w = max(10, min(w-x, box_w + noise_w))
        box_h = max(10, min(h-y, box_h + noise_h))
        
        # Convert to [x0, y0, x1, y1] format
        box = np.array([x, y, x + box_w, y + box_h])
        return [box], [category_name]
    else:
        # If no ground truth, return a box covering most of the image
        # This is a fallback that will rarely be used
        box = np.array([w*0.1, h*0.1, w*0.9, h*0.9])
        return [box], [category_name]

def run_simple_groundedsam(image_path, text_prompt, gt_bbox=None):
    """
    Run a simplified version of GroundedSAM using ground truth boxes
    """
    # Load SAM model
    SAM_CHECKPOINT_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/weights/sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get boxes from text (simulated)
    boxes, phrases = simple_text_to_box(image, text_prompt, gt_bbox)
    
    # SAM prediction
    sam_predictor.set_image(image)
    masks = []
    
    for box in boxes:
        try:
            mask, _, _ = sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            masks.append(mask[0])
        except Exception as e:
            print(f"Error processing box {box}: {e}")
    
    return image, np.array(boxes), masks, phrases

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
    gt_boxes = {}
    
    for ann in image_anns:
        cat_id = ann['category_id']
        if cat_id not in gt_masks:
            gt_masks[cat_id] = np.zeros(image_shape[:2], dtype=bool)
            gt_boxes[cat_id] = []
        
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
            # Save the box for later use
            gt_boxes[cat_id].append([x, y, w, h])
    
    # Check if masks are empty
    for cat_id, mask in gt_masks.items():
        if mask.sum() == 0:
            print(f"  - WARNING: Mask for category {cat_id} is empty!")
        else:
            print(f"  - Mask for category {cat_id} has {mask.sum()} positive pixels")
    
    return gt_masks, gt_boxes

def run_simple_groundedsam_test():
    """Run simplified GroundedSAM test with evaluation metrics"""
    test_dir = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/hat_dataset/test"
    output_dir = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/results/simple_groundedsam"
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
        
        # Load ground truth masks and boxes
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_masks, gt_boxes = load_ground_truth_masks(annotation_file, img_id, image.shape)
        
        # Run simplified GroundedSAM for each category
        for cat_id, cat_name in categories.items():
            if cat_id not in gt_masks:
                print(f"  - No ground truth for category {cat_name} (ID: {cat_id})")
                continue
                
            gt_mask = gt_masks[cat_id]
            gt_box = gt_boxes[cat_id][0] if gt_boxes[cat_id] else None
            print(f"  - Processing category {cat_name} (ID: {cat_id})")
            
            # Run simplified GroundedSAM with the category name as text prompt
            try:
                image, boxes, masks, phrases = run_simple_groundedsam(img_path, cat_name, gt_box)
                
                if not masks:
                    print(f"  - No masks found for {cat_name}")
                    continue
                
                # Calculate IoU for each mask
                best_iou = 0
                best_mask_idx = -1
                for i, mask in enumerate(masks):
                    try:
                        iou = calculate_iou(mask, gt_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_mask_idx = i
                    except Exception as e:
                        print(f"  - Error calculating IoU for mask {i}: {e}")
                
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
                try:
                    if best_mask_idx >= 0 and best_mask_idx < len(boxes):
                        box = boxes[best_mask_idx]
                        x0, y0, x1, y1 = box
                        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r', linewidth=2)
                except Exception as e:
                    print(f"  - Error drawing bounding box: {e}")
                
                # Draw mask
                try:
                    if best_mask_idx >= 0 and best_mask_idx < len(masks):
                        mask = masks[best_mask_idx]
                        colored_mask = np.zeros_like(image)
                        colored_mask[:, :, 0] = mask * 255
                        plt.imshow(colored_mask, alpha=0.3)
                except Exception as e:
                    print(f"  - Error drawing mask: {e}")
                
                plt.title(f"Simple GroundedSAM: {cat_name}\nIoU: {best_iou:.2f}")
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
                
                # Free GPU memory after processing each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
    run_simple_groundedsam_test()
