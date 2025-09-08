import os
import json
import random

def get_common_test_set(test_dir, num_samples=5, seed=42):
    """
    Get a common set of test images to ensure fair comparison across methods
    
    Args:
        test_dir: Directory containing test images
        num_samples: Number of test images to select
        seed: Random seed for reproducibility
        
    Returns:
        List of image filenames to use for testing
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image files
    image_files = os.listdir(test_dir)
    image_files = [f for f in image_files if f.endswith('.jpg')]
    
    # Load annotations to ensure we select images with annotations
    annotation_file = os.path.join(test_dir, "_annotations.coco.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create mapping from filename to image_id
    file_to_image_id = {img['file_name']: img['id'] for img in annotations['images']}
    
    # Get images that have annotations
    valid_images = []
    for img_name in image_files:
        img_id = file_to_image_id.get(img_name)
        if img_id is not None:
            # Check if this image has annotations
            has_annotations = any(ann['image_id'] == img_id for ann in annotations['annotations'])
            if has_annotations:
                valid_images.append(img_name)
    
    # Select a random subset
    if len(valid_images) > num_samples:
        test_images = random.sample(valid_images, num_samples)
    else:
        test_images = valid_images
    
    if len(test_images) <= 10:
        print(f"Selected {len(test_images)} test images: {test_images}")
    else:
        print(f"Selected {len(test_images)} test images. First few: {test_images[:5]}...")
    
    # Save the selected test set for reference
    with open(os.path.join(os.path.dirname(test_dir), "common_test_set.json"), 'w') as f:
        json.dump(test_images, f)
    
    return test_images
