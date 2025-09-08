import os
import json
import random
import shutil
from pathlib import Path

def split_dataset():
    with open('/scratch/tathagata.ghosh/qgsam/hat_dataset/train/_annotations.coco.json', 'r') as f:
        coco_data = json.load(f)
    
    image_ids = [
        image['id'] for image in coco_data['images']
    ]
    random.shuffle(image_ids)
    train_size=int(len(image_ids)*0.8)
    val_size=int(len(image_ids)*0.1)

    train_ids = set(image_ids[:train_size])
    val_ids = set(image_ids[train_size:train_size+val_size])
    test_ids = set(image_ids[train_size+val_size:])

    train_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    val_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    test_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}

    for img in coco_data['images']:

        if img['id'] in train_ids:
            train_data['images'].append(img)
        elif img['id'] in val_ids:
            val_data['images'].append(img)
        else:
            test_data['images'].append(img)
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in train_ids:
            train_data['annotations'].append(ann)
        elif img_id in val_ids:
            val_data['annotations'].append(ann)
        else:
            test_data['annotations'].append(ann)
    
    os.makedirs('/scratch/tathagata.ghosh/qgsam/data/hard_hat/train', exist_ok=True)
    os.makedirs('/scratch/tathagata.ghosh/qgsam/data/hard_hat/val', exist_ok=True)
    os.makedirs('/scratch/tathagata.ghosh/qgsam/data/hard_hat/test', exist_ok=True)

    with open('/scratch/tathagata.ghosh/qgsam/data/hard_hat/train/_annotations.coco.json', 'w') as f:
        json.dump(train_data, f)
    with open('/scratch/tathagata.ghosh/qgsam/data/hard_hat/val/_annotations.coco.json', 'w') as f:
        json.dump(val_data, f)
    with open('/scratch/tathagata.ghosh/qgsam/data/hard_hat/test/_annotations.coco.json', 'w') as f:
        json.dump(test_data, f)
    
    src_dir = Path('/scratch/tathagata.ghosh/qgsam/hat_dataset/train')
    for img in train_data['images']:
        shutil.copy(src_dir / img['file_name'], f'/scratch/tathagata.ghosh/qgsam/data/hard_hat/train/{img["file_name"]}')
    for img in val_data['images']:
        shutil.copy(src_dir / img['file_name'], f'/scratch/tathagata.ghosh/qgsam/data/hard_hat/val/{img["file_name"]}')
    for img in test_data['images']:
        shutil.copy(src_dir / img['file_name'], f'/scratch/tathagata.ghosh/qgsam/data/hard_hat/test/{img["file_name"]}')
    
    print(f"Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"Test: {len(test_data['images'])} images, {len(test_data['annotations'])} annotations")

if __name__ == "__main__":
    split_dataset()