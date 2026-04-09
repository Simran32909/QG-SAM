import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Merge GQA Scene Graph bounding boxes into questions JSON.")
    parser.add_argument("--questions", required=True, help="Path to input questions JSON (e.g. train_balanced_questions.json)")
    parser.add_argument("--scenes", required=True, help="Path to scene graphs JSON (e.g. train_sceneGraphs.json)")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    print(f"Loading scene graphs from {args.scenes} (this might take a minute)...")
    with open(args.scenes, 'r') as f:
        scene_graphs = json.load(f)

    print(f"Loading questions from {args.questions}...")
    with open(args.questions, 'r') as f:
        questions = json.load(f)

    print("Merging bounding boxes...")
    missing_scenes = 0
    missing_objects = 0
    
    for q_id, q_data in tqdm(questions.items()):
        image_id = str(q_data.get("imageId"))
        
        # We need to collect all unique object IDs mentioned in the annotations
        obj_ids = set()
        annotations = q_data.get("annotations", {})
        for ann_type, ann_dict in annotations.items():
            if isinstance(ann_dict, dict):
                for obj_id in ann_dict.values():
                    # Handle cases where multiple IDs are sometimes comma-separated strings
                    for sub_id in str(obj_id).split(','):
                        obj_ids.add(sub_id.strip())

        boxes = []
        if image_id in scene_graphs:
            scene = scene_graphs[image_id]
            width = float(scene.get("width", 1))
            height = float(scene.get("height", 1))
            objects = scene.get("objects", {})
            
            for obj_id in obj_ids:
                if obj_id in objects:
                    obj_info = objects[obj_id]
                    # Convert absolute (x, y, w, h) to normalized (x_min, y_min, x_max, y_max)
                    x1 = obj_info["x"] / width
                    y1 = obj_info["y"] / height
                    x2 = (obj_info["x"] + obj_info["w"]) / width
                    y2 = (obj_info["y"] + obj_info["h"]) / height
                    
                    # Clamp to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    
                    boxes.append([x1, y1, x2, y2])
                else:
                    missing_objects += 1
        else:
            missing_scenes += 1
            
        # Add the 'gt_boxes' key that gqa_dataset.py expects
        q_data["gt_boxes"] = boxes

    print(f"Finished merging. Missing scenes: {missing_scenes}, Missing objects in scenes: {missing_objects}")
    
    print(f"Saving merged data to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(questions, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
