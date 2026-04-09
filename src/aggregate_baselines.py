import json
from pathlib import Path


def main():
    root = Path("/ssd_scratch/jyothi.swaroopa/Simran/qgsam")

    # Paths to per-method metrics
    oracle_path = root / "results/oracle_sam_kvasir/overall_metrics.json"
    auto_path = root / "results/auto_sam_kvasir/overall_metrics.json"
    zero_shot_path = root / "results/true_groundedsam_kvasir/overall_metrics.json"
    gdino_path = root / "results/finetuned_gdino_sam_metrics.json"
    frcnn_path = root / "results/fasterrcnn_sam_kvasir/fasterrcnn_sam_metrics.json"

    with oracle_path.open() as f:
        oracle_metrics = json.load(f)
    with auto_path.open() as f:
        auto_metrics = json.load(f)
    with zero_shot_path.open() as f:
        zero_shot_metrics = json.load(f)
    with gdino_path.open() as f:
        gdino_metrics = json.load(f)
    with frcnn_path.open() as f:
        frcnn_metrics = json.load(f)

    # Build a unified baselines dict
    baselines = {
        "oracle_sam": {
            "name": "Oracle-Guided SAM",
            "type": "oracle_sam",
            "segmentation": {
                "avg_mask_iou": float(oracle_metrics["average_iou"]),
                "avg_dice": None,
            },
        },
        "auto_sam": {
            "name": "Auto SAM",
            "type": "auto_sam",
            "segmentation": {
                "avg_mask_iou": float(auto_metrics["average_iou"]),
                "avg_dice": None,
            },
        },
        "zero_shot_groundedsam": {
            "name": "Zero-shot Grounded-SAM",
            "type": "zero_shot_groundedsam",
            "segmentation": {
                "avg_mask_iou": float(zero_shot_metrics["average_iou"]),
                "avg_dice": None,
            },
        },
        "finetuned_gdino_sam": {
            "name": "Fine-tuned GroundingDINO + SAM",
            "type": "finetuned_gdino_sam",
            "segmentation": gdino_metrics["segmentation"],
            "detection": gdino_metrics["detection"],
            "config": gdino_metrics.get("config", {}),
        },
        "fasterrcnn_sam": {
            "name": "Faster R-CNN + SAM",
            "type": "fasterrcnn_sam",
            "segmentation": frcnn_metrics["segmentation"],
            "detection": frcnn_metrics["detection"],
        },
    }

    out_path = root / "results" / "results_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(baselines, f, indent=4)

    print(f"Wrote combined baselines to: {out_path}")


if __name__ == "__main__":
    main()


