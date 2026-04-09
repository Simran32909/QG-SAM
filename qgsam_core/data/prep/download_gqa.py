"""
Download and prepare GQA dataset for QG-SAM.
Usage: python -m qgsam_core.data.prep.download_gqa --out_dir /path/to/gqa
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare GQA for QG-SAM")
    parser.add_argument("--out_dir", type=str, default="./gqa_data", help="Output directory")
    parser.add_argument("--questions_url", type=str, default="", help="URL to questions JSON (optional)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder: user should download GQA from https://cs.stanford.edu/people/dorarad/gqa/
    # and place questions.json, train_balanced_questions.json, etc. in out_dir
    readme = out_dir / "README.txt"
    readme.write_text(
        "GQA data for QG-SAM.\n"
        "Download from https://cs.stanford.edu/people/dorarad/gqa/\n"
        "Place questions JSON and images in this directory.\n"
    )
    print(f"Created {out_dir}. Add GQA questions JSON and images as per README.")


if __name__ == "__main__":
    main()
