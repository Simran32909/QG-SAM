"""
Prepare 50 diverse test samples from the GQA validation set for demo inference.

Usage:
    python qgsam_core/scripts/prepare_test_samples.py

Output:
    test_samples/
        images/        <- copied GQA images
        samples.json   <- [{image_path, question, answer, question_type}, ...]
"""

import json
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VAL_JSON    = "gqa_data/val_balanced_questions_with_boxes.json"
IMAGES_DIR  = Path("gqa_data/images")
OUT_DIR     = Path("test_samples")
N_SAMPLES   = 200
SEED        = 42

# Answer types to sample from (keeps demo diverse and clean)
# Each bucket: (label, set of valid answers, max samples from this bucket)
BUCKETS = [
    ("yes/no",     {"yes", "no"},                                        55),
    ("color",      {"red", "blue", "green", "white", "black", "brown",
                    "yellow", "orange", "gray", "grey", "pink", "purple"}, 40),
    ("existence",  {"1", "2", "3", "0"},                                  30),
    ("location",   {"left", "right", "top", "bottom", "middle",
                    "behind", "front", "above", "below"},                  35),
    ("object",     {"table", "chair", "car", "dog", "cat", "man", "woman",
                    "person", "tree", "sky", "grass", "road", "water",
                    "building", "shirt", "pants", "hat", "bag"},           40),
]

# ── Load questions ─────────────────────────────────────────────────────────────
print(f"Loading {VAL_JSON} ...")
with open(VAL_JSON) as f:
    data = json.load(f)

# Filter to only questions whose image exists on disk
available = {
    qid: q for qid, q in data.items()
    if (IMAGES_DIR / f"{q['imageId']}.jpg").is_file()
}
print(f"  Total val questions with images on disk: {len(available):,}")

# ── Sample from each bucket ───────────────────────────────────────────────────
random.seed(SEED)
selected = []
used_image_ids = set()

for label, valid_answers, quota in BUCKETS:
    bucket_pool = [
        (qid, q) for qid, q in available.items()
        if str(q.get("answer", "")).strip().lower() in valid_answers
        and q["imageId"] not in used_image_ids   # no duplicate images
    ]
    random.shuffle(bucket_pool)
    picked = bucket_pool[:quota]
    for qid, q in picked:
        selected.append({
            "question_id":  qid,
            "image_id":     q["imageId"],
            "question":     q["question"],
            "answer":       q["answer"],
            "question_type": label,
        })
        used_image_ids.add(q["imageId"])
    print(f"  Bucket '{label}': picked {len(picked)} / {quota} requested")

print(f"\nTotal selected: {len(selected)} samples")

# ── Copy images & save manifest ───────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
img_out = OUT_DIR / "images"
img_out.mkdir(exist_ok=True)

manifest = []
for item in selected:
    src = IMAGES_DIR / f"{item['image_id']}.jpg"
    dst = img_out / f"{item['image_id']}.jpg"
    shutil.copy2(src, dst)
    manifest.append({
        "question_id":   item["question_id"],
        "image_path":    str(dst),
        "question":      item["question"],
        "answer":        item["answer"],
        "question_type": item["question_type"],
    })

out_json = OUT_DIR / "samples.json"
with open(out_json, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ Done!")
print(f"   Images  → {img_out}/")
print(f"   Manifest→ {out_json}")
print(f"\nSample preview:")
for item in manifest[:5]:
    print(f"  [{item['question_type']:10s}]  Q: {item['question']!r:55s}  A: {item['answer']!r}")
