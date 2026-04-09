# QG-SAM: Question-Guided Segmentation and Reasoning

A framework for grounded Visual Question Answering (VQA) that forces the model to commit to visual evidence (segmentation masks) before reasoning over an answer.

## Core Idea
Standard VQA models are often "black boxes" that guess answers via global attention. QG-SAM implements an **evidence-first bottleneck**:
1. **Segmenter:** A cross-attention module that generates $K$ masks based on the question.
2. **Evidence Pooler:** Extracts features only from the masked regions.
3. **Reasoner:** A Transformer encoder that predicts the answer based solely on the extracted evidence tokens.

## Setup
```bash
pip install -r requirements.txt
# Merge GQA scene graph boxes for training supervision
python -m qgsam_core.data.prep.merge_boxes --questions [PATH] --scenes [PATH] --output gqa_data/train_balanced_with_boxes.json
```

## Training
```bash
python -m qgsam_core.scripts.train \
  --questions ./gqa_data/train_balanced_with_boxes.json \
  --images ./gqa_data/images \
  --wandb_run_name "qgsam-v1"
```

## Inference Demo
Run the Streamlit app to visualize heatmaps and reasoning steps:
```bash
streamlit run qgsam_core/scripts/demo_app.py
```

## Performance (GQA Balanced)
| Model | Accuracy | Grounding (mIoU) |
| :--- | :--- | :--- |
| Baseline (CLIP-BERT) | 49.8% | N/A |
| **QG-SAM** | **57.0%** | **0.41** |

## Structure
* `qgsam_core/`: Model architecture, training loops, and data pipelines.
* `src/`: Evaluation scripts and legacy baseline comparisons.
