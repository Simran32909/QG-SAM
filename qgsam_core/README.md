# QG-SAM Core (Phase 3)

Evidence-first VQA: question-aware segmentation + evidence-only reasoning.

## Layout

- `data/`: GQA and VQA-X datasets, collate, prep scripts
- `models/`: QuestionEncoder, QuestionAwareSegmenter, EvidencePool, EvidenceReasoner, QGSAM
- `losses/`: seg BCE+Dice, CLIP alignment (stub)
- `train/`: LightningModule
- `eval/`: metrics and runner
- `configs/default.yaml`: default config
- `scripts/train.py`, `scripts/eval.py`: entry points

## Quick run (from repo root `/ssd_scratch/jyothi.swaroopa/Simran/qgsam`)

1. Prepare GQA: put questions JSON and images in a dir (see `data/prep/download_gqa.py`).
2. Install logging (recommended): `pip install wandb` then `wandb login`.
3. Train (GPU by default if CUDA is available; validation every epoch; logs to W&B + TensorBoard if W&B missing):
   ```bash
   python -m qgsam_core.scripts.train \
     --questions /path/to/train_balanced_questions.json \
     --images /path/to/images \
     --max_epochs 30
   ```
   Val file defaults to `val_balanced_questions.json` next to the train file, or set `data.gqa_val_questions` in `configs/default.yaml` / `--val_questions`.
4. Eval:
   ```bash
   python -m qgsam_core.scripts.eval --checkpoint logs/qgsam/version_0/checkpoints/... --questions ... --images ...
   ```

### CLI flags (training)

- `--no_wandb`: TensorBoard only under `logs/qgsam/`.
- `--wandb_project`, `--wandb_run_name`: override W&B project/run name.
- `--tensorboard`: add TensorBoard alongside W&B.
- `--cpu`: force CPU even if GPU exists.

## Dependencies

- PyTorch, torchvision
- PyTorch Lightning
- `wandb` (for W&B logging)
- `transformers` (BERT tokenizer)
- `PyYAML` (config file)
