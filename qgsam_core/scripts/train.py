import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import torch
import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

from qgsam_core.data import GQADataset, collate_gqa
from qgsam_core.models import QGSAM
from qgsam_core.train import QGSAMLightning


def _infer_val_questions_path(train_questions: str) -> Path:
    """If train file is *train*_balanced_*.json, try sibling val_balanced_questions.json."""
    p = Path(train_questions)
    parent = p.parent
    name = p.name
    if "train" in name:
        candidate = parent / name.replace("train", "val", 1)
        if candidate.is_file():
            return candidate
    return parent / "val_balanced_questions.json"


def _build_loggers(cfg: dict, args, use_wandb: bool):
    loggers = []
    wb_cfg = cfg.get("wandb", {})
    if use_wandb and wb_cfg.get("enabled", True):
        try:
            from pytorch_lightning.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=args.wandb_project or wb_cfg.get("project", "qgsam"),
                    name=args.wandb_run_name or wb_cfg.get("run_name"),
                    log_model=bool(wb_cfg.get("log_model", False)),
                )
            )
        except ImportError:
            print("wandb not installed; run: pip install wandb. Falling back to TensorBoard only.")
    if args.tensorboard or not loggers:
        loggers.append(TensorBoardLogger("logs", name="qgsam"))
    return loggers if len(loggers) > 1 else loggers[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="qgsam_core/configs/default.yaml")
    parser.add_argument("--questions", type=str, help="GQA train questions JSON")
    parser.add_argument("--val_questions", type=str, default=None, help="GQA val questions JSON (default: infer from train path)")
    parser.add_argument("--images", type=str, help="GQA images dir")
    parser.add_argument("--answer_vocab", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dev", action="store_true", help="Quick run 1 epoch, 10 batches")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true", help="Also log TensorBoard (on by top if W&B disabled)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    t_cfg = cfg["train"]

    questions = args.questions or d_cfg.get("gqa_questions")
    images = args.images or d_cfg.get("gqa_images")
    if not questions or not images:
        print("Provide --questions and --images (or set in config). Exiting.")
        return

    print(f"Train questions: {questions}")
    print(f"Images dir:      {images}")

    val_questions = (
        args.val_questions
        or d_cfg.get("gqa_val_questions")
        or str(_infer_val_questions_path(questions))
    )
    if not Path(val_questions).is_file():
        print(f"Warning: val questions not found at {val_questions}. Validation disabled.")
        val_questions = None
    else:
        print(f"Val questions:   {val_questions}")

    max_epochs = args.max_epochs if args.max_epochs is not None else t_cfg.get("max_epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else d_cfg.get("batch_size", 32)
    lr = args.lr if args.lr is not None else t_cfg.get("lr", 1e-4)
    top_k_answers = int(d_cfg.get("top_k_answers", 500))
    unknown_answer_token = str(d_cfg.get("unknown_answer_token", "unknown")).strip().lower()

    train_ds = GQADataset(
        questions_path=questions,
        images_dir=images,
        answer_vocab_path=args.answer_vocab or d_cfg.get("answer_vocab"),
        top_k_answers=top_k_answers,
        unknown_answer_token=unknown_answer_token,
        max_samples=100 if args.dev else d_cfg.get("max_train_samples"),
        is_train=True,
        image_size=d_cfg.get("image_size", 224),
    )
    num_answers = len(train_ds.idx_to_answer)
    unknown_id = train_ds.answer_to_idx.get(unknown_answer_token, -1)
    print(f"Answer vocab size after top-K filtering: {num_answers}")
    print(f"Unknown token/id: '{unknown_answer_token}'/{unknown_id}")
    train_dist = train_ds.get_class_distribution()
    top_dist = sorted(train_dist.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print(f"Top-10 answer classes (post-filter): {top_dist}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=d_cfg.get("num_workers", 4),
        collate_fn=collate_gqa,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )

    val_loader = None
    if val_questions:
        val_ds = GQADataset(
            questions_path=val_questions,
            images_dir=images,
            answer_vocab_path=args.answer_vocab or d_cfg.get("answer_vocab"),
            max_samples=500 if args.dev else d_cfg.get("max_val_samples"),
            is_train=False,
            image_size=d_cfg.get("image_size", 224),
            answer_to_idx_override=train_ds.answer_to_idx,
            idx_to_answer_override=train_ds.idx_to_answer,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=d_cfg.get("num_workers", 4),
            collate_fn=collate_gqa,
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )

    m_cfg = cfg["model"]
    model = QGSAM(
        num_answers=num_answers,
        hidden_size=m_cfg.get("hidden_size", 512),
        image_feat_dim=m_cfg.get("image_feat_dim", 768),
        question_dim=m_cfg.get("question_dim", 768),
        num_masks=m_cfg.get("num_masks", 16),
        num_heads=m_cfg.get("num_heads", 8),
        num_cross_layers=m_cfg.get("num_cross_layers", 1),
        num_reasoner_layers=m_cfg.get("num_reasoner_layers", 2),
        dropout=m_cfg.get("dropout", 0.1),
        clip_model=m_cfg.get("clip_model", "openai/clip-vit-base-patch16"),
    )

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pass

    class_weights = None
    if t_cfg.get("use_class_weights", False):
        mode = str(t_cfg.get("class_weight_mode", "inverse_freq"))
        min_w = float(t_cfg.get("class_weight_min", 0.1))
        class_weights = train_ds.get_class_weights(mode=mode, min_weight=min_w)
        print(f"Class weighting enabled ({mode}), min={min_w}")

    pl_model = QGSAMLightning(
        model=model,
        num_answers=num_answers,
        idx_to_answer=train_ds.idx_to_answer,
        tokenizer=tokenizer,
        lr=float(lr),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        use_seg_loss=t_cfg.get("use_seg_loss", True),
        use_clip_loss=t_cfg.get("use_clip_loss", False),
        seg_loss_weight=t_cfg.get("seg_loss_weight", 1.0),
        clip_loss_weight=t_cfg.get("clip_loss_weight", 0.1),
        label_smoothing=float(t_cfg.get("label_smoothing", 0.1)),
        class_weights=class_weights,
    )

    use_wandb = not args.no_wandb
    logger = _build_loggers(cfg, args, use_wandb)

    if args.cpu or not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1
        if not args.cpu and not torch.cuda.is_available():
            print("CUDA not available; training on CPU.")
    else:
        accelerator = t_cfg.get("accelerator", "gpu")
        devices = int(t_cfg.get("devices", 1))

    log_every_n_steps = int(t_cfg.get("log_every_n_steps", 10))
    val_check_interval = t_cfg.get("val_check_interval", None)

    # Custom Checkpoint Callback
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath="qgsam_core/logs/checkpoints",
        filename=f"epoch{{epoch:02d}}_acc{{val_acc_top1:.4f}}_{timestamp}",
        monitor="val_acc_top1",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    trainer_kwargs = dict(
        max_epochs=1 if args.dev else max_epochs,
        limit_train_batches=10 if args.dev else None,
        limit_val_batches=5 if args.dev and val_loader else None,
        gradient_clip_val=t_cfg.get("gradient_clip_val", 1.0),
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
    )

    # When validation is enabled, optionally run validation more frequently
    # (e.g. every N train batches) to get val losses in-between epochs.
    if val_loader is not None and val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = val_check_interval
    prec = t_cfg.get("precision")
    if prec:
        trainer_kwargs["precision"] = prec

    trainer = pl.Trainer(**trainer_kwargs)

    if val_loader is not None:
        trainer.fit(pl_model, train_loader, val_loader)
    else:
        trainer.fit(pl_model, train_loader)


if __name__ == "__main__":
    main()
