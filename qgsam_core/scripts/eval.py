"""
Evaluate QG-SAM checkpoint.
Usage: python -m qgsam_core.scripts.eval --checkpoint path/to/ckpt --questions ... --images ...
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from torch.utils.data import DataLoader

from qgsam_core.data import GQADataset, collate_gqa
from qgsam_core.models import QGSAM
from qgsam_core.train import QGSAMLightning
from qgsam_core.eval import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--answer_vocab", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model = QGSAMLightning.load_from_checkpoint(args.checkpoint, map_location=device)
    model = pl_model.model
    model.to(device)
    model.eval()

    tokenizer = getattr(pl_model, "tokenizer", None)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            pass

    ds = GQADataset(
        questions_path=args.questions,
        images_dir=args.images,
        answer_vocab_path=args.answer_vocab,
        is_train=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_gqa)

    metrics = run_evaluation(model, loader, device, tokenizer)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
