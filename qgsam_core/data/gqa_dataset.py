"""
GQA dataset for QG-SAM: loads questions, answers, and optional region/box supervision.
Expects GQA JSON format (questions, answers, imageId) and image dir.
"""
import json
import os
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def build_image_transform(is_train: bool, size: int = 224):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class GQADataset(Dataset):
    """
    GQA VQA dataset. Each item: image tensor, question string, answer (str or idx),
    optional GT boxes for evidence supervision.
    """

    def __init__(
        self,
        questions_path: str,
        images_dir: str,
        answer_vocab_path: Optional[str] = None,
        top_k_answers: int = 500,
        unknown_answer_token: str = "unknown",
        max_samples: Optional[int] = None,
        is_train: bool = True,
        image_size: int = 224,
        answer_to_idx_override: Optional[Dict[str, int]] = None,
        idx_to_answer_override: Optional[List[str]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.transform = build_image_transform(is_train, image_size)
        self.top_k_answers = int(top_k_answers)
        self.unknown_answer_token = str(unknown_answer_token).strip().lower()

        with open(questions_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            self.samples = list(data.items())
            if max_samples:
                self.samples = self.samples[:max_samples]
        else:
            self.samples = data[:max_samples] if max_samples else data

        self.answer_to_idx: Dict[str, int] = {}
        self.idx_to_answer: List[str] = []
        self.answer_counts: Counter = Counter()
        if answer_vocab_path and os.path.isfile(answer_vocab_path):
            with open(answer_vocab_path, "r") as f:
                vocab = json.load(f)
            self.answer_to_idx = vocab.get("answer_to_idx", vocab)
            self.idx_to_answer = vocab.get("idx_to_answer", list(self.answer_to_idx.keys()))
            counts = vocab.get("answer_counts")
            if isinstance(counts, dict):
                self.answer_counts = Counter({str(k): int(v) for k, v in counts.items()})
        elif answer_to_idx_override is not None and idx_to_answer_override is not None:
            self.answer_to_idx = dict(answer_to_idx_override)
            self.idx_to_answer = list(idx_to_answer_override)
        else:
            self._build_vocab_from_samples()
        if self.unknown_answer_token not in self.answer_to_idx:
            self.answer_to_idx[self.unknown_answer_token] = len(self.idx_to_answer)
            self.idx_to_answer.append(self.unknown_answer_token)

    def _build_vocab_from_samples(self):
        answers: Counter = Counter()
        for s in self.samples:
            if isinstance(s, tuple):
                ann = s[1]
            else:
                ann = s
            a = ann.get("answer") or ann.get("answers", [{}])[0].get("answer")
            if a is not None:
                answers[str(a).strip().lower()] += 1
        self.answer_counts = answers
        most_common = answers.most_common(max(self.top_k_answers, 0))
        self.idx_to_answer = [a for a, _ in most_common]
        if self.unknown_answer_token not in self.idx_to_answer:
            self.idx_to_answer.append(self.unknown_answer_token)
        self.answer_to_idx = {a: i for i, a in enumerate(self.idx_to_answer)}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        s = self.samples[idx]
        if isinstance(s, tuple):
            qid, ann = s
        else:
            ann = s
            qid = ann.get("questionId", str(idx))
        return qid, ann

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        qid, ann = self._get_sample(idx)
        question = ann.get("question", "")
        answer_raw = ann.get("answer") or (ann.get("answers", [{}])[0].get("answer") if ann.get("answers") else None)
        answer_str = str(answer_raw).strip().lower() if answer_raw else ""
        image_id = ann.get("imageId", ann.get("image_id", ""))
        image_path = self.images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.images_dir / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        unk_id = self.answer_to_idx[self.unknown_answer_token]
        answer_id = self.answer_to_idx.get(answer_str, unk_id)
        gt_boxes = ann.get("boxes", ann.get("gt_boxes", None))
        return {
            "image": image_tensor,
            "question": question,
            "answer_id": answer_id,
            "answer_str": answer_str,
            "image_id": image_id,
            "question_id": qid,
            "gt_boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else None,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Counts answer classes after top-K filtering (includes unknown bucket)."""
        dist = Counter()
        unk_id = self.answer_to_idx[self.unknown_answer_token]
        for i in range(len(self.samples)):
            _, ann = self._get_sample(i)
            answer_raw = ann.get("answer") or (ann.get("answers", [{}])[0].get("answer") if ann.get("answers") else None)
            answer_str = str(answer_raw).strip().lower() if answer_raw else ""
            answer_id = self.answer_to_idx.get(answer_str, unk_id)
            dist[self.idx_to_answer[answer_id]] += 1
        return dict(dist)

    def get_class_weights(self, mode: str = "inverse_freq", min_weight: float = 0.1) -> torch.Tensor:
        """Class weights aligned with idx_to_answer for optional weighted CE."""
        dist = self.get_class_distribution()
        counts = torch.tensor(
            [float(dist.get(ans, 0)) for ans in self.idx_to_answer],
            dtype=torch.float32,
        )
        counts = torch.clamp(counts, min=1.0)
        if mode == "inverse_sqrt_freq":
            weights = 1.0 / torch.sqrt(counts)
        else:
            weights = 1.0 / counts
        weights = weights / weights.mean()
        return torch.clamp(weights, min=min_weight)


def collate_gqa(batch: List[Dict[str, Any]]):
    images = torch.stack([b["image"] for b in batch])
    questions = [b["question"] for b in batch]
    answer_ids = torch.tensor([b["answer_id"] for b in batch], dtype=torch.long)
    gt_boxes_list = [b["gt_boxes"] for b in batch]
    return {
        "image": images,
        "question": questions,
        "answer_id": answer_ids,
        "image_id": [b["image_id"] for b in batch],
        "question_id": [b["question_id"] for b in batch],
        "gt_boxes": gt_boxes_list,
    }
