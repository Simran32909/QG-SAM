"""
VQA-X / VQA-HAT dataset for evidence-quality evaluation.
Loads questions, answers, and human attention / region annotations for mask IoU.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def build_image_transform_vqax(is_train: bool, size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class VQAXDataset(Dataset):
    """
    VQA-X style: question, answer, image, and GT regions/boxes for evidence evaluation.
    """

    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        answer_vocab_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        image_size: int = 224,
    ):
        self.images_dir = Path(images_dir)
        self.transform = build_image_transform_vqax(False, image_size)
        with open(annotations_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "questions" in data:
            qs = data["questions"]
            ans = {a["question_id"]: a for a in data.get("annotations", [])}
            self.samples = []
            for q in qs[:max_samples] if max_samples else qs:
                qid = q.get("question_id", q.get("question_id"))
                a = ans.get(qid, {})
                self.samples.append({
                    "question": q.get("question", ""),
                    "question_id": qid,
                    "image_id": q.get("image_id", a.get("image_id", "")),
                    "answer": a.get("answer", a.get("multiple_choice_answer", "")),
                    "answers": a.get("answers", []),
                    "gt_boxes": a.get("gt_boxes", a.get("regions", [])),
                })
        else:
            self.samples = data[:max_samples] if max_samples else data

        self.answer_to_idx: Dict[str, int] = {}
        self.idx_to_answer: List[str] = []
        if answer_vocab_path:
            try:
                with open(answer_vocab_path, "r") as f:
                    vocab = json.load(f)
                self.answer_to_idx = vocab.get("answer_to_idx", vocab)
                self.idx_to_answer = vocab.get("idx_to_answer", list(self.answer_to_idx.keys()))
            except Exception:
                self._build_vocab()
        else:
            self._build_vocab()

    def _build_vocab(self):
        answers = set()
        for s in self.samples:
            a = s.get("answer") or (s.get("answers"), [{}])[0].get("answer") if s.get("answers") else None
            if a:
                answers.add(str(a).strip().lower())
        self.idx_to_answer = sorted(answers)
        self.answer_to_idx = {a: i for i, a in enumerate(self.idx_to_answer)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image_id = s.get("image_id", "")
        image_path = self.images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.images_dir / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        answer_str = str(s.get("answer", "")).strip().lower()
        answer_idx = self.answer_to_idx.get(answer_str, 0)
        gt_boxes = s.get("gt_boxes")
        gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else None
        return {
            "image": image_tensor,
            "question": s.get("question", ""),
            "answer_idx": answer_idx,
            "answer_str": answer_str,
            "image_id": image_id,
            "question_id": s.get("question_id", idx),
            "gt_boxes": gt_tensor,
        }


def collate_vqax(batch: List[Dict[str, Any]]):
    images = torch.stack([b["image"] for b in batch])
    return {
        "image": images,
        "question": [b["question"] for b in batch],
        "answer_idx": torch.tensor([b["answer_idx"] for b in batch], dtype=torch.long),
        "image_id": [b["image_id"] for b in batch],
        "question_id": [b["question_id"] for b in batch],
        "gt_boxes": [b["gt_boxes"] for b in batch if b["gt_boxes"] is not None],
    }
