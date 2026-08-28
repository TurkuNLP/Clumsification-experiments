# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Dataset and collators for UniEval Boolean-QA training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .data import iter_unieval_rows


class UniEvalDataset(Dataset):
    """In-memory validated view of one released JSONL file."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def from_jsonl(cls, path: str) -> "UniEvalDataset":
        return cls(list(iter_unieval_rows(path)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {"text": row["src"], "label": 1 if row["tgt"] == "Yes" else 0}

    def train_dev_split(self, dev_fraction: float = 0.05, seed: int = 42):
        if not 0 < dev_fraction < 1:
            raise ValueError("dev_fraction must be between 0 and 1")
        indices = list(range(len(self)))
        random.Random(seed).shuffle(indices)
        dev_size = max(1, round(len(indices) * dev_fraction))
        return (
            UniEvalDataset([self.rows[i] for i in indices[dev_size:]]),
            UniEvalDataset([self.rows[i] for i in indices[:dev_size]]),
        )


@dataclass
class EncoderUniEvalCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [feature["text"] for feature in features],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [feature["label"] for feature in features], dtype=torch.float32
        )
        return encoded


@dataclass
class GenerativeUniEvalCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [feature["text"] for feature in features],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            ["Yes" if feature["label"] else "No" for feature in features],
            padding=True, truncation=True, max_length=4,
            return_tensors="pt",
        )["input_ids"]
        pad = self.tokenizer.pad_token_id
        targets[targets == pad] = -100
        encoded["labels"] = targets
        return encoded
