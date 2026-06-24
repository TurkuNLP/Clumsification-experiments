from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class GroupAllPairsCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        flat_texts = []
        group_sizes = []

        max_group_size = max(len(f["texts"]) for f in features)
        padded_labels = []

        for f in features:
            texts = f["texts"]
            labels = f["labels"]

            group_sizes.append(len(texts))
            flat_texts.extend(texts)

            padded = list(labels) + [-100] * (max_group_size - len(labels))
            padded_labels.append(padded)

        tok = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.float32),
        }