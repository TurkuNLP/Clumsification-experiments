# This script has been co-created, refactored, and cleaned using GPT 5.6.
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class GroupedRankingCollator:
    """Existing collator for grouped pairwise ranking training."""

    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        flat_texts = []
        group_sizes = []
        max_group_size = max(len(feature["texts"]) for feature in features)
        padded_labels = []

        for feature in features:
            texts = feature["texts"]
            labels = feature["labels"]
            group_sizes.append(len(texts))
            flat_texts.extend(texts)
            padded_labels.append(
                list(labels) + [-100] * (max_group_size - len(labels))
            )

        tokenized = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.float32),
        }


@dataclass
class RegressionCollator:
    """Tokenize independent text/target regression examples."""

    tokenizer: Any
    max_length: int
    text_prefix: str = ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            [f"{self.text_prefix}{feature['text']}" for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(
                [feature["target"] for feature in features],
                dtype=torch.float32,
            ),
        }
