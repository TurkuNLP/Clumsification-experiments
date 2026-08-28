# This script has been co-created, refactored, and cleaned using GPT 5.6.
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class BinaryCollator:
    """Tokenize independent candidate texts with binary quality labels."""
    tokenizer: Any
    max_length: int
    text_prefix: str = ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            [f"{self.text_prefix}{feature['text']}" for feature in features],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", pad_to_multiple_of=8,
        )
        labels = [feature.get("label", feature.get("target")) for feature in features]
        if any(label is None for label in labels):
            raise ValueError("Binary examples require a 'label' field")
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "labels": torch.tensor(labels, dtype=torch.float32)}


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


@dataclass
class PairwiseCollator:
    """Tokenize explicit chosen/rejected rows for independent scoring."""

    tokenizer: Any
    max_length: int
    text_prefix: str = ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen = self.tokenizer(
            [f"{self.text_prefix}{feature['chosen_text']}" for feature in features],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", pad_to_multiple_of=8,
        )
        rejected = self.tokenizer(
            [f"{self.text_prefix}{feature['rejected_text']}" for feature in features],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", pad_to_multiple_of=8,
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "labels": torch.ones(len(features), dtype=torch.float32),
            "weights": torch.tensor(
                [feature.get("weight", 1.0) for feature in features],
                dtype=torch.float32,
            ),
        }
