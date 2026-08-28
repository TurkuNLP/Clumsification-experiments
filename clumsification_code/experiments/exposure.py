"""Approximate candidate/token exposure for factorial datasets."""
from __future__ import annotations

from typing import Any


def dataset_exposure(dataset: Any, tokenizer: Any, text_fields: tuple[str, ...] = ("text",)) -> dict[str, int]:
    candidates = 0
    tokens = 0
    for row in dataset:
        for field in text_fields:
            if field not in row:
                continue
            candidates += 1
            tokens += len(tokenizer(str(row[field]), add_special_tokens=True)["input_ids"])
    return {"rows": len(dataset), "candidate_presentations": candidates, "token_count": tokens}


def dataset_dict_exposure(dataset_dict: Any, tokenizer: Any, text_fields: tuple[str, ...] = ("text",)) -> dict[str, dict[str, int]]:
    return {split: dataset_exposure(data, tokenizer, text_fields) for split, data in dataset_dict.items()}
