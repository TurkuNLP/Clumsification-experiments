# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Active regression dataset preparation and target-scaling utilities."""
from __future__ import annotations
import math
from typing import Any
from datasets import Dataset, DatasetDict
from .utils import logger

def _valid_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def flatten_regression_split(dataset: Dataset, score_name: str, split_name: str) -> tuple[Dataset, dict[str, int]]:
    if score_name not in dataset.column_names:
        raise ValueError(f"Split {split_name!r} does not contain score field {score_name!r}.")
    rows, missing, invalid = [], 0, 0
    for chain in dataset:
        fields = {"texts": list(chain["texts"]), "labels": list(chain["labels"]), "scores": list(chain[score_name]), "candidate_ids": list(chain.get("candidate_ids", [])), "perturbation_sources": list(chain.get("perturbation_sources", []))}
        lengths = {name: len(values) for name, values in fields.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Misaligned fields in chain {chain.get('id')!r}: {lengths}")
        for index, (text, layer, score) in enumerate(zip(fields["texts"], fields["labels"], fields["scores"])):
            if score is None:
                missing += 1; continue
            if not _valid_score(score):
                invalid += 1; continue
            rows.append({"text": text, "raw_target": float(score), "chain_id": str(chain["id"]), "dataset_name": chain.get("dataset_name"), "source_original_ids": chain.get("source_original_ids", []), "layer": int(layer), "candidate_id": fields["candidate_ids"][index], "perturbation_source": fields["perturbation_sources"][index], "item_index_in_chain": index})
    stats = {"accepted": len(rows), "skipped_missing": missing, "skipped_invalid": invalid}
    if not rows:
        raise ValueError(f"No valid regression examples in split {split_name!r}: {stats}")
    logger.info("Regression split %s for %s: %s", split_name, score_name, stats)
    return Dataset.from_list(rows), stats

def build_regression_dataset_dict(grouped_dataset_dict: DatasetDict, score_name: str) -> tuple[DatasetDict, dict[str, Any]]:
    flattened, statistics = {}, {}
    for split in ("train", "dev", "test"):
        flattened[split], statistics[split] = flatten_regression_split(grouped_dataset_dict[split], score_name, split)
    values = [float(value) for value in flattened["train"]["raw_target"]]
    train_min, train_max = min(values), max(values)
    if train_min == train_max:
        raise ValueError("Regression target is constant in the training split")
    denominator = train_max - train_min
    scaled = DatasetDict({split: data.map(lambda row: {"target": (float(row["raw_target"]) - train_min) / denominator}) for split, data in flattened.items()})
    return scaled, {"training_method": "regression", "score_name": score_name, "target_scaling": {"method": "minmax", "fit_split": "train", "train_min": train_min, "train_max": train_max, "clip": False}, "split_statistics": statistics}

__all__ = ["build_regression_dataset_dict", "flatten_regression_split"]
