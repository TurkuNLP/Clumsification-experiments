# This script has been co-created, refactored, and cleaned using GPT 5.6.

"""Convert source-grouped FE records into explicit training rows.

Chains are deliberately kept upstream for provenance and source-level split
checks.  The training boundary is flat: one candidate per regression row or
two independently scored candidates per ranking row.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from datasets import Dataset, DatasetDict


def _aligned_items(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a chain and return readable candidate records."""
    required = ("id", "texts", "labels", "candidate_ids", "perturbation_sources")
    missing = [name for name in required if name not in chain]
    if missing:
        raise ValueError(f"Chain is missing required field(s): {missing}")

    fields = {
        "texts": list(chain["texts"]),
        "labels": list(chain["labels"]),
        "candidate_ids": list(chain["candidate_ids"]),
        "perturbation_sources": list(chain["perturbation_sources"]),
    }
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Chain {chain['id']!r} has misaligned fields: {lengths}")
    if len(set(fields["candidate_ids"])) != len(fields["candidate_ids"]):
        raise ValueError(f"Chain {chain['id']!r} contains duplicate candidate_ids")

    score_names = [
        name
        for name in chain
        if name not in {
            "id", "dataset_name", "source_original_ids", "texts", "labels",
            "candidate_ids", "perturbation_sources",
        }
    ]
    scores = {
        name: list(chain[name])
        for name in score_names
    }
    for name, values in scores.items():
        if len(values) != len(fields["texts"]):
            raise ValueError(
                f"Chain {chain['id']!r} score field {name!r} is misaligned: "
                f"expected {len(fields['texts'])}, got {len(values)}"
            )

    return [
        {
            "source_id": str(chain["id"]),
            "dataset_name": chain.get("dataset_name"),
            "source_original_ids": list(chain.get("source_original_ids", [])),
            "text": fields["texts"][index],
            "layer": fields["labels"][index],
            "candidate_id": fields["candidate_ids"][index],
            "perturbation_source": fields["perturbation_sources"][index],
            "scores": {name: values[index] for name, values in scores.items()},
        }
        for index in range(len(fields["texts"]))
    ]


def assert_source_split_isolation(dataset_dict: DatasetDict) -> None:
    """Fail if one dataset/source document occurs in multiple splits."""
    seen: dict[tuple[str | None, str], str] = {}
    for split_name in ("train", "dev", "test"):
        if split_name not in dataset_dict:
            raise ValueError(f"Expected split {split_name!r} in DatasetDict")
        for chain in dataset_dict[split_name]:
            dataset_name = chain.get("dataset_name")
            for original_id in chain.get("source_original_ids", []):
                key = (dataset_name, str(original_id))
                previous = seen.get(key)
                if previous is not None and previous != split_name:
                    raise ValueError(
                        f"Source leakage: {dataset_name!r}:{original_id!r} "
                        f"appears in {previous!r} and {split_name!r}"
                    )
                seen[key] = split_name


def flatten_regression_dataset(dataset: Dataset, score_name: str) -> Dataset:
    """Flatten chains into one row per candidate with a finite score."""
    rows: list[dict[str, Any]] = []
    for chain in dataset:
        for item in _aligned_items(chain):
            if score_name not in item["scores"]:
                raise ValueError(
                    f"Chain {item['source_id']!r} has no score field {score_name!r}"
                )
            value = item["scores"][score_name]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Non-numeric {score_name!r} value for {item['candidate_id']!r}: {value!r}"
                )
            if not math.isfinite(float(value)):
                continue
            rows.append({
                "input_id": item["candidate_id"],
                "source_id": item["source_id"],
                "dataset_name": item["dataset_name"],
                "source_original_ids": item["source_original_ids"],
                "text": item["text"],
                "label": float(value),
                "layer": item["layer"],
                "perturbation_source": item["perturbation_source"],
            })
    if not rows:
        raise ValueError(f"No valid regression rows found for score {score_name!r}")
    return Dataset.from_list(rows)


def flatten_pairwise_dataset(
    dataset: Dataset,
    *,
    policy: Literal["all_unequal_layers"] = "all_unequal_layers",
    include_weight: bool = True,
) -> Dataset:
    """Flatten each chain into deterministic chosen/rejected candidate pairs."""
    if policy != "all_unequal_layers":
        raise ValueError(f"Unknown pair policy: {policy!r}")

    rows: list[dict[str, Any]] = []
    for chain in dataset:
        items = _aligned_items(chain)
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                first, second = items[left], items[right]
                if first["layer"] == second["layer"]:
                    continue
                chosen, rejected = (
                    (first, second)
                    if first["layer"] < second["layer"]
                    else (second, first)
                )
                row = {
                    "pair_id": f"{chain['id']}__pair_{left}_{right}",
                    "source_id": str(chain["id"]),
                    "dataset_name": chain.get("dataset_name"),
                    "source_original_ids": list(chain.get("source_original_ids", [])),
                    "chosen_id": chosen["candidate_id"],
                    "rejected_id": rejected["candidate_id"],
                    "chosen_text": chosen["text"],
                    "rejected_text": rejected["text"],
                    "chosen_layer": chosen["layer"],
                    "rejected_layer": rejected["layer"],
                }
                if include_weight:
                    row["weight"] = float(abs(chosen["layer"] - rejected["layer"]))
                rows.append(row)

    if not rows:
        raise ValueError("No unequal-label pairwise rows could be constructed")
    return Dataset.from_list(rows)


def flatten_dataset_dict(
    dataset_dict: DatasetDict,
    *,
    training_method: Literal["regression", "pairwise"],
    score_name: str | None = None,
    pair_policy: Literal["all_unequal_layers"] = "all_unequal_layers",
) -> DatasetDict:
    """Validate split isolation and flatten every split with one schema."""
    assert_source_split_isolation(dataset_dict)
    if training_method == "regression" and not score_name:
        raise ValueError("score_name is required for regression flattening")
    if training_method not in {"regression", "pairwise"}:
        raise ValueError(f"Unknown training method: {training_method!r}")

    if training_method == "regression":
        def builder(split: Dataset) -> Dataset:
            return flatten_regression_dataset(split, score_name)  # type: ignore[arg-type]
    else:
        def builder(split: Dataset) -> Dataset:
            return flatten_pairwise_dataset(split, policy=pair_policy)

    return DatasetDict({name: builder(dataset_dict[name]) for name in ("train", "dev", "test")})
