# This script has been co-created, refactored, and cleaned using GPT 5.6.
import json
import random
import shutil
from typing import Optional
import os

from datasets import Dataset, DatasetDict, load_from_disk

from clumsification_code.data.format_dataset import (
    format_custom_dataset,
    scored_original_ids,
)
from clumsification_code.data.pairing import (
    generate_training_pairs_random,
    get_aligned_candidate_items,
)
from clumsification_code.data.splitting import (
    assert_no_original_id_leakage,
    split_ids_to_metadata,
    split_original_ids_by_dataset,
)


def shuffle_and_transform_formatted_dataset(
    formatted_dataset: list[dict],
    score_names: list[str],
    seed: Optional[int] = None,
):
    """
    Convert chain records into a Hugging Face Dataset.

    The resulting rows retain the existing fields plus aligned identity fields:
      id, dataset_name, source_original_ids, texts, labels, candidate_ids,
      perturbation_sources

    Each discovered numeric score method is also stored as an aligned list:
      method_1: [score_or_none, ...]
    """
    rng = random.Random(seed)
    rows = []

    for entry in formatted_dataset:
        items = get_aligned_candidate_items(entry)

        # Preserve one-item chains as regression may use a scored original.
        if not items:
            continue

        rng.shuffle(items)

        row = {
            "id": str(entry["id"]),
            "dataset_name": entry.get("dataset_name"),
            "source_original_ids": list(entry.get("source_original_ids", [])),
            "texts": [item["text"] for item in items],
            "labels": [item["label"] for item in items],
            "candidate_ids": [item["candidate_id"] for item in items],
            "perturbation_sources": [item["perturbation_source"] for item in items],
        }

        for score_name in score_names:
            row[score_name] = [
                item["score_dict"].get(score_name) for item in items
            ]

        rows.append(row)

    return Dataset.from_list(rows)


def _downsample_dataset_dict(
    dataset_dict: DatasetDict,
    downsample_size: int,
    seed: int,
) -> DatasetDict:
    total = sum(len(dataset_dict[name]) for name in ("train", "dev", "test"))

    if downsample_size >= total:
        return dataset_dict

    if downsample_size < 3:
        raise ValueError("downsample_size must be at least 3 to keep train/dev/test.")

    split_names = ("train", "dev", "test")
    quotas = {
        name: max(1, int(round(downsample_size * len(dataset_dict[name]) / total)))
        for name in split_names
    }

    while sum(quotas.values()) > downsample_size:
        candidates = [name for name in split_names if quotas[name] > 1]
        name = max(candidates, key=lambda candidate: quotas[candidate])
        quotas[name] -= 1

    while sum(quotas.values()) < downsample_size:
        candidates = [
            name
            for name in split_names
            if quotas[name] < len(dataset_dict[name])
        ]

        if not candidates:
            break

        name = max(
            candidates,
            key=lambda candidate: len(dataset_dict[candidate]) - quotas[candidate],
        )
        quotas[name] += 1

    return DatasetDict(
        {
            name: dataset_dict[name].shuffle(seed=seed).select(range(quotas[name]))
            for name in split_names
        }
    )


def _collect_score_names(records_by_split: dict[str, list[dict]]) -> list[str]:
    score_names = set()

    for records in records_by_split.values():
        for entry in records:
            for score_dict in entry.get("item_score_dicts", []):
                score_names.update(score_dict.keys())

    return sorted(score_names)


def create_formatted_dataset_dict(
    dataset_names: list[str],
    max_layers: Optional[int] = None,
    layer_type: str = "clumsy",
    seed: int = 42,
    random_pairs: bool = False,
    reuse_limit: int = 5,
    downsample_size: Optional[int] = None,
    heldout_ratio: float = 0.3,
    test_ratio_within_heldout: float = 0.5,
    score_names: Optional[list[str]] = None,
    return_metadata: bool = False,
):
    """
    Build one shared train/dev/test DatasetDict.

    Splitting occurs by original custom_id before chains and random pairs are
    constructed. This prevents source-document leakage across splits.
    """
    if not dataset_names:
        raise ValueError("At least one dataset name must be supplied.")

    score_name_filter = set(score_names) if score_names is not None else None
    if score_name_filter is not None and not score_name_filter:
        raise ValueError("score_names must not be empty when supplied.")

    eligible_original_ids = None
    if score_name_filter is not None:
        layer_folders = {
            "clumsy": {"perturbed_layers"},
            "trad": {"trad_perturbed_layers"},
            "mix": {"perturbed_layers", "trad_perturbed_layers"},
            "all": {"perturbed_layers", "trad_perturbed_layers"},
        }[layer_type]
        eligible_original_ids = {
            dataset_name: scored_original_ids(
                dataset_name,
                layer_folders=layer_folders,
                score_names=score_name_filter,
            )
            for dataset_name in dataset_names
        }

    split_ids = split_original_ids_by_dataset(
        dataset_names=dataset_names,
        heldout_ratio=heldout_ratio,
        test_ratio_within_heldout=test_ratio_within_heldout,
        seed=seed,
        eligible_original_ids=eligible_original_ids,
    )
    assert_no_original_id_leakage(split_ids)

    records_by_split: dict[str, list[dict]] = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for split_name in ("train", "dev", "test"):
        for dataset_name in dataset_names:
            records = format_custom_dataset(
                custom_dataset_name=dataset_name,
                max_layers=max_layers,
                layer_type=layer_type,
                seed=seed,
                reuse_limit=reuse_limit,
                original_id_filter=split_ids[split_name][dataset_name],
            )

            if random_pairs and layer_type != "all":
                records = generate_training_pairs_random(
                    formatted_dataset=records,
                    seed=seed,
                    n=reuse_limit,
                )

            records_by_split[split_name].extend(records)

    score_names = _collect_score_names(records_by_split)

    final_splits = {
        split_name: shuffle_and_transform_formatted_dataset(
            formatted_dataset=records_by_split[split_name],
            score_names=score_names,
            seed=seed,
        ).shuffle(seed=seed)
        for split_name in ("train", "dev", "test")
    }

    if any(len(final_splits[name]) == 0 for name in ("train", "dev", "test")):
        sizes = {
            name: len(final_splits[name])
            for name in ("train", "dev", "test")
        }
        raise ValueError(
            "At least one split has no usable formatted examples after "
            f"chain construction: {sizes}"
        )

    dataset_dict = DatasetDict(final_splits)

    if downsample_size is not None:
        dataset_dict = _downsample_dataset_dict(
            dataset_dict=dataset_dict,
            downsample_size=downsample_size,
            seed=seed,
        )

    metadata = {
        "split_strategy": "original_custom_id_group_split_before_formatting",
        "split_original_ids": split_ids_to_metadata(split_ids),
        "score_fields": score_names,
        "score_aware_split_fields": sorted(score_name_filter)
        if score_name_filter is not None
        else None,
        "num_original_ids": {
            split_name: {
                dataset_name: len(ids)
                for dataset_name, ids in by_dataset.items()
            }
            for split_name, by_dataset in split_ids.items()
        },
    }

    if return_metadata:
        return dataset_dict, metadata

    return dataset_dict


def save_formatted_dataset_dict(
    dataset_dict: DatasetDict,
    output_path: str,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
):
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Formatted dataset already exists: {output_path}. "
                "Pass --overwrite to replace it."
            )

        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    if metadata is not None:
        metadata_path = os.path.join(output_path, "metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, indent=2)


def load_formatted_dataset_dict(path: str) -> DatasetDict:
    dataset_dict = load_from_disk(path)

    required_splits = {"train", "dev", "test"}
    actual_splits = set(dataset_dict.keys())
    missing = required_splits - actual_splits

    if missing:
        raise ValueError(
            f"Saved dataset at {path} is missing required split(s): {sorted(missing)}"
        )

    return dataset_dict
