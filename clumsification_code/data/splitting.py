# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Leakage-safe splitting of canonical source-document identities."""
from __future__ import annotations

import random
from typing import Iterable

from clumsification_code.data.repository import DatasetRepository


def split_original_ids_by_dataset(
    dataset_names: list[str],
    heldout_ratio: float = 0.3,
    test_ratio_within_heldout: float = 0.5,
    seed: int = 42,
    eligible_original_ids: dict[str, set[str]] | None = None,
    *,
    dataset_root: str = "data/custom_datasets",
    repositories: dict[str, DatasetRepository] | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Split canonical base-text IDs before any candidate composition."""
    if not 0.0 < heldout_ratio < 1.0:
        raise ValueError(f"heldout_ratio must be between 0 and 1, got {heldout_ratio}")
    if not 0.0 < test_ratio_within_heldout < 1.0:
        raise ValueError(
            "test_ratio_within_heldout must be between 0 and 1, "
            f"got {test_ratio_within_heldout}"
        )

    repository_map = repositories or {
        name: DatasetRepository.from_root(dataset_root, name)
        for name in dataset_names
    }
    groups: list[tuple[str, str]] = []
    for dataset_name in dataset_names:
        original_ids = {
            record.base_text_id for record in repository_map[dataset_name].read_originals()
        }
        if eligible_original_ids is not None:
            requested_ids = {
                str(value) for value in eligible_original_ids.get(dataset_name, set())
            }
            unknown_ids = requested_ids - original_ids
            if unknown_ids:
                raise ValueError(
                    f"{dataset_name}: eligible score IDs are not present in original.jsonl: "
                    f"{sorted(unknown_ids)[:20]}"
                )
            original_ids = requested_ids
        groups.extend((dataset_name, original_id) for original_id in sorted(original_ids))

    if len(groups) < 3:
        raise ValueError(
            "Need at least 3 original documents for train/dev/test splitting; "
            f"got {len(groups)}."
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_heldout = max(2, min(len(groups) - 1, int(round(len(groups) * heldout_ratio))))
    n_test = max(1, min(n_heldout - 1, int(round(n_heldout * test_ratio_within_heldout))))
    partitions = {
        "test": groups[:n_test],
        "dev": groups[n_test:n_heldout],
        "train": groups[n_heldout:],
    }
    result = {
        split: {dataset_name: set() for dataset_name in dataset_names}
        for split in ("train", "dev", "test")
    }
    for split, values in partitions.items():
        for dataset_name, original_id in values:
            result[split][dataset_name].add(original_id)
    return result


def split_ids_to_metadata(
    split_ids: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {
        split: {dataset: sorted(ids) for dataset, ids in by_dataset.items()}
        for split, by_dataset in split_ids.items()
    }


def assert_no_original_id_leakage(
    split_ids: dict[str, dict[str, Iterable[str]]],
) -> None:
    seen: dict[tuple[str, str], str] = {}
    for split, by_dataset in split_ids.items():
        for dataset_name, original_ids in by_dataset.items():
            for original_id in original_ids:
                key = (dataset_name, str(original_id))
                previous = seen.get(key)
                if previous is not None:
                    raise ValueError(
                        f"Original-document leakage: {dataset_name}:{original_id} "
                        f"appears in both {previous!r} and {split!r}."
                    )
                seen[key] = split
