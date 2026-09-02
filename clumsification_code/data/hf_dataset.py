# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Build Hugging Face datasets from the canonical candidate graph."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk

from clumsification_code.data.candidate_identity import make_original_candidate_id
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import COMPOSITION_POLICIES, HFBuildSpec, PAIR_POLICIES
from clumsification_code.data.splitting import (
    assert_no_original_id_leakage,
    split_ids_to_metadata,
    split_original_ids_by_dataset,
)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _original_item(repository: DatasetRepository, original: Any) -> dict[str, Any]:
    return {
        "text": original.text,
        "label": 0,
        "candidate_id": make_original_candidate_id(
            dataset_name=repository.dataset_name, base_text_id=original.base_text_id
        ),
        "perturbation_source": "original",
        "perturbation_method": "original",
        "perturbation_run_id": "original",
        "parent_candidate_id": None,
        "source_layer": 0,
        "source_method": None,
        "source_run_id": None,
        "score_dict": {},
    }


def _candidate_item(candidate: Any) -> dict[str, Any]:
    return {
        "text": candidate.text,
        "label": candidate.target_layer,
        "candidate_id": candidate.candidate_id,
        "perturbation_source": candidate.perturbation_source,
        "perturbation_method": candidate.perturbation_method,
        "perturbation_run_id": candidate.run_id,
        "parent_candidate_id": candidate.parent_candidate_id,
        "source_layer": candidate.source_layer,
        "source_method": candidate.source_method,
        "source_run_id": candidate.source_run_id,
        "score_dict": {},
    }


def _attach_scores(
    repository: DatasetRepository,
    items_by_id: dict[str, dict[str, Any]],
    score_names: set[str] | None,
    score_run_ids: list[str] | None,
) -> list[str]:
    if score_names is not None and not score_names:
        raise ValueError("score_names must not be empty when supplied")
    discovered: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for score in repository.read_scores(
        scoring_methods=score_names, scoring_run_ids=score_run_ids
    ):
        item = items_by_id.get(score.candidate_id)
        if item is None:
            continue
        identity = (score.candidate_id, score.scoring_method)
        if identity in identities:
            raise ValueError(
                "Multiple scoring runs were selected for candidate/method "
                f"{score.candidate_id!r}/{score.scoring_method!r}"
            )
        identities.add(identity)
        item["score_dict"][score.scoring_method] = score.score_value
        discovered.add(score.scoring_method)
    return sorted(discovered)


def _load_repository_groups(
    repository: DatasetRepository,
    *,
    methods: list[str] | None,
    run_ids: list[str] | None,
    layers: list[int] | None,
    score_names: set[str] | None,
    score_run_ids: list[str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    repository.validate_lineage()
    groups = {
        original.base_text_id: [_original_item(repository, original)]
        for original in repository.read_originals()
    }
    items_by_id = {
        item["candidate_id"]: item for items in groups.values() for item in items
    }
    for entry in repository.list_layers(
        methods=methods, run_ids=run_ids, target_layers=layers
    ):
        for candidate in repository.read_candidates(entry):
            item = _candidate_item(candidate)
            groups[candidate.base_text_id].append(item)
            items_by_id[item["candidate_id"]] = item
    discovered = _attach_scores(repository, items_by_id, score_names, score_run_ids)
    for items in groups.values():
        items.sort(key=lambda item: (
            int(item["label"]), str(item["perturbation_method"]),
            str(item["perturbation_run_id"]), str(item["candidate_id"]),
        ))
    return groups, discovered


def _compose_items(
    items: list[dict[str, Any]],
    *,
    composition: str,
    method_weights: dict[str, float] | None,
    samples_per_source: int,
    seed: int,
) -> list[dict[str, Any]]:
    if composition not in COMPOSITION_POLICIES:
        raise ValueError(f"Unknown composition policy: {composition!r}")
    originals = [item for item in items if item["perturbation_method"] == "original"]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item["perturbation_method"] != "original":
            by_method.setdefault(str(item["perturbation_method"]), []).append(item)
    if composition == "all" or not by_method:
        return list(items)
    rng = random.Random(seed)
    methods = sorted(by_method)
    selected: list[dict[str, Any]] = []
    if composition == "source_exclusive":
        weights = [float((method_weights or {}).get(method, 1.0)) for method in methods]
        if any(weight < 0 for weight in weights) or not any(weights):
            raise ValueError("Method weights must be non-negative and not all zero")
        selected.extend(by_method[rng.choices(methods, weights=weights, k=1)[0]])
    elif composition == "fixed_per_source":
        for method in methods:
            values = list(by_method[method])
            rng.shuffle(values)
            selected.extend(values[:samples_per_source])
    elif composition == "balanced":
        quota = min(len(values) for values in by_method.values())
        for method in methods:
            values = list(by_method[method])
            rng.shuffle(values)
            selected.extend(values[:quota])
    else:
        weights = {
            method: float((method_weights or {}).get(method, 0.0))
            for method in methods
        }
        if any(weight < 0 for weight in weights.values()) or not any(weights.values()):
            raise ValueError("Weighted composition requires positive method weights")
        target = min(sum(map(len, by_method.values())), samples_per_source * len(methods))
        choices = [item for method in methods for item in by_method[method]]
        choice_weights = [weights[str(item["perturbation_method"])] for item in choices]
        while choices and len(selected) < target and any(choice_weights):
            index = rng.choices(range(len(choices)), weights=choice_weights, k=1)[0]
            selected.append(choices.pop(index))
            choice_weights.pop(index)
    return originals + sorted(selected, key=lambda item: (
        int(item["label"]), str(item["perturbation_method"]), str(item["candidate_id"])
    ))


def _chain(
    dataset_name: str,
    base_text_ids: list[str],
    items: list[dict[str, Any]],
    chain_id: str,
) -> dict[str, Any]:
    return {
        "id": chain_id,
        "dataset_name": dataset_name,
        "source_original_ids": base_text_ids,
        "items": items,
    }


def _apply_pair_policy(
    chains: list[dict[str, Any]],
    *,
    policy: str,
    reuse_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if policy not in PAIR_POLICIES:
        raise ValueError(f"Unknown pair policy: {policy!r}")
    if policy == "none":
        return chains
    pairs: list[dict[str, Any]] = []
    if policy == "cross_source_unmatched":
        pool = [(chain, item) for chain in chains for item in chain["items"]]
        rng = random.Random(seed)
        rng.shuffle(pool)
        reuse = {item["candidate_id"]: 0 for _, item in pool}
        for left_index, (left_chain, left) in enumerate(pool):
            if reuse[left["candidate_id"]] >= reuse_limit:
                continue
            alternatives = [
                (chain, item) for chain, item in pool[left_index + 1:]
                if chain["dataset_name"] == left_chain["dataset_name"]
                and chain["source_original_ids"] != left_chain["source_original_ids"]
                and item["label"] != left["label"]
                and reuse[item["candidate_id"]] < reuse_limit
            ]
            if not alternatives:
                continue
            right_chain, right = rng.choice(alternatives)
            reuse[left["candidate_id"]] += 1
            reuse[right["candidate_id"]] += 1
            pairs.append(_chain(
                left_chain["dataset_name"],
                left_chain["source_original_ids"] + right_chain["source_original_ids"],
                [left, right],
                f"unmatched__{left['candidate_id']}__{right['candidate_id']}",
            ))
        return pairs

    for chain in chains:
        items = chain["items"]
        by_id = {item["candidate_id"]: item for item in items}
        original = next(
            (item for item in items if item["perturbation_method"] == "original"), None
        )
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if policy == "parent_child":
            candidate_pairs = [
                (by_id[item["parent_candidate_id"]], item)
                for item in items if item["parent_candidate_id"] in by_id
            ]
        elif policy == "original_only" and original is not None:
            candidate_pairs = [(original, item) for item in items if item is not original]
        elif policy == "all_unequal_layers":
            candidate_pairs = [
                (items[left], items[right])
                for left in range(len(items)) for right in range(left + 1, len(items))
                if items[left]["label"] != items[right]["label"]
            ]
        for index, (left, right) in enumerate(candidate_pairs):
            pairs.append(_chain(
                chain["dataset_name"], chain["source_original_ids"], [left, right],
                f"{chain['id']}__{policy}_{index}",
            ))
    return pairs


def _rows_from_chains(
    chains: list[dict[str, Any]], score_names: list[str], seed: int
) -> Dataset:
    rows = []
    for chain in chains:
        items = list(chain["items"])
        random.Random(_stable_seed(seed, chain["id"])).shuffle(items)
        row = {
            "id": str(chain["id"]),
            "dataset_name": chain["dataset_name"],
            "source_original_ids": list(chain["source_original_ids"]),
            "texts": [item["text"] for item in items],
            "labels": [item["label"] for item in items],
            "candidate_ids": [item["candidate_id"] for item in items],
            "perturbation_sources": [item["perturbation_source"] for item in items],
            "perturbation_methods": [item["perturbation_method"] for item in items],
            "perturbation_run_ids": [item["perturbation_run_id"] for item in items],
            "parent_candidate_ids": [item["parent_candidate_id"] for item in items],
            "source_layers": [item["source_layer"] for item in items],
            "source_methods": [item["source_method"] for item in items],
            "source_run_ids": [item["source_run_id"] for item in items],
        }
        for score_name in score_names:
            row[score_name] = [item["score_dict"].get(score_name) for item in items]
        rows.append(row)
    return Dataset.from_list(rows)


def _downsample_dataset_dict(
    dataset_dict: DatasetDict, downsample_size: int, seed: int
) -> DatasetDict:
    total = sum(len(dataset_dict[name]) for name in ("train", "dev", "test"))
    if downsample_size >= total:
        return dataset_dict
    if downsample_size < 3:
        raise ValueError("downsample_size must be at least 3")
    quotas = {name: 1 for name in ("train", "dev", "test")}
    remaining = downsample_size - 3
    while remaining:
        available = [name for name in quotas if quotas[name] < len(dataset_dict[name])]
        if not available:
            break
        name = max(available, key=lambda value: len(dataset_dict[value]) - quotas[value])
        quotas[name] += 1
        remaining -= 1
    return DatasetDict({
        name: dataset_dict[name].shuffle(seed=seed).select(range(quotas[name]))
        for name in quotas
    })


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
    methods: Optional[list[str]] = None,
    composition: str = "all",
    method_weights: Optional[dict[str, float]] = None,
    samples_per_source: int = 1,
    return_metadata: bool = False,
    *,
    dataset_root: str = "data/custom_datasets",
    run_ids: Optional[list[str]] = None,
    include_layers: Optional[list[int]] = None,
    pair_policy: str = "none",
    score_run_ids: Optional[list[str]] = None,
):
    """Build source-isolated HF splits from manifests and parent links."""
    if not dataset_names:
        raise ValueError("At least one dataset name must be supplied")
    if include_layers is None and max_layers is not None:
        include_layers = list(range(1, max_layers + 1))
    if methods is None and layer_type == "trad":
        methods = ["unieval", "unieval_trad", "trad_single", "trad_multi"]
    if random_pairs:
        pair_policy = "cross_source_unmatched"
    repositories = {
        name: DatasetRepository.from_root(dataset_root, name) for name in dataset_names
    }
    groups_by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    discovered_scores: set[str] = set()
    requested_scores = set(score_names) if score_names is not None else None
    for name, repository in repositories.items():
        groups, found_scores = _load_repository_groups(
            repository, methods=methods, run_ids=run_ids, layers=include_layers,
            score_names=requested_scores, score_run_ids=score_run_ids,
        )
        groups_by_dataset[name] = groups
        discovered_scores.update(found_scores)

    eligible_ids = None
    if requested_scores is not None:
        missing = requested_scores - discovered_scores
        if missing:
            raise ValueError(f"No selected score records found for: {sorted(missing)}")
        eligible_ids = {
            name: {
                base_id for base_id, items in groups.items()
                if any(requested_scores <= set(item["score_dict"]) for item in items)
            }
            for name, groups in groups_by_dataset.items()
        }
    split_ids = split_original_ids_by_dataset(
        dataset_names, heldout_ratio, test_ratio_within_heldout, seed,
        eligible_ids, dataset_root=dataset_root, repositories=repositories,
    )
    assert_no_original_id_leakage(split_ids)
    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "dev", "test")
    }
    for split in records:
        for dataset_name in dataset_names:
            for base_id in sorted(split_ids[split][dataset_name]):
                items = _compose_items(
                    groups_by_dataset[dataset_name][base_id], composition=composition,
                    method_weights=method_weights, samples_per_source=samples_per_source,
                    seed=_stable_seed(seed, dataset_name, base_id, composition),
                )
                records[split].append(
                    _chain(dataset_name, [base_id], items, f"{dataset_name}:{base_id}")
                )
        records[split] = _apply_pair_policy(
            records[split], policy=pair_policy, reuse_limit=reuse_limit,
            seed=_stable_seed(seed, split, pair_policy),
        )
    final = DatasetDict({
        split: _rows_from_chains(chains, sorted(discovered_scores), seed).shuffle(seed=seed)
        for split, chains in records.items()
    })
    if any(len(final[split]) == 0 for split in final):
        sizes = {split: len(final[split]) for split in final}
        raise ValueError(f"At least one split has no usable examples: {sizes}")
    if downsample_size is not None:
        final = _downsample_dataset_dict(final, downsample_size, seed)
    metadata = {
        "split_strategy": "canonical_base_text_id_before_composition",
        "split_original_ids": split_ids_to_metadata(split_ids),
        "score_fields": sorted(discovered_scores),
        "score_run_ids": score_run_ids,
        "include_methods": methods,
        "include_runs": run_ids,
        "include_layers": include_layers,
        "composition": composition,
        "method_weights": method_weights,
        "samples_per_source": samples_per_source,
        "pair_policy": pair_policy,
        "num_examples": {split: len(final[split]) for split in final},
        "selected_layer_hashes": {
            dataset_name: {
                f"{entry.method}:{entry.run_id}:{entry.target_layer}": {
                    "content_hash": entry.content_hash,
                    "config_hash": entry.config_hash,
                }
                for entry in repositories[dataset_name].list_layers(
                    methods=methods, run_ids=run_ids, target_layers=include_layers
                )
            }
            for dataset_name in dataset_names
        },
    }
    return (final, metadata) if return_metadata else final


def save_formatted_dataset_dict(
    dataset_dict: DatasetDict,
    output_path: str,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
):
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(f"Formatted dataset already exists: {output_path}")
        shutil.rmtree(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    if metadata is not None:
        with open(os.path.join(output_path, "metadata.json"), "w", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2)


def build_hf_dataset(
    spec: HFBuildSpec,
    *,
    dataset_root: str | Path = "data/custom_datasets",
    output_root: str | Path = "data/hf_datasets",
    overwrite: bool = False,
) -> Path:
    """Execute one validated HF build specification."""
    spec.validate()
    dataset_dict, metadata = create_formatted_dataset_dict(
        dataset_names=list(spec.datasets),
        dataset_root=str(dataset_root),
        methods=list(spec.include_methods) or None,
        run_ids=list(spec.include_runs) or None,
        include_layers=list(spec.include_layers) or None,
        composition=spec.composition,
        method_weights=dict(spec.method_weights) or None,
        samples_per_source=spec.samples_per_source,
        pair_policy=spec.pair_policy,
        reuse_limit=spec.reuse_limit,
        downsample_size=spec.downsample_size,
        heldout_ratio=spec.heldout_ratio,
        test_ratio_within_heldout=spec.test_ratio_within_heldout,
        score_names=list(spec.score_names) or None,
        score_run_ids=list(spec.score_run_ids) or None,
        seed=spec.seed,
        return_metadata=True,
    )
    metadata["hf_build_spec"] = asdict(spec)
    destination = Path(output_root) / spec.output_name
    save_formatted_dataset_dict(
        dataset_dict, str(destination), metadata=metadata, overwrite=overwrite
    )
    return destination


def load_formatted_dataset_dict(path: str) -> DatasetDict:
    dataset_dict = load_from_disk(path)
    missing = {"train", "dev", "test"} - set(dataset_dict.keys())
    if missing:
        raise ValueError(f"Saved dataset at {path} is missing split(s): {sorted(missing)}")
    return dataset_dict
