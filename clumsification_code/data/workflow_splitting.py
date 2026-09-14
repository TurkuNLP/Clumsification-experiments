# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Lightweight source-level split planning from completed workflow outputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Iterable, Mapping

from .io import read_jsonl, write_jsonl_atomic
from .schemas import CandidateRecord, OriginalRecord


WORKFLOW_METHODS = ("llm_single", "llm_sampled", "trad_single", "trad_sampled")
DEFAULT_SPLIT_ASSIGNMENT_FILENAME = "split_assignments.jsonl"


@dataclass(frozen=True)
class SplitAssignment:
    base_text_id: str
    split: str

    def to_row(self) -> dict[str, str]:
        return {"base_text_id": self.base_text_id, "split": self.split}

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "SplitAssignment":
        if set(row) != {"base_text_id", "split"}:
            raise ValueError("Split assignment rows require only base_text_id and split")
        base_text_id = row["base_text_id"]
        split = row["split"]
        if not isinstance(base_text_id, str) or not base_text_id:
            raise ValueError("Split assignment base_text_id must be a non-empty string")
        if not isinstance(split, str) or not split:
            raise ValueError("Split assignment split must be a non-empty string")
        return cls(base_text_id=base_text_id, split=split)


def _length_buckets(records: list[OriginalRecord], *, buckets: int = 10) -> dict[str, int]:
    ordered = sorted(records, key=lambda record: (len(record.text), record.base_text_id))
    return {
        record.base_text_id: min(buckets - 1, index * buckets // len(ordered))
        for index, record in enumerate(ordered)
    }


def _workflow_records(
    records: Iterable[CandidateRecord], *, method: str, source_ids: set[str]
) -> dict[str, CandidateRecord]:
    values = tuple(records)
    by_id = {record.base_text_id: record for record in values}
    if len(by_id) != len(values):
        raise ValueError(f"{method} output contains duplicate base_text_id values")
    unexpected = sorted(set(by_id) - source_ids)
    if unexpected:
        raise ValueError(
            f"{method} output contains sources not present in originals; "
            f"unexpected={len(unexpected)}"
        )
    return by_id


def _features_for_source(
    record: OriginalRecord,
    workflow_rows: Mapping[str, Mapping[str, CandidateRecord]],
    length_bucket: int,
) -> Counter[str]:
    features: Counter[str] = Counter({f"length:{length_bucket}": 1})
    for method in ("llm_single", "llm_sampled"):
        row = workflow_rows[method][record.base_text_id]
        features[f"{method}:severity:{row.severity}"] += 1
        features[f"{method}:edit_count:{row.edit_count}"] += 1
        for edit in row.perturbation_edits:
            features[f"{method}:edit:{edit}"] += 1
    for method in ("trad_single", "trad_sampled"):
        for edit in workflow_rows[method][record.base_text_id].perturbation_edits:
            features[f"{method}:edit:{edit}"] += 1
    return features


def _feature_weight(feature: str) -> float:
    return 0.25 if feature.startswith("trad_") else 1.0


def make_workflow_split_plan(
    originals: Iterable[OriginalRecord],
    workflow_outputs: Mapping[str, Iterable[CandidateRecord]],
    *,
    train_size: int = 50_000,
    dev_size: int = 5_000,
    test_size: int = 5_000,
    seed: int = 42,
) -> tuple[SplitAssignment, ...]:
    """Assign fully generated sources to exact-size splits with lightweight balancing."""
    records = list(originals)
    if not records:
        raise ValueError("Cannot split an empty source collection")
    source_ids = {record.base_text_id for record in records}
    if len(source_ids) != len(records):
        raise ValueError("Original records must have unique base_text_id values")
    if set(workflow_outputs) != set(WORKFLOW_METHODS):
        raise ValueError(f"workflow_outputs must contain exactly {WORKFLOW_METHODS}")
    rows = {
        method: _workflow_records(values, method=method, source_ids=source_ids)
        for method, values in workflow_outputs.items()
    }
    eligible_ids = set.intersection(*(set(method_rows) for method_rows in rows.values()))
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in (train_size, dev_size, test_size)
    ):
        raise ValueError("train_size, dev_size, and test_size must be positive integers")
    requested_total = train_size + dev_size + test_size
    if len(eligible_ids) < requested_total:
        raise ValueError(
            "Not enough sources have outputs from every workflow: "
            f"eligible={len(eligible_ids)}, requested={requested_total}"
        )
    records = [record for record in records if record.base_text_id in eligible_ids]
    # Allocate surplus eligible sources to an internal bucket so the retained
    # splits remain representative without writing assignments for unused IDs.
    capacities = {
        "train": train_size,
        "dev": dev_size,
        "test": test_size,
        "excluded": len(records) - requested_total,
    }
    buckets = _length_buckets(records)
    features = {
        record.base_text_id: _features_for_source(record, rows, buckets[record.base_text_id])
        for record in records
    }
    totals = Counter()
    for values in features.values():
        totals.update(values)

    def rarity(record: OriginalRecord) -> float:
        return sum(
            _feature_weight(name) / totals[name]
            for name in features[record.base_text_id]
        )

    ordering = sorted(records, key=lambda record: (-rarity(record), record.base_text_id))
    rng = random.Random(int.from_bytes(
        hashlib.blake2b(f"{seed}:workflow-split-order".encode(), digest_size=8).digest(),
        "big",
    ))
    # Randomize exact ties while preserving the rare-feature-first ordering.
    grouped: dict[float, list[OriginalRecord]] = {}
    for record in ordering:
        grouped.setdefault(rarity(record), []).append(record)
    ordering = []
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        rng.shuffle(group)
        ordering.extend(group)

    assigned = Counter()
    observed = {split: Counter() for split in capacities}
    result: list[SplitAssignment] = []
    for record in ordering:
        available = [split for split, capacity in capacities.items() if assigned[split] < capacity]
        minimum_fill = min(assigned[split] / capacities[split] for split in available)
        candidates = [
            split
            for split in available
            if assigned[split] / capacities[split] == minimum_fill
        ]

        def cost(split: str) -> tuple[float, str]:
            value = 0.0
            for name, count in features[record.base_text_id].items():
                target = totals[name] * capacities[split] / len(records)
                before = observed[split][name] - target
                after = before + count
                value += _feature_weight(name) * (after * after - before * before) / max(target, 1.0)
            return value, split

        split = min(cost(candidate) for candidate in candidates)[1]
        assigned[split] += 1
        observed[split].update(features[record.base_text_id])
        if split != "excluded":
            result.append(SplitAssignment(record.base_text_id, split))
    if dict(assigned) != {name: size for name, size in capacities.items() if size}:
        raise AssertionError("Split allocator did not satisfy its requested capacities")
    return tuple(sorted(result, key=lambda item: item.base_text_id))


def load_split_assignments(path: str | Path) -> tuple[SplitAssignment, ...]:
    assignments = tuple(SplitAssignment.from_row(row) for row in read_jsonl(path))
    ids = [assignment.base_text_id for assignment in assignments]
    if len(ids) != len(set(ids)):
        raise ValueError("Split assignment file contains duplicate base_text_id values")
    return assignments


def write_split_assignments(
    path: str | Path,
    assignments: Iterable[SplitAssignment],
    *,
    overwrite: bool = False,
) -> Path:
    rows = sorted(assignments, key=lambda assignment: assignment.base_text_id)
    return write_jsonl_atomic(path, (assignment.to_row() for assignment in rows), overwrite=overwrite)


__all__ = [
    "DEFAULT_SPLIT_ASSIGNMENT_FILENAME",
    "SplitAssignment",
    "WORKFLOW_METHODS",
    "load_split_assignments",
    "make_workflow_split_plan",
    "write_split_assignments",
]
