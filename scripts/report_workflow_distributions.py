# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Print compact source and perturbation distributions for completed workflows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.workflow_splitting import WORKFLOW_METHODS


def _percentile(values: Iterable[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/custom_datasets"))
    for method in WORKFLOW_METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DatasetRepository.from_root(args.dataset_root, args.dataset)
    split_by_id = repository.read_split_assignments()
    if split_by_id is None:
        raise FileNotFoundError("reporting requires split_assignments.jsonl")
    originals = {record.base_text_id: record for record in repository.read_originals()}
    workflow_rows = {}
    for method in WORKFLOW_METHODS:
        run_id = getattr(args, f"{method}_run_id")
        rows = repository.read_candidates(repository.get_layer(method, run_id, 1))
        by_id = {row.base_text_id: row for row in rows}
        if set(by_id) != set(originals):
            raise ValueError(f"{method} does not cover the same source IDs as original.jsonl")
        workflow_rows[method] = by_id

    report = {}
    for split in ("train", "dev", "test"):
        source_ids = sorted(base_text_id for base_text_id, value in split_by_id.items() if value == split)
        lengths = [len(originals[base_text_id].text) for base_text_id in source_ids]
        workflows = {}
        for method, by_id in workflow_rows.items():
            rows = [by_id[base_text_id] for base_text_id in source_ids]
            workflows[method] = {
                "edit_counts": _counts(row.edit_count for row in rows),
                "edit_types": _counts(edit for row in rows for edit in row.perturbation_edits),
                "severities": _counts(row.severity for row in rows if row.severity is not None),
                "target_dimensions": _counts(
                    dimension for row in rows for dimension in row.target_dimensions
                ),
            }
        report[split] = {
            "source_count": len(source_ids),
            "source_length_chars": {
                "mean": sum(lengths) / len(lengths),
                "p10": _percentile(lengths, 10),
                "p90": _percentile(lengths, 90),
            },
            "workflows": workflows,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
