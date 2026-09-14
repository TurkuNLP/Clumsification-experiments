"""Export split-selected originals and completed layer-one workflows to a new dataset."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.candidate_identity import (
    make_candidate_id,
    make_original_candidate_id,
)
from clumsification_code.data.io import write_json_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import PerturbationManifest
from clumsification_code.data.workflow_splitting import (
    WORKFLOW_METHODS,
    SplitAssignment,
    write_split_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--destination-dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/custom_datasets"))
    for method in WORKFLOW_METHODS:
        parser.add_argument(
            f"--{method.replace('_', '-')}-run-id",
            required=True,
            help=f"Layer-one run ID to export for {method}.",
        )
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument(
        "--copy-scores",
        action="store_true",
        help="Copy completed scores for exported candidates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_assignments(
    assignments: dict[str, str], *, train_size: int, dev_size: int, test_size: int
) -> set[str]:
    expected = {"train": train_size, "dev": dev_size, "test": test_size}
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in expected.values()):
        raise ValueError("Split sizes must be positive integers")
    counts = {split: sum(value == split for value in assignments.values()) for split in expected}
    if counts != expected:
        raise ValueError(f"Split assignments do not have the requested sizes: {counts}")
    unknown = set(assignments.values()) - set(expected)
    if unknown:
        raise ValueError(f"Split assignments contain unsupported splits: {sorted(unknown)}")
    return set(assignments)


def export_final_dataset(
    source: DatasetRepository,
    destination: DatasetRepository,
    *,
    run_ids: dict[str, str],
    train_size: int = 50_000,
    dev_size: int = 5_000,
    test_size: int = 5_000,
    copy_scores: bool = False,
    overwrite: bool = False,
) -> None:
    if source.dataset_name == destination.dataset_name:
        raise ValueError("Source and destination datasets must be different")
    assignments = source.read_split_assignments()
    if assignments is None:
        raise FileNotFoundError("Source dataset requires split_assignments.jsonl")
    selected_ids = _validate_assignments(
        assignments, train_size=train_size, dev_size=dev_size, test_size=test_size
    )
    originals = [record for record in source.read_originals() if record.base_text_id in selected_ids]
    if len(originals) != len(selected_ids):
        raise ValueError("Split assignments reference originals missing from the source dataset")

    entries = {
        method: source.get_layer(method, run_ids[method], 1)
        for method in WORKFLOW_METHODS
    }
    source_records = {
        method: [record for record in source.read_candidates(entry) if record.base_text_id in selected_ids]
        for method, entry in entries.items()
    }
    incomplete = {
        method: len(records)
        for method, records in source_records.items()
        if len(records) != len(selected_ids)
        or {record.base_text_id for record in records} != selected_ids
    }
    if incomplete:
        raise ValueError(f"Selected sources are incomplete in exported layers: {incomplete}")

    if destination.dataset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination.dataset_dir}; use --overwrite to replace canonical files"
        )
    destination.dataset_dir.mkdir(parents=True, exist_ok=True)
    destination_originals = [replace(record, dataset_name=destination.dataset_name) for record in originals]
    destination.write_originals(destination_originals, overwrite=overwrite)
    write_split_assignments(
        destination.split_assignments_path,
        (SplitAssignment(base_text_id, assignments[base_text_id]) for base_text_id in selected_ids),
        overwrite=overwrite,
    )

    # Start a fresh canonical manifest. Existing unreferenced files, if any,
    # are harmless; only files registered here are visible to workflows.
    write_json_atomic(
        destination.manifest_path,
        PerturbationManifest(dataset_name=destination.dataset_name).to_dict(),
        overwrite=True,
    )
    candidate_ids = {
        make_original_candidate_id(dataset_name=source.dataset_name, base_text_id=record.base_text_id):
        make_original_candidate_id(dataset_name=destination.dataset_name, base_text_id=record.base_text_id)
        for record in originals
    }
    for method in WORKFLOW_METHODS:
        entry = entries[method]
        exported = []
        for record in source_records[method]:
            parent_candidate_id = candidate_ids.get(record.parent_candidate_id)
            if parent_candidate_id is None:
                raise ValueError(
                    f"Candidate {record.candidate_id!r} has a parent outside the selected originals"
                )
            candidate_id = make_candidate_id(
                dataset_name=destination.dataset_name,
                perturbation_method=record.perturbation_method,
                run_id=record.run_id,
                base_text_id=record.base_text_id,
                target_layer=record.target_layer,
                parent_candidate_id=parent_candidate_id,
                candidate_index=record.candidate_index,
            )
            exported.append(replace(
                record,
                dataset_name=destination.dataset_name,
                candidate_id=candidate_id,
                parent_candidate_id=parent_candidate_id,
            ))
            candidate_ids[record.candidate_id] = candidate_id
        destination.write_candidate_layer(
            exported,
            method=entry.method,
            run_id=entry.run_id,
            target_layer=entry.target_layer,
            source_layer=entry.source_layer,
            source_method=entry.source_method,
            source_run_id=entry.source_run_id,
            config=entry.config,
            input_count=len(selected_ids),
            overwrite=overwrite,
        )
    destination.validate_lineage()
    if copy_scores:
        _copy_scores(source, destination, candidate_ids, overwrite=overwrite)


def _copy_scores(
    source: DatasetRepository,
    destination: DatasetRepository,
    candidate_ids: dict[str, str],
    *,
    overwrite: bool,
) -> None:
    grouped = {}
    for record in source.read_scores():
        candidate_id = candidate_ids.get(record.candidate_id)
        if candidate_id is None:
            continue
        reference_candidate_id = (
            None if record.reference_candidate_id is None
            else candidate_ids.get(record.reference_candidate_id)
        )
        if record.reference_candidate_id is not None and reference_candidate_id is None:
            raise ValueError(
                f"Score references a candidate outside the export: {record.reference_candidate_id!r}"
            )
        copied = replace(
            record,
            dataset_name=destination.dataset_name,
            candidate_id=candidate_id,
            reference_candidate_id=reference_candidate_id,
        )
        grouped.setdefault((record.scoring_method, record.scoring_run_id), []).append(copied)
    for (scoring_method, scoring_run_id), records in grouped.items():
        destination.write_scores(
            records,
            scoring_method=scoring_method,
            scoring_run_id=scoring_run_id,
            metadata={
                "schema_version": 3,
                "dataset_name": destination.dataset_name,
                "scoring_method": scoring_method,
                "scoring_run_id": scoring_run_id,
                "migrated_from_dataset": source.dataset_name,
                "num_successful_scores": len(records),
            },
            overwrite=overwrite,
        )


def main() -> None:
    args = parse_args()
    source = DatasetRepository.from_root(args.dataset_root, args.source_dataset)
    destination = DatasetRepository.from_root(args.dataset_root, args.destination_dataset)
    run_ids = {method: getattr(args, f"{method}_run_id") for method in WORKFLOW_METHODS}
    export_final_dataset(
        source,
        destination,
        run_ids=run_ids,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        copy_scores=args.copy_scores,
        overwrite=args.overwrite,
    )
    print(f"Exported {args.train_size + args.dev_size + args.test_size} sources to {destination.dataset_dir}")


if __name__ == "__main__":
    main()
