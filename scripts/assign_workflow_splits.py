# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Create source-level split assignments from four completed workflows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.io import write_json_atomic
from clumsification_code.data.workflow_splitting import (
    DEFAULT_SPLIT_ASSIGNMENT_FILENAME,
    WORKFLOW_METHODS,
    make_workflow_split_plan,
    write_split_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/custom_datasets"))
    for method in WORKFLOW_METHODS:
        parser.add_argument(
            f"--{method.replace('_', '-')}-run-id",
            required=True,
            help=f"Completed target-layer-one run ID for {method}.",
        )
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument(
        "--round-to",
        type=int,
        default=5_000,
        help="Round the usable source count down to this block size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DatasetRepository.from_root(args.dataset_root, args.dataset)
    outputs = {}
    for method in WORKFLOW_METHODS:
        run_id = getattr(args, f"{method}_run_id")
        outputs[method] = repository.read_candidates(repository.get_layer(method, run_id, 1))
    assignments, metadata = make_workflow_split_plan(
        repository.read_originals(),
        outputs,
        dev_size=args.dev_size,
        test_size=args.test_size,
        round_to=args.round_to,
        seed=args.seed,
        return_metadata=True,
    )
    output = args.output or repository.dataset_dir / DEFAULT_SPLIT_ASSIGNMENT_FILENAME
    metadata_output = output.with_name(f"{output.stem}.metadata.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Split assignments already exist: {output}")
    if metadata_output.exists() and not args.overwrite:
        raise FileExistsError(f"Split metadata already exists: {metadata_output}")
    write_split_assignments(output, assignments, overwrite=args.overwrite)
    write_json_atomic(metadata_output, metadata, overwrite=args.overwrite)
    print(f"Wrote {len(assignments)} split assignments: {output}")
    print(f"Wrote split audit metadata: {metadata_output}")


if __name__ == "__main__":
    main()
