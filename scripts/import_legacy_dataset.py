# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Import one legacy perturbation directory into the canonical dataset store."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data import DatasetRepository
from clumsification_code.data.legacy_import import LegacyLayoutImporter
from clumsification_code.perturbations import list_method_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Custom dataset directory name.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/custom_datasets"),
        help="Directory containing custom datasets.",
    )
    parser.add_argument(
        "--source-directory",
        required=True,
        help="Legacy directory relative to the dataset (for example, perturbed_layers).",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=[spec.name for spec in list_method_specs()],
        help="Canonical method represented by every imported layer.",
    )
    parser.add_argument(
        "--run-id",
        default="legacy-import",
        help="Canonical run identifier assigned to the imported layers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DatasetRepository.from_root(args.dataset_root, args.dataset)
    importer = LegacyLayoutImporter(repository)
    entries = importer.import_directory(
        args.source_directory,
        method=args.method,
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    for entry in entries:
        print(repository.dataset_dir / entry.path)


if __name__ == "__main__":
    main()
