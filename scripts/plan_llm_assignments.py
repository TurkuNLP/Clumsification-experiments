# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Create frozen balanced assignments for the LLM perturbation workflows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.repository import DatasetRepository
from clumsification_code.perturbations.assignment_plan import (
    DEFAULT_ASSIGNMENT_FILENAME,
    plan_llm_assignment_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/custom_datasets"))
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/perturbation_prompts/english/edit_types.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DatasetRepository.from_root(args.dataset_root, args.dataset)
    output = args.output or repository.dataset_dir / DEFAULT_ASSIGNMENT_FILENAME
    assignments = plan_llm_assignment_file(
        (
            {"base_text_id": record.base_text_id, "text": record.text}
            for record in repository.read_originals()
        ),
        catalog_path=args.catalog,
        output_path=output,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(assignments)} LLM assignments: {output}")


if __name__ == "__main__":
    main()
