# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Generate one method-separated perturbation layer for a custom dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.perturbations import (
    generate_layer,
    list_method_specs,
    load_source_items,
)


def _load_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Method configuration must be a JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/custom_datasets"))
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--source-method", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument(
        "--method",
        required=True,
        choices=[spec.name for spec in list_method_specs()],
    )
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--language", default="english")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help=(
            "Absolute context ceiling. Text bucketing and generation limits "
            "are derived automatically by the LLM runner."
        ),
    )
    parser.add_argument("--method-config", default=None)
    parser.add_argument(
        "--assignment-file",
        type=Path,
        default=None,
        help=(
            "Frozen LLM assignment JSONL created by plan_llm_assignments.py. "
            "When supplied, its matching method rows replace runtime sampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source-partitions",
        nargs="+",
        default=None,
        help="Generate only sources assigned to these canonical partitions.",
    )
    parser.add_argument("--allow-unchanged", action="store_true", default=None)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Deprecated compatibility option; must be 0. Use --retry-failed on a later submission.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Number of source items committed after each generation batch "
            "within each context bucket (default: 512 for LLM methods)."
        ),
    )
    parser.add_argument(
        "--max-output-char-tolerance",
        type=int,
        default=None,
        help=(
            "Additional characters allowed above the derived LLM output limit "
            "(default: 256)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry only sources explicitly recorded as failed in the existing layer, "
            "then merge recovered candidates into that layer."
        ),
    )
    args = parser.parse_args()
    if args.source_layer < 0:
        parser.error("--source-layer must be non-negative")
    if args.overwrite and args.retry_failed:
        parser.error("--overwrite and --retry-failed cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    config = _load_json(args.method_config)
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be a positive integer")
    config.update(
        {
            key: value
            for key, value in {
                "language": args.language,
                "model": args.model_path,
                "max_model_len": args.max_model_len,
                "seed": args.seed,
                "n_jobs": args.n_jobs,
                "allow_unchanged": args.allow_unchanged,
                "max_retries": args.max_retries,
                "batch_size": args.batch_size,
                "max_output_char_tolerance": args.max_output_char_tolerance,
                "assignment_file": (
                    str(args.assignment_file) if args.assignment_file is not None else None
                ),
            }.items()
            if value is not None
        }
    )
    output = generate_layer(
        args.dataset,
        dataset_root=args.dataset_root,
        source_layer=args.source_layer,
        source_method=args.source_method,
        source_run_id=args.source_run_id,
        method=args.method,
        run_id=args.run_id,
        target_layer=args.target_layer,
        config=config,
        source_partitions=(
            tuple(args.source_partitions) if args.source_partitions is not None else None
        ),
        limit=args.limit,
        overwrite=args.overwrite,
        retry_failed=args.retry_failed,
    )
    print(f"Wrote perturbation layer: {output}")


if __name__ == "__main__":
    main()


__all__ = ["generate_layer", "load_source_items", "main", "parse_args"]
