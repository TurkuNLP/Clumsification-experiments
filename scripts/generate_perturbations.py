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
            "Absolute maximum rendered LLM request length in tokens. "
            "Bucketing and over-limit filtering are applied by the LLM runner."
        ),
    )
    parser.add_argument(
        "--bucket-policy",
        choices=["legacy_text_length", "rendered_prompt"],
        default=None,
        help="LLM context bucketing policy; legacy_text_length preserves historical behavior.",
    )
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-noise", type=int, default=None)
    parser.add_argument("--n-edits", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--operation", default=None)
    parser.add_argument("--operations", nargs="+", default=None)
    parser.add_argument("--target-dimensions", nargs="+", default=None)
    parser.add_argument("--severity", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-unchanged", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_layer < 0:
        parser.error("--source-layer must be non-negative")
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
                "bucket_policy": args.bucket_policy,
                "seed": args.seed,
                "n_noise": args.n_noise,
                "n_edits": args.n_edits,
                "n_jobs": args.n_jobs,
                "operation": args.operation,
                "operations": args.operations,
                "target_dimensions": args.target_dimensions,
                "severity": args.severity,
                "allow_unchanged": args.allow_unchanged,
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
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote perturbation layer: {output}")


if __name__ == "__main__":
    main()


__all__ = ["generate_layer", "load_source_items", "main", "parse_args"]
