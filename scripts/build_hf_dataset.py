# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Build one canonical Hugging Face dataset from custom-dataset repositories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.hf_dataset import build_hf_dataset
from clumsification_code.data.schemas import (
    COMPOSITION_POLICIES,
    HFBuildSpec,
    PAIR_POLICIES,
)


def _parse_weights(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    result = {}
    for value in values:
        try:
            method, weight = value.split("=", 1)
            result[method] = float(weight)
        except ValueError as exc:
            raise ValueError(
                f"Method weight must have METHOD=FLOAT form: {value!r}"
            ) from exc
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON file containing an HFBuildSpec object")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-name")
    parser.add_argument("--dataset-root", default="data/custom_datasets")
    parser.add_argument("--output-root", default="data/hf_datasets")
    parser.add_argument("--include-methods", nargs="+")
    parser.add_argument("--include-runs", nargs="+")
    parser.add_argument("--include-layers", nargs="+", type=int)
    parser.add_argument("--composition", choices=sorted(COMPOSITION_POLICIES), default="all")
    parser.add_argument("--method-weights", nargs="+", metavar="METHOD=WEIGHT")
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--pair-policy", choices=sorted(PAIR_POLICIES), default="none")
    parser.add_argument("--reuse-limit", type=int, default=5)
    parser.add_argument("--score-names", nargs="+")
    parser.add_argument("--score-run-ids", nargs="+")
    parser.add_argument("--downsample-size", type=int)
    parser.add_argument("--heldout-ratio", type=float, default=0.3)
    parser.add_argument("--test-ratio-within-heldout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _spec_from_args(args: argparse.Namespace) -> HFBuildSpec:
    if args.config:
        conflicting = [
            name for name in ("datasets", "output_name", "include_methods", "include_runs", "include_layers")
            if getattr(args, name) is not None
        ]
        if conflicting:
            raise ValueError(f"--config cannot be combined with: {', '.join('--' + name.replace('_', '-') for name in conflicting)}")
        with Path(args.config).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("HF configuration must be a JSON object")
        return HFBuildSpec.from_dict(value)
    if not args.datasets or not args.output_name:
        raise ValueError("--datasets and --output-name are required without --config")
    return HFBuildSpec.from_dict({
        "output_name": args.output_name,
        "datasets": args.datasets,
        "include_methods": args.include_methods or [],
        "include_runs": args.include_runs or [],
        "include_layers": args.include_layers or [],
        "composition": args.composition,
        "method_weights": _parse_weights(args.method_weights) or {},
        "samples_per_source": args.samples_per_source,
        "pair_policy": args.pair_policy,
        "reuse_limit": args.reuse_limit,
        "downsample_size": args.downsample_size,
        "heldout_ratio": args.heldout_ratio,
        "test_ratio_within_heldout": args.test_ratio_within_heldout,
        "score_names": args.score_names or [],
        "score_run_ids": args.score_run_ids or [],
        "seed": args.seed,
    })


def main() -> None:
    args = parse_args()
    destination = build_hf_dataset(
        _spec_from_args(args), dataset_root=args.dataset_root,
        output_root=args.output_root, overwrite=args.overwrite,
    )
    print(f"Saved HF dataset: {destination}")


if __name__ == "__main__":
    main()
