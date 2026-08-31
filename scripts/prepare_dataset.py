# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Run canonical perturbation generation and Hugging Face dataset workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.hf_dataset import build_hf_dataset
from clumsification_code.data.schemas import HFBuildSpec, WorkflowConfig
from clumsification_code.perturbations import generate_layer, get_method_spec


def preset_config(name: str, dataset: str) -> dict[str, Any]:
    presets = {
        "zero_shot_ablation": {
            "dataset": dataset,
            "generations": [{"method": "llm_zero_shot", "source_layer": 0}],
            "hf": {
                "output_name": f"{dataset}_zero_shot", "include_layers": [1],
                "include_methods": ["llm_zero_shot"],
            },
        },
        "sampled_llm_ablation": {
            "dataset": dataset,
            "generations": [{"method": "llm_sampled", "source_layer": 0}],
            "hf": {
                "output_name": f"{dataset}_sampled_llm", "include_layers": [1],
                "include_methods": ["llm_sampled"],
            },
        },
        "traditional_comparison": {
            "dataset": dataset,
            "generations": [
                {"method": method, "source_layer": 0}
                for method in ("unieval", "unieval_summinflect", "trad_single", "trad_multi")
            ],
            "hf": {
                "output_name": f"{dataset}_traditional", "include_layers": [1],
                "include_methods": [
                    "unieval", "unieval_summinflect", "trad_single", "trad_multi"
                ],
            },
        },
    }
    try:
        return presets[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown workflow preset {name!r}; choose one of: {', '.join(sorted(presets))}"
        ) from exc


def load_workflow_config(
    path: str | None, *, preset: str | None, dataset: str | None
) -> WorkflowConfig:
    if path is not None and preset is not None:
        raise ValueError("Specify either --config or --preset, not both")
    if path is None and preset is None:
        raise ValueError("One of --config or --preset is required")
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Workflow configuration must be a JSON object")
    else:
        if not dataset:
            raise ValueError("--dataset is required when using --preset")
        value = preset_config(str(preset), dataset)
    if dataset is not None:
        value["dataset"] = dataset
    config = WorkflowConfig.from_dict(value)
    for generation in config.generations:
        get_method_spec(generation.method)
    return config


def run_generations(config: WorkflowConfig, *, overwrite: bool = False) -> list[Path]:
    outputs = []
    for generation in config.generations:
        outputs.append(generate_layer(
            config.dataset,
            dataset_root=config.dataset_root,
            source_layer=generation.source_layer,
            source_method=generation.source_method,
            source_run_id=generation.source_run_id,
            method=generation.method,
            run_id=generation.run_id,
            target_layer=generation.target_layer,
            config=dict(generation.config),
            limit=generation.limit,
            overwrite=overwrite,
        ))
    return outputs


def build_hf(
    config: WorkflowConfig,
    *,
    output_root: str = "data/hf_datasets",
    overwrite: bool = False,
) -> Path:
    if config.hf is None:
        raise ValueError("Workflow configuration has no 'hf' section")
    spec: HFBuildSpec = config.hf
    return build_hf_dataset(
        spec, dataset_root=config.dataset_root, output_root=output_root,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "build-hf", "run-all"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument(
        "--preset",
        choices=["zero_shot_ablation", "sampled_llm_ablation", "traditional_comparison"],
    )
    parser.add_argument("--dataset")
    parser.add_argument("--output-root", default="data/hf_datasets")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_workflow_config(args.config, preset=args.preset, dataset=args.dataset)
    if args.command in {"generate", "run-all"}:
        for path in run_generations(config, overwrite=args.overwrite):
            print(f"Generated: {path}")
    if args.command in {"build-hf", "run-all"}:
        print(f"Built: {build_hf(config, output_root=args.output_root, overwrite=args.overwrite)}")


if __name__ == "__main__":
    main()
