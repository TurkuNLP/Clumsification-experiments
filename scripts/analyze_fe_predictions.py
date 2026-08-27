# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Generate FE predictions and optional offline diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from datasets import load_from_disk

from clumsification_code.evals.inference.fe import load_fe_inference_model
from clumsification_code.fe.diagnostics import (
    flat_pairwise_length_diagnostics,
    write_relative_length_pairwise_accuracy_plot,
)
from clumsification_code.fe.metrics import pairwise_metrics, regression_metrics


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("dataset_path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--training-method", choices=["regression", "pairwise"], required=True)
    parser.add_argument("--output", required=True, help="Output prediction JSONL path")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--length-bins", type=int, default=10)
    args = parser.parse_args()

    dataset_dict = load_from_disk(args.dataset_path)
    dataset = dataset_dict[args.split] if hasattr(dataset_dict, "keys") else dataset_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = load_fe_inference_model(args.model_dir, device=device)

    output_rows = []
    if args.training_method == "regression":
        scores = scorer.score_texts(
            list(dataset["text"]), device=device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        for row, score in zip(dataset, scores):
            output_rows.append({**row, "prediction": float(score)})
        metrics = regression_metrics(SimpleNamespace(
            predictions=scores, label_ids=dataset["target"]
        ))
    else:
        chosen_scores = scorer.score_texts(
            list(dataset["chosen_text"]), device=device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        rejected_scores = scorer.score_texts(
            list(dataset["rejected_text"]), device=device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        differences = chosen_scores - rejected_scores
        for row, chosen, rejected, difference in zip(dataset, chosen_scores, rejected_scores, differences):
            output_rows.append({**row, "chosen_score": float(chosen), "rejected_score": float(rejected), "score_difference": float(difference)})
        metrics = pairwise_metrics(SimpleNamespace(predictions=differences, label_ids=None))
        if args.diagnostics_dir:
            diagnostic_pairs = flat_pairwise_length_diagnostics(dataset, differences)
            write_relative_length_pairwise_accuracy_plot(
                args.diagnostics_dir, step=None, epoch=None,
                diagnostic_pairs=diagnostic_pairs, num_bins=args.length_bins,
            )

    _write_jsonl(args.output, output_rows)
    print(json.dumps({"num_rows": len(output_rows), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
