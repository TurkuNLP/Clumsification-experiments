# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Rank checkpoint evaluations with equal weight for each external-dev dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
import sys


METRIC_GROUPS = {
    "ELLIPSE": (
        "external_dev__ELLIPSE_train__grammar_spearman_rho",
        "external_dev__ELLIPSE_train__cohesion_spearman_rho",
    ),
    "JFLEG": (
        "external_dev__JFLEG_validation__correction_preference_strict_acc",
    ),
    "CoheSentia": (
        "external_dev__CoheSentia_train__coherence_holistic_spearman_rho",
        "external_dev__CoheSentia_train__coherence_incremental_spearman_rho",
    ),
}


def _finite_number(value, *, field: str, model_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name}: missing or invalid metric {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{model_name}: non-finite metric {field}")
    return number


def _descending_average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for model_name, _ in ordered[start:end]:
            ranks[model_name] = average_rank
        start = end
    return ranks


def rank_records(records: list[dict]) -> list[dict]:
    dataset_scores: dict[str, dict[str, float]] = {
        dataset: {} for dataset in METRIC_GROUPS
    }
    latest_by_model: dict[str, dict] = {}
    for record in records:
        if record.get("evaluation_role") != "external-dev":
            continue
        model_name = str(record.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("Every external-dev record requires model_name")
        previous = latest_by_model.get(model_name)
        if previous is None or str(record.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            latest_by_model[model_name] = record

    for model_name, record in latest_by_model.items():
        for dataset, fields in METRIC_GROUPS.items():
            dataset_scores[dataset][model_name] = mean(
                _finite_number(record.get(field), field=field, model_name=model_name)
                for field in fields
            )

    if not latest_by_model:
        raise ValueError("No external-dev result records found")
    dataset_ranks = {
        dataset: _descending_average_ranks(scores)
        for dataset, scores in dataset_scores.items()
    }
    rows = []
    for model_name in latest_by_model:
        ranks = [dataset_ranks[dataset][model_name] for dataset in METRIC_GROUPS]
        row = {"model_name": model_name}
        for dataset in METRIC_GROUPS:
            row[f"{dataset}_score"] = dataset_scores[dataset][model_name]
            row[f"{dataset}_rank"] = dataset_ranks[dataset][model_name]
        row["mean_dataset_rank"] = mean(ranks)
        row["worst_dataset_rank"] = max(ranks)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["mean_dataset_rank"],
            row["worst_dataset_rank"],
            row["model_name"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    rows = rank_records(records)
    fieldnames = list(rows[0])
    if args.output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
