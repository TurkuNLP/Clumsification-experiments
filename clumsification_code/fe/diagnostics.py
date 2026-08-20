# This script has been co-created, refactored, and cleaned using GPT 5.6.
import csv
import json
import os
import shutil
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def diagnostic_dir(output_dir: Optional[str]) -> str:
    root = output_dir or "."
    diag_dir = os.path.join(root, "length_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    return diag_dir


def safe_float_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int_metric(metrics: Dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def write_pairwise_accuracy_history_plot(
    output_dir: Optional[str],
    step: Optional[int],
    epoch: Optional[float],
    metrics: Dict[str, Any],
) -> None:
    """
    Appends one evaluation record and updates a persistent training-history plot.

    Produces:
        length_diagnostics/pairwise_accuracy_history.jsonl
        length_diagnostics/pairwise_accuracy_history.png
    """
    diag_dir = diagnostic_dir(output_dir)

    history_path = os.path.join(diag_dir, "pairwise_accuracy_history.jsonl")
    plot_path = os.path.join(diag_dir, "pairwise_accuracy_history.png")

    record = {
        "step": int(step) if step is not None else None,
        "epoch": float(epoch) if epoch is not None else None,

        "pairwise_accuracy": safe_float_metric(metrics, "pairwise_accuracy"),
        "pairwise_accuracy_when_shorter_is_better": safe_float_metric(
            metrics, "pairwise_accuracy_when_shorter_is_better"
        ),
        "pairwise_accuracy_when_longer_is_better": safe_float_metric(
            metrics, "pairwise_accuracy_when_longer_is_better"
        ),

        "correct_points": safe_float_metric(metrics, "correct_points"),
        "correct_pairs": safe_float_metric(metrics, "correct_pairs"),

        "strict_correct_pairs": safe_int_metric(metrics, "strict_correct_pairs"),
        "score_tie_pairs": safe_int_metric(metrics, "score_tie_pairs"),
        "total_pairs": safe_int_metric(metrics, "total_pairs"),
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    if all(r.get("step") is not None for r in records):
        x = [r["step"] for r in records]
        xlabel = "Global step"
    else:
        x = list(range(1, len(records) + 1))
        xlabel = "Evaluation number"

    series = [
        ("pairwise_accuracy", "Overall pairwise accuracy"),
        ("pairwise_accuracy_when_shorter_is_better", "Pairwise accuracy when shorter is better"),
        ("pairwise_accuracy_when_longer_is_better", "Pairwise accuracy when longer is better"),
    ]

    plt.figure(figsize=(9, 5))

    for key, label in series:
        y = [r.get(key) for r in records]
        valid = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]

        if not valid:
            continue

        xx, yy = zip(*valid)
        plt.plot(xx, yy, marker="o", linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Pairwise accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def build_relative_length_quantile_rows(
    diagnostic_pairs: List[tuple],
    num_bins: int,
) -> List[Dict[str, Any]]:
    """
    Builds data-derived quantile buckets.

    diagnostic_pairs contains:
        (relative_length_difference, point_value)

    point_value is:
        1.0 if the model strictly ranked the pair correctly.
        0.5 if the model assigned exactly tied scores.
        0.0 if the model strictly ranked the pair incorrectly.
    """
    if not diagnostic_pairs:
        return []

    diagnostic_pairs = sorted(diagnostic_pairs, key=lambda x: x[0])

    n = len(diagnostic_pairs)
    k = max(1, min(int(num_bins), n))

    rows = []

    for b in range(k):
        start = b * n // k
        end = (b + 1) * n // k

        chunk = diagnostic_pairs[start:end]

        if not chunk:
            continue

        rels = [float(x[0]) for x in chunk]
        points = [float(x[1]) for x in chunk]

        total = len(chunk)
        correct_points = sum(points)

        rows.append(
            {
                "bin_id": b,
                "rel_len_diff_min": min(rels),
                "rel_len_diff_max": max(rels),
                "rel_len_diff_mean": sum(rels) / total,
                "pairwise_accuracy": correct_points / total if total > 0 else 0.0,
                "total_pairs": total,
                "correct_points": correct_points,
                "correct_pairs": correct_points,
            }
        )

    return rows


def write_relative_length_pairwise_accuracy_plot(
    output_dir: Optional[str],
    step: Optional[int],
    epoch: Optional[float],
    diagnostic_pairs: List[tuple],
    num_bins: int = 10,
) -> None:
    """
    Writes a plot of pairwise accuracy versus relative length difference.

    Relative length difference:
        abs(len_i - len_j) / max(len_i, len_j)

    Produces:
        length_diagnostics/relative_length_pairwise_accuracy_step_<STEP>.png
        length_diagnostics/relative_length_pairwise_accuracy_latest.png
        length_diagnostics/relative_length_pairwise_accuracy_step_<STEP>.csv
    """
    if not diagnostic_pairs:
        return

    diag_dir = diagnostic_dir(output_dir)

    step_str = str(step) if step is not None else "unknown"

    rows = build_relative_length_quantile_rows(
        diagnostic_pairs=diagnostic_pairs,
        num_bins=num_bins,
    )

    if not rows:
        return

    csv_path = os.path.join(
        diag_dir,
        f"relative_length_pairwise_accuracy_step_{step_str}.csv",
    )

    png_path = os.path.join(
        diag_dir,
        f"relative_length_pairwise_accuracy_step_{step_str}.png",
    )

    latest_png_path = os.path.join(
        diag_dir,
        "relative_length_pairwise_accuracy_latest.png",
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_id",
                "rel_len_diff_min",
                "rel_len_diff_max",
                "rel_len_diff_mean",
                "pairwise_accuracy",
                "total_pairs",
                "correct_points",
                "correct_pairs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    x = [row["rel_len_diff_mean"] for row in rows]
    y = [row["pairwise_accuracy"] for row in rows]
    counts = [row["total_pairs"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=2)

    for xx, yy, count in zip(x, y, counts):
        plt.annotate(
            str(count),
            xy=(xx, yy),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.75,
        )

    title = "Pairwise accuracy by relative length difference"
    if epoch is not None:
        title += f" | epoch={float(epoch):.3f}"
    if step is not None:
        title += f" | step={step}"

    plt.title(title)
    plt.xlabel("Relative length difference: abs(len_i - len_j) / max(len_i, len_j)")
    plt.ylabel("Pairwise accuracy")
    plt.ylim(0.0, 1.0)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    shutil.copyfile(png_path, latest_png_path)
