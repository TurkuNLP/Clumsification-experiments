from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

from clumsification_code.evals.inference.ltr import (
    LTRInferenceModel,
    load_ltr_inference_model,
)
from clumsification_code.evals.result_writer import json_sanitize
from clumsification_code.evals.run_benchmark import info_ds_parser

STATIC_SANITY_TEXTS = [
    "This is a perfectly fine English sentence.",
    "This no be fluent English sentence",
    "This sentence can be called by the term fluent and is as such an English sentence with such a quality.",
    "Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan.",
    "Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen jaksoi kantaa vuorollaan.",
    "Opettaja antoi meille pitkän laudan, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan.",
]

EVAL_LOG_DIR = Path("data/evals/ud_regen")

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────


def clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_run_id(path: Path) -> Optional[int]:
    match = re.search(r"_regens_(\d+)\.jsonl$", path.name)
    return int(match.group(1)) if match else None


def safe_split_part(x: Any) -> str:
    return re.sub(r"\s+", "-", str(x).strip())


def make_split_name(
    generator_model: str,
    effort: str,
    group_by: str,
    source_path: Path,
) -> str:
    if group_by == "file":
        return source_path.name.removesuffix(".jsonl")

    if group_by == "model_effort":
        return f"{safe_split_part(generator_model)}__{safe_split_part(effort)}"

    raise ValueError(f"Unknown group_by value: {group_by}")


def load_regen_rows(
    data_folder: Path,
    language: Optional[str] = None,
    group_by: str = "file",
) -> pd.DataFrame:
    """
    Load regenerated documents only.

    ud_data.jsonl is intentionally ignored. Original human texts are not read,
    scored, aligned, or compared.
    """
    data_folder = Path(data_folder)

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    regen_paths = sorted(data_folder.glob("*_regens_*.jsonl"))
    if not regen_paths:
        raise FileNotFoundError(f"No *_regens_*.jsonl files found in {data_folder}")

    inferred_language = language or data_folder.name
    rows: List[Dict[str, Any]] = []
    row_order = 0

    for regen_path in regen_paths:
        run_id = extract_run_id(regen_path)

        for item in read_jsonl(regen_path):
            text = clean_text(item.get("text"))
            uid = item.get("id")

            if uid is None or not text:
                continue

            generator_model = str(item.get("model", "unknown_model"))
            effort = str(item.get("effort", "unknown_effort"))

            rows.append(
                {
                    "split": make_split_name(
                        generator_model=generator_model,
                        effort=effort,
                        group_by=group_by,
                        source_path=regen_path,
                    ),
                    "kind": "generated",
                    "id": str(uid),
                    "text": text,
                    "generator_model": generator_model,
                    "effort": effort,
                    "language": str(item.get("language", inferred_language)),
                    "source_file": str(regen_path),
                    "run_id": run_id,
                    "_row_order": row_order,
                }
            )
            row_order += 1

    if not rows:
        raise ValueError(f"No valid regenerated rows found in {data_folder}")

    return pd.DataFrame(rows)


def restrict_to_common_regen_ids(
    df: pd.DataFrame,
    duplicate_id_policy: str = "mean",
) -> pd.DataFrame:
    """
    Keep only IDs present in every regeneration split.

    Duplicates are resolved before scoring so each split scores exactly one
    text per common ID. For pre-scoring duplicates, both "mean" and "first"
    keep the first row. Use "error" to fail instead.
    """
    valid_policies = {"mean", "first", "error"}
    if duplicate_id_policy not in valid_policies:
        raise ValueError(
            f"duplicate_id_policy must be one of {sorted(valid_policies)}, "
            f"got {duplicate_id_policy}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["split"] = df["split"].astype(str)

    split_names = sorted(df["split"].unique())
    if len(split_names) < 2:
        raise ValueError(
            "Need at least two regeneration splits to compare. "
            f"Found: {split_names}"
        )

    deduped_parts = []

    for split_name in split_names:
        split_df = df[df["split"] == split_name].sort_values("_row_order").copy()
        duplicated = split_df.duplicated(subset=["id"], keep=False)

        if duplicated.any():
            n_dup_rows = int(duplicated.sum())
            n_dup_ids = int(split_df.loc[duplicated, "id"].nunique())

            if duplicate_id_policy == "error":
                raise ValueError(
                    f"Split '{split_name}' contains duplicate IDs before scoring: "
                    f"{n_dup_ids} IDs / {n_dup_rows} rows."
                )

            split_df = split_df.drop_duplicates(subset=["id"], keep="first")

        deduped_parts.append(split_df)

    df = pd.concat(deduped_parts, ignore_index=True)

    split_to_ids = {
        split_name: set(df.loc[df["split"] == split_name, "id"])
        for split_name in split_names
    }

    reference_split = (
        df.sort_values("_row_order")
        .groupby("split", sort=False)
        .head(1)
        .sort_values("_row_order")
        .iloc[0]["split"]
    )

    reference_ids = (
        df[df["split"] == reference_split]
        .sort_values("_row_order")["id"]
        .tolist()
    )

    common_ids = [
        uid
        for uid in reference_ids
        if all(uid in split_to_ids[split_name] for split_name in split_names)
    ]

    if not common_ids:
        raise ValueError("No common regeneration IDs remain after intersection.")

    ordered_parts = []

    for split_name in split_names:
        split_df = df[df["split"] == split_name].copy()
        split_df = split_df[split_df["id"].isin(common_ids)]
        split_df = split_df.set_index("id", drop=False).loc[common_ids].reset_index(drop=True)

        if split_df["id"].tolist() != common_ids:
            raise RuntimeError(f"Internal alignment error for split '{split_name}'.")

        ordered_parts.append(split_df)

    aligned_df = pd.concat(ordered_parts, ignore_index=True)

    counts = aligned_df.groupby("split")["id"].agg(["count", "nunique"])
    expected_n = len(common_ids)

    if not (
        counts["count"].nunique() == 1
        and counts["nunique"].nunique() == 1
        and int(counts["count"].iloc[0]) == expected_n
    ):
        raise RuntimeError("Internal alignment error after common-ID filtering.")

    return aligned_df.drop(columns=["_row_order"], errors="ignore")


# ──────────────────────────────────────────────────────────────────────
# Scoring and metrics
# ──────────────────────────────────────────────────────────────────────


def score_dataframe(
    df: pd.DataFrame,
    model: LTRInferenceModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    parts = []

    for split_name, split_df in df.groupby("split", sort=True):
        split_df = split_df.copy()

        print(f"\n>>> Scoring split '{split_name}' ({len(split_df)} texts)")

        split_df["score"] = model.score_texts(
            texts=split_df["text"].tolist(),
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        ).astype(float)

        parts.append(split_df)

    return pd.concat(parts, ignore_index=True)


def scores_by_split_and_id(scored_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        split_name: {
            str(row["id"]): float(row["score"])
            for _, row in split_df.iterrows()
            if math.isfinite(float(row["score"]))
        }
        for split_name, split_df in scored_df.groupby("split", sort=True)
    }


def compute_summary_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    grouped = scored_df.groupby("split", sort=True)

    summary = grouped["score"].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))

    for col in ["kind", "generator_model", "effort", "language"]:
        summary[col] = grouped[col].agg(
            lambda s: "|".join(sorted({str(x) for x in s.dropna()}))
        )

    summary["rank_by_mean"] = summary["mean"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    return summary[
        [
            "kind",
            "generator_model",
            "effort",
            "language",
            "n",
            "mean",
            "std",
            "sem",
            "median",
            "min",
            "max",
            "rank_by_mean",
        ]
    ].sort_values(["rank_by_mean", "split"])


def align_scores(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    common_ids = [uid for uid in scores_a if uid in scores_b]
    arr_a = np.asarray([scores_a[uid] for uid in common_ids], dtype=np.float64)
    arr_b = np.asarray([scores_b[uid] for uid in common_ids], dtype=np.float64)
    return arr_a, arr_b, common_ids


def compute_pairwise_winrates(scored: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []

    for split_a, split_b in combinations(sorted(scored), 2):
        arr_a, arr_b, common_ids = align_scores(scored[split_a], scored[split_b])
        n = len(common_ids)

        if n == 0:
            rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "n_shared": 0,
                    "wins_a": 0,
                    "wins_b": 0,
                    "ties": 0,
                    "win_rate_a": np.nan,
                    "win_rate_b": np.nan,
                    "tie_rate": np.nan,
                    "tie_aware_win_rate_a": np.nan,
                    "tie_aware_win_rate_b": np.nan,
                    "mean_delta_a_minus_b": np.nan,
                    "median_delta_a_minus_b": np.nan,
                    "binom_p": np.nan,
                }
            )
            continue

        deltas = arr_a - arr_b
        wins_a = int((deltas > 0).sum())
        wins_b = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())
        decisive = wins_a + wins_b

        rows.append(
            {
                "split_a": split_a,
                "split_b": split_b,
                "n_shared": n,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "win_rate_a": wins_a / n,
                "win_rate_b": wins_b / n,
                "tie_rate": ties / n,
                "tie_aware_win_rate_a": (wins_a + 0.5 * ties) / n,
                "tie_aware_win_rate_b": (wins_b + 0.5 * ties) / n,
                "mean_delta_a_minus_b": float(np.mean(deltas)),
                "median_delta_a_minus_b": float(np.median(deltas)),
                "binom_p": (
                    float(scipy_stats.binomtest(wins_a, decisive, 0.5).pvalue)
                    if decisive
                    else 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_pairwise_matrix(
    scored: Dict[str, Dict[str, float]],
    value: str,
) -> pd.DataFrame:
    valid_values = {"strict_winrate", "tie_aware_winrate", "mean_delta"}
    if value not in valid_values:
        raise ValueError(f"value must be one of {sorted(valid_values)}, got {value}")

    splits = sorted(scored)
    matrix = pd.DataFrame(np.nan, index=splits, columns=splits)

    for split_a, split_b in combinations(splits, 2):
        arr_a, arr_b, _ = align_scores(scored[split_a], scored[split_b])
        n = len(arr_a)

        if n == 0:
            continue

        deltas = arr_a - arr_b
        wins_a = float((deltas > 0).sum())
        wins_b = float((deltas < 0).sum())
        ties = float((deltas == 0).sum())

        if value == "strict_winrate":
            matrix.loc[split_a, split_b] = wins_a / n
            matrix.loc[split_b, split_a] = wins_b / n

        elif value == "tie_aware_winrate":
            matrix.loc[split_a, split_b] = (wins_a + 0.5 * ties) / n
            matrix.loc[split_b, split_a] = (wins_b + 0.5 * ties) / n

        else:
            mean_delta = float(np.mean(deltas))
            matrix.loc[split_a, split_b] = mean_delta
            matrix.loc[split_b, split_a] = -mean_delta

    return matrix


def compute_expected_order_metrics(
    expected_order: List[str],
    scored: Dict[str, Dict[str, float]],
    summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    expected_order = [x for x in expected_order if x]
    present = [x for x in expected_order if x in scored]
    missing = [x for x in expected_order if x not in scored]

    if len(present) < 2:
        return {
            "expected_order": expected_order,
            "expected_order_present": present,
            "expected_order_missing": missing,
            "expected_order_n_present": len(present),
            "expected_order_spearman_r": np.nan,
            "expected_order_spearman_p": np.nan,
            "expected_order_kendall_tau": np.nan,
            "expected_order_kendall_p": np.nan,
            "expected_order_pairwise_tie_aware_acc": np.nan,
            "expected_order_pairwise_strict_acc": np.nan,
            "expected_order_n_pairwise_split_comparisons": 0,
        }

    expected_strength = np.asarray(range(len(present), 0, -1), dtype=np.float64)
    observed_means = np.asarray(
        [summary_df.loc[split_name, "mean"] for split_name in present],
        dtype=np.float64,
    )

    if len(present) >= 3:
        spearman_r, spearman_p = scipy_stats.spearmanr(
            expected_strength,
            observed_means,
        )
    else:
        spearman_r, spearman_p = np.nan, np.nan

    kendall_tau, kendall_p = scipy_stats.kendalltau(
        expected_strength,
        observed_means,
    )

    strict_credits = []
    tie_aware_credits = []
    n_split_comparisons = 0

    for better_idx, worse_idx in combinations(range(len(present)), 2):
        better = present[better_idx]
        worse = present[worse_idx]

        arr_better, arr_worse, common_ids = align_scores(scored[better], scored[worse])
        if not common_ids:
            continue

        deltas = arr_better - arr_worse

        strict_credits.extend((deltas > 0).astype(float).tolist())
        tie_aware_credits.extend(
            ((deltas > 0).astype(float) + 0.5 * (deltas == 0).astype(float)).tolist()
        )
        n_split_comparisons += 1

    return {
        "expected_order": expected_order,
        "expected_order_present": present,
        "expected_order_missing": missing,
        "expected_order_n_present": len(present),
        "expected_order_spearman_r": float(spearman_r),
        "expected_order_spearman_p": float(spearman_p),
        "expected_order_kendall_tau": float(kendall_tau),
        "expected_order_kendall_p": float(kendall_p),
        "expected_order_pairwise_tie_aware_acc": (
            float(np.mean(tie_aware_credits)) if tie_aware_credits else np.nan
        ),
        "expected_order_pairwise_strict_acc": (
            float(np.mean(strict_credits)) if strict_credits else np.nan
        ),
        "expected_order_n_pairwise_split_comparisons": n_split_comparisons,
    }


# ──────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────


def utc_timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return json_sanitize(float(x))

    if isinstance(x, np.bool_):
        return bool(x)

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    if not isinstance(x, (str, bytes)):
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

    return x


def append_compact_jsonl(log_dir: Path, model_name: str, record: Dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")
    out_path = log_dir / f"{safe_model_name}.jsonl"

    with out_path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(json_sanitize(record), ensure_ascii=False) + "\n")

    return out_path


def build_compact_record(
    args: argparse.Namespace,
    run_output_dir: Path,
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    expected_order_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    summary_sorted = summary_df.sort_values("rank_by_mean", ascending=True)
    best_split = str(summary_sorted.index[0]) if len(summary_sorted) else None

    split_mean_scores = {
        str(split): float(row["mean"])
        for split, row in summary_df.iterrows()
    }

    split_ranks_by_mean = {
        str(split): int(row["rank_by_mean"])
        for split, row in summary_df.iterrows()
    }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_type": "ud_regen_only",

        # Raw user-provided model identifier.
        "model_name_arg": args.model_name,

        # Parsed from info_ds_parser(args.model_name).
        "evaluator_model_name": args.parsed_model_name,
        "training_language": args.parsed_training_language,
        "training_dataset": args.training_dataset,
        "perturbation_type": args.perturbation_type,
        "num_layers": args.num_layers,

        # Runtime/config.
        "evaluator_model_dir": str(args.model_dir),
        "base_dir": str(args.base_dir),
        "data_folder": str(args.resolved_data_folder),
        "language": args.language,
        "group_by": args.group_by,
        "duplicate_id_policy": args.duplicate_id_policy,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "context_length": args.context_length,
        "max_seq_len": args.max_seq_len,
        "dtype": str(args.dtype).replace("torch.", ""),
        "attn_implementation": args.attn_implementation,

        # Outputs/results.
        "run_output_dir": str(run_output_dir),
        "n_scored_texts": int(len(scored_df)),
        "n_splits": int(len(summary_df)),
        "best_split_by_mean": best_split,
        "split_mean_scores": split_mean_scores,
        "split_ranks_by_mean": split_ranks_by_mean,
    }

    record.update(expected_order_metrics)
    return record


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────


def run_evaluation(
    model: LTRInferenceModel,
    data_df: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
    run_output_dir: Path,
) -> Dict[str, Any]:
    scored_df = score_dataframe(
        df=data_df,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    summary_df = compute_summary_scores(scored_df)
    scored = scores_by_split_and_id(scored_df)

    pairwise_df = compute_pairwise_winrates(scored)
    strict_matrix = build_pairwise_matrix(scored, "strict_winrate")
    tie_aware_matrix = build_pairwise_matrix(scored, "tie_aware_winrate")
    mean_delta_matrix = build_pairwise_matrix(scored, "mean_delta")

    expected_order_metrics = {}
    if args.expected_order:
        expected_order_metrics = compute_expected_order_metrics(
            expected_order=[
                x.strip()
                for x in args.expected_order.split(",")
                if x.strip()
            ],
            scored=scored,
            summary_df=summary_df,
        )

    print("\n" + "=" * 70)
    print("PER-SPLIT AVERAGE QUALITY SCORES")
    print("=" * 70)
    print(summary_df.to_string())

    print("\n" + "=" * 70)
    print("PAIRWISE WIN RATES")
    print("=" * 70)
    print(pairwise_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TIE-AWARE WIN-RATE MATRIX — row beats column")
    print("=" * 70)
    print(tie_aware_matrix.to_string())

    if expected_order_metrics:
        print("\n" + "=" * 70)
        print("EXPECTED ORDER METRICS")
        print("=" * 70)
        for key, value in expected_order_metrics.items():
            print(f"  {key}: {value}")

    run_output_dir.mkdir(parents=True, exist_ok=True)

    scored_df.to_json(
        run_output_dir / "raw_scores.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    summary_df.to_csv(run_output_dir / "summary_scores.csv")
    pairwise_df.to_csv(run_output_dir / "pairwise_winrates.csv", index=False)
    strict_matrix.to_csv(run_output_dir / "strict_winrate_matrix.csv")
    tie_aware_matrix.to_csv(run_output_dir / "tie_aware_winrate_matrix.csv")
    mean_delta_matrix.to_csv(run_output_dir / "mean_delta_matrix.csv")

    compact_record = build_compact_record(
        args=args,
        run_output_dir=run_output_dir,
        scored_df=scored_df,
        summary_df=summary_df,
        expected_order_metrics=expected_order_metrics,
    )

    with (run_output_dir / "compact_summary.json").open("w", encoding="utf-8") as writer:
        json.dump(json_sanitize(compact_record), writer, ensure_ascii=False, indent=2)

    jsonl_path = append_compact_jsonl(
        log_dir=run_output_dir.parent,
        model_name=args.model_name,
        record=compact_record,
    )

    print(f"\n✓ Saved run outputs to: {run_output_dir}")
    print(f"✓ Appended compact JSONL record to: {jsonl_path}")

    return {
        "scored_df": scored_df,
        "summary_df": summary_df,
        "pairwise_df": pairwise_df,
        "strict_matrix": strict_matrix,
        "tie_aware_matrix": tie_aware_matrix,
        "mean_delta_matrix": mean_delta_matrix,
        "expected_order_metrics": expected_order_metrics,
        "compact_record": compact_record,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a QE/LTR model on every UD regeneration subfolder inside "
            "a base directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Base directory containing language subfolders, e.g. "
            "data/benchmarks/ud_regens/ with subfolders en, es, fi."
        ),
    )

    parser.add_argument(
        "--model-dir",
        "--model",
        dest="model_dir",
        required=True,
        help="Path to final trainer directory containing ltr_head.pt.",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Model/dataset-style name parsed by info_ds_parser, e.g. "
            "QE0.6B_fi_some_training_dataset_5."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        required=True,
        help="Used both as scoring max_length and logged context/max sequence length.",
    )

    return parser.parse_args()

def attach_default_runtime_config(args: argparse.Namespace, device: torch.device) -> argparse.Namespace:
    parsed_model_name, parsed_lan, pert_type, num_layers, training_ds_name = info_ds_parser(
        args.model_name
    )

    try:
        num_layers = int(num_layers)
    except Exception:
        num_layers = -1

    args.parsed_model_name = parsed_model_name
    args.parsed_training_language = parsed_lan
    args.perturbation_type = pert_type
    args.num_layers = num_layers
    args.training_dataset = training_ds_name

    # The user's max length is used for both fields.
    args.context_length = args.max_length
    args.max_seq_len = args.max_length

    # Internal defaults that are no longer user-facing CLI args.
    args.group_by = "file"
    args.duplicate_id_policy = "mean"
    args.batch_size = 32
    args.expected_order = ""

    # Safer default on CPU. You can force bfloat16 here if your loader requires it.
    args.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # If flash_attention_2 is unavailable in your environment, use "eager".
    args.attn_implementation = "flash_attention_2" if device.type == "cuda" else "eager"

    return args

def discover_eval_folders(base_dir: Path) -> List[Path]:
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_dir}")

    eval_folders = []

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue

        # Skip output folders if someone points base-dir too low.
        if child.name == "results":
            continue

        if list(child.glob("*_regens_*.jsonl")):
            eval_folders.append(child)
        else:
            print(f"Skipping {child}: no *_regens_*.jsonl files found.")

    if not eval_folders:
        raise FileNotFoundError(
            f"No language/eval subfolders with *_regens_*.jsonl files found in {base_dir}"
        )

    return eval_folders


def main() -> None:
    args = parse_args()

    args.base_dir = Path(args.base_dir)
    args.model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = attach_default_runtime_config(args, device=device)

    print(f"Device          : {device}")
    print(f"Model dir       : {args.model_dir}")
    print(f"Model name arg  : {args.model_name}")
    print(f"Parsed model    : {args.parsed_model_name}")
    print(f"Training lang   : {args.parsed_training_language}")
    print(f"Training dataset: {args.training_dataset}")
    print(f"Perturbation    : {args.perturbation_type}")
    print(f"Num layers      : {args.num_layers}")
    print(f"Max length      : {args.max_length}")
    print(f"dtype           : {args.dtype}")
    print(f"Base dir        : {args.base_dir}")
    print()

    eval_folders = discover_eval_folders(args.base_dir)

    print("Evaluation folders:")
    for folder in eval_folders:
        print(f"  - {folder}")
    print()

    model = load_ltr_inference_model(
        model_dir=args.model_dir,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    print("Running static sanity check texts:")
    for i, text in enumerate(STATIC_SANITY_TEXTS, start=1):
        print(f"  [{i}] {text}")

    sanity_scores = model.score_texts(
        STATIC_SANITY_TEXTS,
        device=device,
        batch_size=min(3, args.batch_size),
        max_length=min(args.max_length, 128),
    )

    print("Static sanity check scores:", sanity_scores)
    print()

    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model_name).strip("-")

    for eval_folder in eval_folders:
        language = eval_folder.name

        print("\n" + "#" * 80)
        print(f"Evaluating folder: {eval_folder}")
        print(f"Language         : {language}")
        print("#" * 80)

        run_args = argparse.Namespace(**vars(args))
        run_args.language = language
        run_args.resolved_data_folder = str(eval_folder)

        data_df = load_regen_rows(
            data_folder=eval_folder,
            language=language,
            group_by=run_args.group_by,
        )

        print(
            f"Loaded {len(data_df)} regenerated texts across "
            f"{data_df['split'].nunique()} raw splits."
        )

        data_df = restrict_to_common_regen_ids(
            data_df,
            duplicate_id_policy=run_args.duplicate_id_policy,
        )

        print(
            f"After filtering to common regeneration IDs: "
            f"{len(data_df)} texts across {data_df['split'].nunique()} splits."
        )
        print("Every split now has the same IDs in the same reference order:")

        for split_name, split_df in data_df.groupby("split", sort=True):
            print(
                f"  {split_name}: "
                f"{len(split_df)} rows, {split_df['id'].nunique()} unique IDs"
            )

        results_dir = eval_folder / "results"
        run_output_dir = results_dir / f"{safe_model_name}_{utc_timestamp_for_filename()}"

        run_evaluation(
            model=model,
            data_df=data_df,
            device=device,
            args=run_args,
            run_output_dir=run_output_dir,
        )


if __name__ == "__main__":
    main()