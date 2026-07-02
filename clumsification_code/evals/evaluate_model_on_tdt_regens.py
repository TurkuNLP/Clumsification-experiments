from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from clumsification_code.evals.inference.base import TextScorer
from clumsification_code.evals.run_benchmark import (
    add_common_args,
    add_gptscore_args,
    add_metricx_args,
    build_scorer,
    info_ds_parser,
)


STATIC_SANITY_TEXTS = [
    "This is a perfectly fine English sentence.",
    "This no be fluent English sentence",
    (
        "This sentence can be called by the term fluent and is as such an "
        "English sentence with such a quality."
    ),
    (
        "Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin "
        "kevyt, että jokainen jaksoi kantaa sitä vuorollaan."
    ),
    (
        "Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen jaksoi "
        "kantaa vuorollaan."
    ),
    (
        "Opettaja antoi meille pitkän laudan, joka oli niin kevyt, että "
        "jokainen jaksoi kantaa sitä vuorollaan."
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────


def utc_timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def safe_filename_part(x: Any) -> str:
    value = str(x).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


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


def maybe_set_prompt_context(
    model: TextScorer,
    task_name: str,
    aspect: str,
) -> None:
    """
    GPTScore/G-Eval-style scorers can optionally expose set_prompt_context().
    LTR and MetricX simply ignore this.
    """
    setter = getattr(model, "set_prompt_context", None)
    if callable(setter):
        setter(task_name, aspect)


def scorer_model_path(args: argparse.Namespace) -> str:
    return (
        getattr(args, "model_dir", "")
        or getattr(args, "hf_model_name_or_path", "")
        or getattr(args, "metricx_model_name_or_path", "")
        or ""
    )


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def extract_run_id(path: Path) -> Optional[int]:
    match = re.search(r"_regens_(\d+)\.jsonl$", path.name)
    return int(match.group(1)) if match else None


def make_split_name(
    generator_model: str,
    effort: str,
    group_by: str,
    source_path: Path,
) -> str:
    if group_by == "file":
        return source_path.name.removesuffix(".jsonl")

    if group_by == "model_effort":
        return f"{safe_filename_part(generator_model)}__{safe_filename_part(effort)}"

    raise ValueError(f"Unknown group_by value: {group_by!r}")


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
            generator_family = parse_model_family(generator_model)
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
                    "generator_family": generator_family,
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
    duplicate_id_policy: str = "first",
) -> pd.DataFrame:
    """
    Keep only IDs present in every regeneration split.

    Duplicates are resolved before scoring so each split scores exactly one text
    per common ID.

    duplicate_id_policy:
      - "first": keep the first row per duplicate ID inside each split
      - "mean": backward-compatible alias for "first" before scoring
      - "error": fail on duplicate IDs
    """
    valid_policies = {"first", "mean", "error"}

    if duplicate_id_policy not in valid_policies:
        raise ValueError(
            f"duplicate_id_policy must be one of {sorted(valid_policies)}, "
            f"got {duplicate_id_policy!r}"
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
                    f"Split {split_name!r} contains duplicate IDs before scoring: "
                    f"{n_dup_ids} IDs / {n_dup_rows} rows."
                )

            print(
                f"Warning: split {split_name!r} contains duplicate IDs "
                f"({n_dup_ids} IDs / {n_dup_rows} rows). Keeping first occurrence."
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
        split_df = split_df.set_index("id", drop=False).loc[common_ids].reset_index(
            drop=True
        )

        if split_df["id"].tolist() != common_ids:
            raise RuntimeError(f"Internal alignment error for split {split_name!r}.")

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


def discover_eval_folders(
    base_dir: Path,
    languages: Optional[List[str]] = None,
) -> List[Path]:
    """
    Accept either:
      1. a base dir containing language subfolders, or
      2. a single eval/language dir containing *_regens_*.jsonl files.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_dir}")

    allowed = set(languages or [])

    if list(base_dir.glob("*_regens_*.jsonl")):
        if allowed and base_dir.name not in allowed:
            return []
        return [base_dir]

    eval_folders: List[Path] = []

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue

        if child.name == "results":
            continue

        if allowed and child.name not in allowed:
            continue

        if list(child.glob("*_regens_*.jsonl")):
            eval_folders.append(child)
        else:
            print(f"Skipping {child}: no *_regens_*.jsonl files found.")

    if not eval_folders:
        raise FileNotFoundError(
            f"No eval folders with *_regens_*.jsonl files found in {base_dir}"
        )

    return eval_folders


# ──────────────────────────────────────────────────────────────────────
# Scoring and metrics
# ──────────────────────────────────────────────────────────────────────

def parse_model_family(model_name: str) -> str:
    name = clean_text(model_name).lower()
    # Expected formats: gpt-5.4-mini, gemini-3-flash-preview
    if name.startswith("gpt"):
        return "gpt"
    if name.startswith("gemini"):
        return "gemini"

    # Fallback if prefixes are not exact
    m = re.match(r"^(gpt|gemini)\b", name)
    if m:
        return m.group(1)

    return "other"


def compute_model_summary_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (generator_family, generator_model) score stats and mean-based ranking
    inside each family.
    """
    grouped = scored_df.groupby(["generator_family", "generator_model"], sort=True)

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

    summary["rank_by_mean"] = (
        summary.groupby(level=0)["mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return summary[
        ["n", "mean", "std", "sem", "median", "min", "max", "rank_by_mean"]
    ].sort_values(["generator_family", "rank_by_mean", "generator_model"])


def build_family_model_score_maps(
    scored_df: pd.DataFrame,
    duplicate_id_policy: str = "first",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build nested dict:
      family -> model -> {id -> score}
    Duplicates within (family, model, id) are handled by policy.
    """
    valid_policies = {"first", "mean", "error"}

    if duplicate_id_policy not in valid_policies:
        raise ValueError(
            f"duplicate_id_policy must be one of {sorted(valid_policies)}, "
            f"got {duplicate_id_policy!r}"
        )

    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for (family, model_name), model_df in scored_df.groupby(
        ["generator_family", "generator_model"], sort=True
    ):
        tmp = model_df[["id", "score"]].copy()
        tmp["id"] = tmp["id"].astype(str)

        dup_mask = tmp.duplicated(subset=["id"], keep=False)
        if dup_mask.any():
            n_dup_rows = int(dup_mask.sum())
            n_dup_ids = int(tmp.loc[dup_mask, "id"].nunique())

            if duplicate_id_policy == "error":
                raise ValueError(
                    f"Duplicate IDs for family/model ({family!r}, {model_name!r}): "
                    f"{n_dup_ids} IDs / {n_dup_rows} rows."
                )

            if duplicate_id_policy == "mean":
                tmp = tmp.groupby("id", as_index=False)["score"].mean()
            else:
                # "first" (also backward-compatible behavior)
                print(
                    f"Warning: duplicate IDs for family/model ({family!r}, {model_name!r}) "
                    f"({n_dup_ids} IDs / {n_dup_rows} rows). Keeping first occurrence."
                )
                tmp = tmp.drop_duplicates(subset=["id"], keep="first")

        score_map = {
            str(row["id"]): float(row["score"])
            for _, row in tmp.iterrows()
            if math.isfinite(float(row["score"]))
        }

        out.setdefault(str(family), {})[str(model_name)] = score_map

    return out


def compute_pairwise_winrates_by_family(
    family_model_scores: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    rows = []

    for family, model_scores in sorted(family_model_scores.items()):
        models = sorted(model_scores.keys())

        if len(models) < 2:
            continue

        for model_a, model_b in combinations(models, 2):
            arr_a, arr_b, common_ids = align_scores(
                model_scores[model_a], model_scores[model_b]
            )
            n = len(common_ids)

            if n == 0:
                rows.append(
                    {
                        "generator_family": family,
                        "model_a": model_a,
                        "model_b": model_b,
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
                    "generator_family": family,
                    "model_a": model_a,
                    "model_b": model_b,
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


def compute_pairwise_model_ranking(
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models within each family using average tie-aware pairwise win rate
    against other models in the same family.
    """
    if pairwise_df.empty:
        return pd.DataFrame(
            columns=[
                "generator_family",
                "generator_model",
                "pairwise_matchups",
                "avg_tie_aware_winrate_vs_others",
                "avg_strict_winrate_vs_others",
                "rank_by_pairwise_winrate",
            ]
        )

    rows = []

    for family, fam_df in pairwise_df.groupby("generator_family", sort=True):
        per_model = []

        models = sorted(set(fam_df["model_a"]).union(set(fam_df["model_b"])))

        for model_name in models:
            as_a = fam_df[fam_df["model_a"] == model_name][
                ["tie_aware_win_rate_a", "win_rate_a"]
            ].rename(
                columns={
                    "tie_aware_win_rate_a": "tie_aware_wr",
                    "win_rate_a": "strict_wr",
                }
            )
            as_b = fam_df[fam_df["model_b"] == model_name][
                ["tie_aware_win_rate_b", "win_rate_b"]
            ].rename(
                columns={
                    "tie_aware_win_rate_b": "tie_aware_wr",
                    "win_rate_b": "strict_wr",
                }
            )

            joined = pd.concat([as_a, as_b], ignore_index=True)

            per_model.append(
                {
                    "generator_family": family,
                    "generator_model": model_name,
                    "pairwise_matchups": int(len(joined)),
                    "avg_tie_aware_winrate_vs_others": float(
                        joined["tie_aware_wr"].mean()
                    )
                    if len(joined)
                    else np.nan,
                    "avg_strict_winrate_vs_others": float(
                        joined["strict_wr"].mean()
                    )
                    if len(joined)
                    else np.nan,
                }
            )

        fam_rank_df = pd.DataFrame(per_model)
        fam_rank_df["rank_by_pairwise_winrate"] = (
            fam_rank_df["avg_tie_aware_winrate_vs_others"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        rows.append(
            fam_rank_df.sort_values(
                ["rank_by_pairwise_winrate", "generator_model"]
            )
        )

    return pd.concat(rows, ignore_index=True)


def build_pairwise_matrix_by_family(
    family_model_scores: Dict[str, Dict[str, Dict[str, float]]],
    value: str,
) -> Dict[str, pd.DataFrame]:
    valid_values = {"strict_winrate", "tie_aware_winrate", "mean_delta"}

    if value not in valid_values:
        raise ValueError(f"value must be one of {sorted(valid_values)}, got {value}")

    out: Dict[str, pd.DataFrame] = {}

    for family, model_scores in sorted(family_model_scores.items()):
        models = sorted(model_scores)
        matrix = pd.DataFrame(np.nan, index=models, columns=models)

        for model_a, model_b in combinations(models, 2):
            arr_a, arr_b, _ = align_scores(model_scores[model_a], model_scores[model_b])
            n = len(arr_a)

            if n == 0:
                continue

            deltas = arr_a - arr_b
            wins_a = float((deltas > 0).sum())
            wins_b = float((deltas < 0).sum())
            ties = float((deltas == 0).sum())

            if value == "strict_winrate":
                matrix.loc[model_a, model_b] = wins_a / n
                matrix.loc[model_b, model_a] = wins_b / n
            elif value == "tie_aware_winrate":
                matrix.loc[model_a, model_b] = (wins_a + 0.5 * ties) / n
                matrix.loc[model_b, model_a] = (wins_b + 0.5 * ties) / n
            else:
                mean_delta = float(np.mean(deltas))
                matrix.loc[model_a, model_b] = mean_delta
                matrix.loc[model_b, model_a] = -mean_delta

        out[family] = matrix

    return out

def score_dataframe(
    df: pd.DataFrame,
    model: TextScorer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    task_name: str,
    aspect: str,
    lower_is_better: bool = False,
) -> pd.DataFrame:
    maybe_set_prompt_context(model, task_name=task_name, aspect=aspect)

    parts = []
    score_multiplier = -1.0 if lower_is_better else 1.0

    for split_name, split_df in df.groupby("split", sort=True):
        split_df = split_df.copy()

        print(f"\n>>> Scoring split {split_name!r} ({len(split_df)} texts)")

        scores = model.score_texts(
            texts=split_df["text"].tolist(),
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        scores = np.asarray(scores, dtype=np.float64)

        if len(scores) != len(split_df):
            raise RuntimeError(
                f"Scorer returned {len(scores)} scores for {len(split_df)} texts "
                f"in split {split_name!r}."
            )

        split_df["score"] = scores * score_multiplier
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
            (
                (deltas > 0).astype(float)
                + 0.5 * (deltas == 0).astype(float)
            ).tolist()
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


def append_compact_jsonl(
    log_dir: Path,
    run_name: str,
    record: Dict[str, Any],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = log_dir / f"{safe_filename_part(run_name)}.jsonl"

    with out_path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(json_sanitize(record), ensure_ascii=False) + "\n")

    return out_path


def build_compact_record(
    args: argparse.Namespace,
    run_output_dir: Path,
    scored_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    pairwise_rank_df: pd.DataFrame,
) -> Dict[str, Any]:
    model_summary_rows = model_summary_df.reset_index().to_dict("records")
    pairwise_rows = pairwise_df.to_dict("records")
    pairwise_rank_rows = pairwise_rank_df.to_dict("records")

    best_by_mean: Dict[str, Optional[str]] = {}
    best_by_pairwise: Dict[str, Optional[str]] = {}

    if not model_summary_df.empty:
        tmp = model_summary_df.reset_index()
        for fam, fam_df in tmp.groupby("generator_family", sort=True):
            fam_df = fam_df.sort_values(["rank_by_mean", "generator_model"])
            best_by_mean[str(fam)] = (
                str(fam_df.iloc[0]["generator_model"]) if len(fam_df) else None
            )

    if not pairwise_rank_df.empty:
        for fam, fam_df in pairwise_rank_df.groupby("generator_family", sort=True):
            fam_df = fam_df.sort_values(
                ["rank_by_pairwise_winrate", "generator_model"]
            )
            best_by_pairwise[str(fam)] = (
                str(fam_df.iloc[0]["generator_model"]) if len(fam_df) else None
            )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_type": "tdt_regen_familywise",
        "scorer": args.scorer,
        "model_name_arg": args.model_name,
        "evaluator_model_name": args.evaluator_model_name,
        "evaluator_model_path": scorer_model_path(args),
        "training_language": args.training_language,
        "training_dataset": args.training_dataset,
        "perturbation_type": args.perturbation_type,
        "num_layers": args.num_layers,
        "base_dir": str(args.base_dir),
        "data_folder": str(args.resolved_data_folder),
        "language": args.language,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "context_length": args.context_length,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "score_task_name": args.score_task_name,
        "score_aspect": args.score_aspect,
        "score_direction": "lower_is_better_inverted"
        if args.lower_is_better
        else "higher_is_better",
        "run_output_dir": str(run_output_dir),
        "n_scored_texts": int(len(scored_df)),
        "n_families": int(scored_df["generator_family"].nunique()),
        "n_models": int(scored_df["generator_model"].nunique()),
        "best_model_by_mean_per_family": best_by_mean,
        "best_model_by_pairwise_winrate_per_family": best_by_pairwise,
        # Easy to consume later in notebooks:
        "model_summary_rows": model_summary_rows,
        "pairwise_winrates_rows": pairwise_rows,
        "pairwise_ranking_rows": pairwise_rank_rows,
    }

    return record


def run_evaluation(
    model: TextScorer,
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
        task_name=args.score_task_name,
        aspect=args.score_aspect,
        lower_is_better=args.lower_is_better,
    )

    model_summary_df = compute_model_summary_scores(scored_df)

    family_model_scores = build_family_model_score_maps(
        scored_df,
        duplicate_id_policy=args.duplicate_id_policy,
    )
    pairwise_df = compute_pairwise_winrates_by_family(family_model_scores)
    pairwise_rank_df = compute_pairwise_model_ranking(pairwise_df)

    strict_mats = build_pairwise_matrix_by_family(
        family_model_scores, value="strict_winrate"
    )
    tie_aware_mats = build_pairwise_matrix_by_family(
        family_model_scores, value="tie_aware_winrate"
    )
    mean_delta_mats = build_pairwise_matrix_by_family(
        family_model_scores, value="mean_delta"
    )

    print("\n" + "=" * 70)
    print("PER-MODEL SCORE SUMMARY (within families)")
    print("=" * 70)
    print(model_summary_df.to_string())

    print("\n" + "=" * 70)
    print("PAIRWISE WIN RATES (within families only)")
    print("=" * 70)
    if len(pairwise_df):
        print(pairwise_df.to_string(index=False))
    else:
        print("No within-family model pairs available.")

    print("\n" + "=" * 70)
    print("PAIRWISE-BASED MODEL RANKING (within families)")
    print("=" * 70)
    if len(pairwise_rank_df):
        print(pairwise_rank_df.to_string(index=False))
    else:
        print("No pairwise ranking available.")

    run_output_dir.mkdir(parents=True, exist_ok=True)

    scored_df.to_json(
        run_output_dir / "raw_scores.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    model_summary_df.to_csv(run_output_dir / "model_summary_scores.csv")
    pairwise_df.to_csv(run_output_dir / "pairwise_winrates_by_family.csv", index=False)
    pairwise_rank_df.to_csv(
        run_output_dir / "pairwise_ranking_by_family.csv", index=False
    )

    for family, mat in strict_mats.items():
        mat.to_csv(run_output_dir / f"strict_winrate_matrix_{safe_filename_part(family)}.csv")
    for family, mat in tie_aware_mats.items():
        mat.to_csv(run_output_dir / f"tie_aware_winrate_matrix_{safe_filename_part(family)}.csv")
    for family, mat in mean_delta_mats.items():
        mat.to_csv(run_output_dir / f"mean_delta_matrix_{safe_filename_part(family)}.csv")

    compact_record = build_compact_record(
        args=args,
        run_output_dir=run_output_dir,
        scored_df=scored_df,
        model_summary_df=model_summary_df,
        pairwise_df=pairwise_df,
        pairwise_rank_df=pairwise_rank_df,
    )

    with (run_output_dir / "compact_summary.json").open("w", encoding="utf-8") as writer:
        json.dump(json_sanitize(compact_record), writer, ensure_ascii=False, indent=2)

    # Keep this behavior: one JSONL per evaluator/scorer combo
    jsonl_path = append_compact_jsonl(
        log_dir=run_output_dir.parent,
        run_name=f"{args.scorer}-{args.model_name}",
        record=compact_record,
    )

    print(f"\n✓ Saved run outputs to: {run_output_dir}")
    print(f"✓ Appended compact JSONL record to: {jsonl_path}")

    return {
        "scored_df": scored_df,
        "model_summary_df": model_summary_df,
        "pairwise_df": pairwise_df,
        "pairwise_rank_df": pairwise_rank_df,
        "strict_mats": strict_mats,
        "tie_aware_mats": tie_aware_mats,
        "mean_delta_mats": mean_delta_mats,
        "compact_record": compact_record,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a QE scorer on TDT/UD regenerated texts. The scorer "
            "backend is shared with run_benchmark.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared benchmark/scorer args:
    add_common_args(parser)
    add_gptscore_args(parser)
    add_metricx_args(parser)

    # Optional G-Eval args if available in this checkout.
    try:
        from clumsification_code.evals.geval.cli import add_geval_args

        add_geval_args(parser)
    except Exception:
        pass

    # TDT regeneration specific args:
    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Base directory containing language subfolders, e.g. "
            "data/benchmarks/ud_regens, or a single folder containing "
            "*_regens_*.jsonl files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output root. By default, outputs are written under "
            "<eval_folder>/results/."
        ),
    )
    parser.add_argument(
        "--languages",
        default="",
        help=(
            "Optional comma-separated language/eval folder names to run, e.g. "
            "fi,en. Empty means all discovered folders."
        ),
    )
    parser.add_argument(
        "--group-by",
        default="file",
        choices=["file", "model_effort"],
        help="How to define comparison splits from regeneration JSONL files.",
    )
    parser.add_argument(
        "--duplicate-id-policy",
        default="first",
        choices=["first", "mean", "error"],
        help=(
            "How to handle duplicate IDs inside a split before scoring. "
            "'mean' is kept as a backward-compatible alias for 'first'."
        ),
    )
    parser.add_argument(
        "--expected-order",
        default="",
        help=(
            "Optional comma-separated split names from expected best to worst. "
            "Used only for extra diagnostics."
        ),
    )
    parser.add_argument(
        "--score-task-name",
        default="tdt_regen",
        help="Prompt task name for prompt-aware scorers such as GPTScore/G-Eval.",
    )
    parser.add_argument(
        "--score-aspect",
        default="quality",
        help="Prompt aspect for prompt-aware scorers such as GPTScore/G-Eval.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help=(
            "Use if a custom scorer returns lower-is-better values. Scores are "
            "multiplied by -1 so all downstream metrics remain higher-is-better. "
            "Do not use this for the current MetricX24 adapter because it already "
            "returns higher-is-better by default."
        ),
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip scoring the static sanity-check texts before the eval.",
    )

    return parser.parse_args(argv)


def attach_runtime_metadata(
    args: argparse.Namespace,
    device: torch.device,
) -> argparse.Namespace:
    args.base_dir = Path(args.base_dir)

    if getattr(args, "model_dir", ""):
        args.model_dir = str(Path(args.model_dir))

    if args.context_length < 0:
        args.context_length = args.max_length

    args.evaluator_model_name = args.model_name
    args.training_language = ""

    if args.scorer == "ltr":
        try:
            parsed_model_name, parsed_lan, pert_type, num_layers, training_ds_name = (
                info_ds_parser(args.model_name)
            )

            args.evaluator_model_name = parsed_model_name
            args.training_language = parsed_lan

            if not args.perturbation_type:
                args.perturbation_type = pert_type

            if args.num_layers < 0:
                try:
                    args.num_layers = int(num_layers)
                except Exception:
                    args.num_layers = -1

            if not args.training_dataset:
                args.training_dataset = training_ds_name

        except Exception as exc:
            print(
                f"Warning: could not parse LTR model name {args.model_name!r} "
                f"with info_ds_parser: {exc}"
            )

    else:
        if not args.training_dataset:
            args.training_dataset = "none"

        if not args.perturbation_type:
            args.perturbation_type = "none"

        if args.num_layers < 0:
            args.num_layers = 0

    # The default shared args use flash_attention_2. Avoid surprising CPU failure.
    if device.type == "cpu" and args.attn_implementation == "flash_attention_2":
        args.attn_implementation = "eager"

    # Same practical safeguard for local torch models on CPU.
    if device.type == "cpu" and args.dtype in {"bfloat16", "bf16", "float16", "fp16"}:
        print(
            f"CPU detected with dtype={args.dtype!r}; overriding to float32 "
            "for safer local inference."
        )
        args.dtype = "float32"

    return args


def run_static_sanity_check(
    model: TextScorer,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    maybe_set_prompt_context(
        model,
        task_name=args.score_task_name,
        aspect=args.score_aspect,
    )

    print("Running static sanity check texts:")

    for i, text in enumerate(STATIC_SANITY_TEXTS, start=1):
        print(f"  [{i}] {text}")

    scores = model.score_texts(
        STATIC_SANITY_TEXTS,
        device=device,
        batch_size=min(3, args.batch_size),
        max_length=min(args.max_length, 128),
    )

    scores = np.asarray(scores, dtype=np.float64)

    if args.lower_is_better:
        scores = -scores

    print("Static sanity check scores:", scores)
    print()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = attach_runtime_metadata(args, device=device)

    print(f"Device          : {device}")
    print(f"Scorer          : {args.scorer}")
    print(f"Model name arg  : {args.model_name}")
    print(f"Model path      : {scorer_model_path(args)}")
    print(f"Parsed/eval name: {args.evaluator_model_name}")
    print(f"Training lang   : {args.training_language}")
    print(f"Training dataset: {args.training_dataset}")
    print(f"Perturbation    : {args.perturbation_type}")
    print(f"Num layers      : {args.num_layers}")
    print(f"Max length      : {args.max_length}")
    print(f"Batch size      : {args.batch_size}")
    print(f"dtype           : {args.dtype}")
    print(f"Base dir        : {args.base_dir}")
    print(f"Prompt context  : task={args.score_task_name!r}, aspect={args.score_aspect!r}")
    print()

    languages = [
        x.strip()
        for x in args.languages.split(",")
        if x.strip()
    ]

    eval_folders = discover_eval_folders(args.base_dir, languages=languages or None)

    print("Evaluation folders:")

    for folder in eval_folders:
        print(f"  - {folder}")

    print()

    model = build_scorer(args, device)

    if not args.skip_sanity_check:
        run_static_sanity_check(model=model, device=device, args=args)

    safe_run_name = safe_filename_part(f"{args.scorer}-{args.model_name}")

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

        print(
            f"Loaded {len(data_df)} regenerated texts across "
            f"{data_df['generator_family'].nunique()} families and "
            f"{data_df['generator_model'].nunique()} models."
        )

        for fam, fam_df in data_df.groupby("generator_family", sort=True):
            print(
                f"  Family={fam}: {fam_df['generator_model'].nunique()} models, "
                f"{len(fam_df)} texts"
            )

        if args.output_dir:
            results_dir = Path(args.output_dir) / language
        else:
            results_dir = eval_folder / "results"

        run_output_dir = results_dir / f"{safe_run_name}_{utc_timestamp_for_filename()}"

        run_evaluation(
            model=model,
            data_df=data_df,
            device=device,
            args=run_args,
            run_output_dir=run_output_dir,
        )


if __name__ == "__main__":
    main()