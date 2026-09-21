# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _finite_pair_mask(labels, preds):
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    mask = np.isfinite(labels) & np.isfinite(preds)
    return labels[mask], preds[mask]


def safe_spearman(labels, preds, name: str = "metric"):
    labels, preds = _finite_pair_mask(labels, preds)

    if len(labels) < 2:
        print(f"{name}: not enough valid points for Spearman.", flush=True)
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"{name}: predictions are constant; Spearman undefined.", flush=True)
        return float("nan"), float("nan")

    rho, p = spearmanr(labels, preds)
    return float(rho), float(p)


def safe_kendall(labels, preds, name: str = "metric"):
    labels, preds = _finite_pair_mask(labels, preds)

    if len(labels) < 2:
        print(f"{name}: not enough valid points for Kendall tau.", flush=True)
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"{name}: predictions are constant; Kendall tau undefined.", flush=True)
        return float("nan"), float("nan")

    tau, p = kendalltau(labels, preds)
    return float(tau), float(p)


def _group_bootstrap_interval(
    *,
    group_ids,
    statistic: Callable[[np.ndarray], float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a statistic by source group rather than dependent rows."""
    groups = np.asarray(group_ids, dtype=object)
    if samples < 1 or len(groups) == 0:
        return float("nan"), float("nan")
    unique_groups = list(dict.fromkeys(groups.tolist()))
    indices_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled_groups = rng.integers(0, len(indices_by_group), len(indices_by_group))
        indices = np.concatenate([indices_by_group[index] for index in sampled_groups])
        value = float(statistic(indices))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def correlation_bundle(
    labels,
    preds,
    name: str = "metric",
    *,
    group_ids=None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    finite_mask = np.isfinite(labels) & np.isfinite(preds)
    labels = labels[finite_mask]
    preds = preds[finite_mask]
    if group_ids is not None:
        group_ids = np.asarray(group_ids, dtype=object)
        if group_ids.shape != finite_mask.shape:
            raise ValueError(f"{name}: group_ids length mismatch")
        group_ids = group_ids[finite_mask]

    rho, rho_p = safe_spearman(labels, preds, name=name)
    tau, tau_p = safe_kendall(labels, preds, name=name)

    print(f"  Spearman rho ({name}): {rho:.4f} (p={rho_p:.2e})", flush=True)
    print(f"  Kendall tau ({name}): {tau:.4f} (p={tau_p:.2e})", flush=True)

    result = {
        f"{name}_spearman_rho": rho,
        f"{name}_spearman_p": rho_p,
        f"{name}_kendall_tau": tau,
        f"{name}_kendall_p": tau_p,
    }
    if group_ids is not None and bootstrap_samples:
        def bootstrap_spearman(indices: np.ndarray) -> float:
            sample_labels = labels[indices]
            sample_preds = preds[indices]
            if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_preds)) < 2:
                return float("nan")
            return float(spearmanr(sample_labels, sample_preds).statistic)

        def bootstrap_kendall(indices: np.ndarray) -> float:
            sample_labels = labels[indices]
            sample_preds = preds[indices]
            if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_preds)) < 2:
                return float("nan")
            return float(kendalltau(sample_labels, sample_preds).statistic)

        rho_low, rho_high = _group_bootstrap_interval(
            group_ids=group_ids,
            statistic=bootstrap_spearman,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        tau_low, tau_high = _group_bootstrap_interval(
            group_ids=group_ids,
            statistic=bootstrap_kendall,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
        result.update(
            {
                f"{name}_spearman_ci95_low": rho_low,
                f"{name}_spearman_ci95_high": rho_high,
                f"{name}_kendall_ci95_low": tau_low,
                f"{name}_kendall_ci95_high": tau_high,
                f"{name}_bootstrap_samples": bootstrap_samples,
                f"{name}_bootstrap_groups": len(set(group_ids.tolist())),
            }
        )
    return result


def preference_metrics(
    preferred_scores,
    dispreferred_scores,
    human_ties=None,
    name: str = "preference",
    group_ids=None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    preferred_scores = np.asarray(preferred_scores, dtype=np.float64)
    dispreferred_scores = np.asarray(dispreferred_scores, dtype=np.float64)

    if preferred_scores.shape != dispreferred_scores.shape:
        raise ValueError(
            f"{name}: preferred/dispreferred score shape mismatch: "
            f"{preferred_scores.shape} vs {dispreferred_scores.shape}"
        )

    mask = np.isfinite(preferred_scores) & np.isfinite(dispreferred_scores)
    preferred_scores = preferred_scores[mask]
    dispreferred_scores = dispreferred_scores[mask]
    if group_ids is not None:
        group_ids = np.asarray(group_ids, dtype=object)
        if group_ids.shape != mask.shape:
            raise ValueError(f"{name}: group_ids shape mismatch")
        group_ids = group_ids[mask]

    if len(preferred_scores) == 0:
        return {
            "n": 0,
            "tie_aware_acc": float("nan"),
            "strict_acc": float("nan"),
            "tie_rate": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
        }

    deltas = preferred_scores - dispreferred_scores
    if human_ties is None:
        human_ties = np.zeros(len(deltas), dtype=bool)
    else:
        human_ties = np.asarray(human_ties, dtype=bool)
        if human_ties.shape != mask.shape:
            raise ValueError(f"{name}: human tie mask shape mismatch")
        human_ties = human_ties[mask]
    wins = (deltas > 0) & ~human_ties
    ties = human_ties | ((deltas == 0) & ~human_ties)

    result = {
        "n": int(len(deltas)),
        "tie_aware_acc": float(np.mean(wins.astype(float) + 0.5 * ties.astype(float))),
        "strict_acc": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }
    if group_ids is not None and bootstrap_samples:
        def strict_accuracy(indices: np.ndarray) -> float:
            return float(np.mean(((deltas[indices] > 0) & ~human_ties[indices])))

        def tie_aware_accuracy(indices: np.ndarray) -> float:
            sample_deltas = deltas[indices]
            sample_human_ties = human_ties[indices]
            sample_wins = (sample_deltas > 0) & ~sample_human_ties
            sample_ties = sample_human_ties | (
                (sample_deltas == 0) & ~sample_human_ties
            )
            return float(
                np.mean(sample_wins.astype(float) + 0.5 * sample_ties.astype(float))
            )

        strict_low, strict_high = _group_bootstrap_interval(
            group_ids=group_ids,
            statistic=strict_accuracy,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        tie_low, tie_high = _group_bootstrap_interval(
            group_ids=group_ids,
            statistic=tie_aware_accuracy,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
        result.update(
            {
                "strict_acc_ci95_low": strict_low,
                "strict_acc_ci95_high": strict_high,
                "tie_aware_acc_ci95_low": tie_low,
                "tie_aware_acc_ci95_high": tie_high,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_groups": len(set(group_ids.tolist())),
            }
        )
    return result


def flatten_preference_metrics(
    name: str,
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if metrics is None:
        return {}
    return {f"{name}_{k}": v for k, v in metrics.items()}
