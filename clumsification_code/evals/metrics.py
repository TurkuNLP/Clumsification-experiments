# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from typing import Any, Dict, Optional

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
        print(f"{name}: not enough valid points for Spearman.")
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"{name}: predictions are constant; Spearman undefined.")
        return float("nan"), float("nan")

    rho, p = spearmanr(labels, preds)
    return float(rho), float(p)


def safe_kendall(labels, preds, name: str = "metric"):
    labels, preds = _finite_pair_mask(labels, preds)

    if len(labels) < 2:
        print(f"{name}: not enough valid points for Kendall tau.")
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"{name}: predictions are constant; Kendall tau undefined.")
        return float("nan"), float("nan")

    tau, p = kendalltau(labels, preds)
    return float(tau), float(p)


def correlation_bundle(labels, preds, name: str = "metric") -> Dict[str, float]:
    labels, preds = _finite_pair_mask(labels, preds)

    rho, rho_p = safe_spearman(labels, preds, name=name)
    tau, tau_p = safe_kendall(labels, preds, name=name)

    print(f"  Spearman rho ({name}): {rho:.4f} (p={rho_p:.2e})")
    print(f"  Kendall tau ({name}): {tau:.4f} (p={tau_p:.2e})")

    return {
        f"{name}_spearman_rho": rho,
        f"{name}_spearman_p": rho_p,
        f"{name}_kendall_tau": tau,
        f"{name}_kendall_p": tau_p,
    }


def preference_metrics(
    preferred_scores,
    dispreferred_scores,
    name: str = "preference",
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
    wins = deltas > 0
    ties = deltas == 0

    return {
        "n": int(len(deltas)),
        "tie_aware_acc": float(np.mean(wins.astype(float) + 0.5 * ties.astype(float))),
        "strict_acc": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }


def flatten_preference_metrics(
    name: str,
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if metrics is None:
        return {}
    return {f"{name}_{k}": v for k, v in metrics.items()}
