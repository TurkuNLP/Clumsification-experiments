# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Pure metrics for flattened FE training rows. Replaces the old per-training method metris implementation."""

from __future__ import annotations

from typing import Any

import torch


def _as_vector(values: Any) -> torch.Tensor:
    if isinstance(values, tuple):
        values = values[0]
    return torch.as_tensor(values).reshape(-1).float()


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Average-tie ranks, matching the usual Spearman definition."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return 0.0
    return float(((left * right).sum() / denominator).item())


def regression_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = _as_vector(eval_prediction.predictions)
    targets = _as_vector(eval_prediction.label_ids)
    if predictions.numel() != targets.numel():
        raise ValueError("Regression predictions and labels have different lengths")
    if not predictions.numel():
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}
    errors = predictions - targets
    return {
        "mae": float(errors.abs().mean().item()),
        "rmse": float(errors.square().mean().sqrt().item()),
        "pearson": _correlation(predictions, targets),
        "spearman": _correlation(_rank(predictions), _rank(targets)),
    }


def pairwise_metrics(eval_prediction: Any) -> dict[str, float]:
    differences = _as_vector(eval_prediction.predictions)
    if not differences.numel():
        return {"pairwise_accuracy": 0.0, "score_tie_rate": 0.0}
    ties = torch.isclose(differences, torch.zeros_like(differences), atol=1e-6, rtol=0.0)
    return {
        "pairwise_accuracy": float((differences > 0).float().mean().item()),
        "score_tie_rate": float(ties.float().mean().item()),
    }
