# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Pure metrics for flattened FE training rows.

The functions accept Hugging Face ``EvalPrediction`` objects (or any object
with matching ``predictions``/``label_ids`` attributes) and return JSON-safe
scalar values. Pairwise predictions are the chosen-minus-rejected score
difference produced by :class:`FEModel`.
"""

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
    """Return error and rank-correlation metrics for scalar predictions."""
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


def binary_metrics(eval_prediction: Any) -> dict[str, float]:
    """Return pointwise binary accuracy from logits."""
    logits = _as_vector(eval_prediction.predictions)
    labels = _as_vector(eval_prediction.label_ids)
    if logits.numel() != labels.numel():
        raise ValueError("Binary predictions and labels have different lengths")
    if not logits.numel():
        return {"binary_accuracy": 0.0}
    accuracy = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
    return {"binary_accuracy": float(accuracy.item())}


def pairwise_metrics(eval_prediction: Any) -> dict[str, float]:
    """Return tie-aware accuracy from chosen-minus-rejected score differences.

    A positive difference is correct, a negative difference is incorrect, and
    a numerical tie contributes half a point. ``label_ids`` is intentionally
    unused because pairwise flattening always orders the lower-quality item as
    rejected and the model output already encodes that target direction.
    """
    differences = _as_vector(eval_prediction.predictions)
    if not differences.numel():
        return {"pairwise_accuracy": 0.0, "score_tie_rate": 0.0}
    ties = torch.isclose(
        differences, torch.zeros_like(differences), atol=1e-6, rtol=0.0
    )
    points = torch.where(
        ties,
        torch.full_like(differences, 0.5),
        (differences > 0).float(),
    )
    return {
        "pairwise_accuracy": float(points.mean().item()),
        "score_tie_rate": float(ties.float().mean().item()),
    }
