# This script has been co-created, refactored, and cleaned using GPT 5.6.
from typing import Callable, Dict

import torch

from .utils import tensor_debug_summary


LOSS_ALIASES: Dict[str, str] = {
    "logistic": "logistic",
    "pairwise_logistic": "logistic",
    "hinge": "hinge",
    "margin": "hinge",
    "weighted_logistic": "weighted_logistic",
    "logistic_weighted": "weighted_logistic",
    "weighted-logistic": "weighted_logistic",
}


def canonicalize_loss_name(loss: str) -> str:
    if loss is None:
        return "logistic"

    key = str(loss).strip().lower()

    if key not in LOSS_ALIASES:
        valid = ", ".join(sorted(LOSS_ALIASES))
        raise ValueError(f"Unknown loss: {loss!r}. Valid losses are: {valid}")

    return LOSS_ALIASES[key]


def get_available_loss_names():
    return tuple(sorted(LOSS_ALIASES))


def _validate_normalization(normalization: str) -> None:
    if normalization not in {"pairs", "items"}:
        raise ValueError(f"Unknown normalization: {normalization}")


def _pairwise_loss_loop(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    normalization: str,
    loss_kind: str,
    epsilon: float = 0.2,
    scale: float = 5.0,
) -> torch.Tensor:
    _validate_normalization(normalization)

    loss_kind = canonicalize_loss_name(loss_kind)

    scores = flat_scores.float()
    device = scores.device
    dtype = scores.dtype

    if not torch.isfinite(scores).all():
        bad = tensor_debug_summary(scores)
        raise FloatingPointError(f"Non-finite flat_scores before loss: {bad}")

    graph_zero = scores.sum() * 0.0
    total_loss = graph_zero
    denom = scores.new_zeros(())

    start = 0
    sizes = group_sizes.detach().cpu().tolist()
    any_valid_pair = False

    for b, n in enumerate(sizes):
        group_scores = scores[start:start + n]
        group_labels = labels[b, :n].to(device=device)

        if normalization == "items":
            denom = denom + float(n)

        if n >= 2:
            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)

            si = group_scores[idx_i]
            sj = group_scores[idx_j]

            li = group_labels[idx_i]
            lj = group_labels[idx_j]

            valid = (li != -100) & (lj != -100) & (li != lj)

            if valid.any():
                any_valid_pair = True

                si = si[valid]
                sj = sj[valid]
                li = li[valid]
                lj = lj[valid]

                # Lower label value means better item.
                # If li < lj, item i should score higher than item j.
                sign = torch.where(
                    lj > li,
                    torch.ones_like(lj, dtype=dtype),
                    -torch.ones_like(lj, dtype=dtype),
                )

                diff = si - sj

                if loss_kind == "hinge":
                    losses = torch.relu(epsilon - sign * diff)

                elif loss_kind == "logistic":
                    losses = torch.nn.functional.softplus(-scale * sign * diff)

                elif loss_kind == "weighted_logistic":
                    weights = (li.float() - lj.float()).abs()
                    losses = weights * torch.nn.functional.softplus(
                        -scale * sign * diff
                    )

                else:
                    # This should be unreachable because canonicalize_loss_name
                    # already validates loss_kind.
                    raise ValueError(f"Unknown loss_kind: {loss_kind}")

                if not torch.isfinite(losses).all():
                    raise FloatingPointError(
                        f"Non-finite pairwise losses. "
                        f"si={tensor_debug_summary(si)}, "
                        f"sj={tensor_debug_summary(sj)}, "
                        f"li={tensor_debug_summary(li)}, "
                        f"lj={tensor_debug_summary(lj)}, "
                        f"diff={tensor_debug_summary(diff)}, "
                        f"losses={tensor_debug_summary(losses)}"
                    )

                total_loss = total_loss + losses.sum()

                if normalization == "pairs":
                    denom = denom + losses.numel()

        start += n

    if normalization == "pairs" and not any_valid_pair:
        return graph_zero

    denom = denom.clamp_min(1.0)
    loss = total_loss / denom

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite final loss. "
            f"total_loss={tensor_debug_summary(total_loss)}, "
            f"denom={tensor_debug_summary(denom)}, "
            f"loss={tensor_debug_summary(loss)}"
        )

    return loss


def pairwise_ranking_loss_flat(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    loss: str = "logistic",
    epsilon: float = 0.2,
    scale: float = 5.0,
    normalization: str = "items",
) -> torch.Tensor:
    """
    Generic pairwise ranking loss dispatcher.

    Lower label value means better item.

    Available canonical losses:
      - logistic
      - hinge
      - weighted_logistic

    The default is logistic, preserving previous behavior.
    """
    return _pairwise_loss_loop(
        flat_scores=flat_scores,
        labels=labels,
        group_sizes=group_sizes,
        normalization=normalization,
        loss_kind=loss,
        epsilon=epsilon,
        scale=scale,
    )


def pairwise_margin_ranking_loss_flat(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    epsilon: float = 0.2,
    scale: float = 5.0,
    normalization: str = "items",
) -> torch.Tensor:
    """
    Pairwise hinge ranking loss over variable-size groups.

    Lower label value means better item.
    If label_i < label_j, score_i should be greater than score_j.
    """
    return pairwise_ranking_loss_flat(
        flat_scores=flat_scores,
        labels=labels,
        group_sizes=group_sizes,
        loss="hinge",
        epsilon=epsilon,
        scale=scale,
        normalization=normalization,
    )


def pairwise_logistic_ranking_loss_flat(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    epsilon: float = 0.2,
    scale: float = 5.0,
    normalization: str = "items",
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss over variable-size groups.

    Lower label value means better item.
    If label_i < label_j, score_i should be greater than score_j.
    """
    return pairwise_ranking_loss_flat(
        flat_scores=flat_scores,
        labels=labels,
        group_sizes=group_sizes,
        loss="logistic",
        epsilon=epsilon,
        scale=scale,
        normalization=normalization,
    )


def pairwise_logistic_weighted_ranking_loss_flat(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    epsilon: float = 0.2,
    scale: float = 5.0,
    normalization: str = "items",
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss with absolute label-gap weighting.
    """
    return pairwise_ranking_loss_flat(
        flat_scores=flat_scores,
        labels=labels,
        group_sizes=group_sizes,
        loss="weighted_logistic",
        epsilon=epsilon,
        scale=scale,
        normalization=normalization,
    )
