# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Scoring helpers shared by generative and encoder UniEval models."""

from __future__ import annotations

import torch


def normalized_yes_probability(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """Return P(Yes)/(P(Yes)+P(No)) from one decoder-step logits."""
    selected = logits[..., [yes_id, no_id]]
    return torch.softmax(selected, dim=-1)[..., 0]


def generative_step_loss(logits: torch.Tensor, labels: torch.Tensor,
                         yes_id: int, no_id: int) -> torch.Tensor:
    """Binary cross-entropy on Yes/No logits for a decoder step."""
    selected = logits[..., [no_id, yes_id]]
    return torch.nn.functional.cross_entropy(selected, labels.long())
