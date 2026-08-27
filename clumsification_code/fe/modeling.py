# This script has been co-created, refactored, and cleaned using GPT 5.6.
import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .losses import canonicalize_loss_name, pairwise_ranking_loss_flat, regression_loss
from .utils import assert_uniform_floating_dtype, get_preferred_param_dtype


class EvaluationHead(nn.Module):
    """A single linear projection to the candidate quality scalar.

    ``hidden_dim`` and ``dropout`` remain accepted temporarily so old callers
    fail gracefully during the migration, but are intentionally ignored.  The
    FE contract is a linear head, not an MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        del hidden_dim, dropout
        self.net = nn.Linear(input_dim, 1)
        nn.init.normal_(self.net.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LegacyEvaluationHead(nn.Module):
    """Archived MLP head used only to evaluate pre-migration checkpoints."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pad_group_scores(flat_scores: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    sizes = group_sizes.detach().cpu().tolist()
    padded = flat_scores.new_zeros((len(sizes), max(sizes)))
    start = 0
    for batch_index, group_size in enumerate(sizes):
        padded[batch_index, :group_size] = flat_scores[start:start + group_size]
        start += group_size
    return padded


def _load_auto_model_with_dtype(model_name: str, attn_implementation: str,
                                param_dtype: torch.dtype):
    try:
        return AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            attn_implementation=attn_implementation, dtype=param_dtype,
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            attn_implementation=attn_implementation, torch_dtype=param_dtype,
        )


class FEModel(nn.Module):
    """Shared encoder and candidate-only linear scalar evaluator.

    The optional head arguments are retained only for checkpoint/caller
    compatibility during migration; they do not alter the architecture.
    """

    def __init__(self, model_name: str, hidden_dim: int = 256,
                 dropout: float = 0.1, attn_implementation: str = "sdpa",
                 param_dtype: Optional[torch.dtype] = None,
                 legacy_head: bool = False):
        super().__init__()
        self.param_dtype = param_dtype or get_preferred_param_dtype()
        self.encoder = _load_auto_model_with_dtype(
            model_name, attn_implementation, self.param_dtype
        )
        self.encoder.config.use_cache = False
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.evaluation_head = (
            LegacyEvaluationHead(self.encoder.config.hidden_size, hidden_dim, dropout)
            if legacy_head else EvaluationHead(self.encoder.config.hidden_size)
        )
        self.evaluation_head.to(
            device=next(self.encoder.parameters()).device,
            dtype=self.param_dtype,
        )
        self.to(dtype=self.param_dtype)
        assert_uniform_floating_dtype(
            self, expected_dtype=self.param_dtype, name="FEModel before FSDP"
        )

    def save_pretrained(self, save_directory: str, *, tokenizer=None, metadata=None) -> None:
        """Save a complete new-format FE checkpoint.

        The nested encoder keeps its normal Hugging Face files; the complete
        FE state is additionally stored so the linear head is never separated
        from the encoder for new checkpoints.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.encoder.save_pretrained(save_directory, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        config = {
                "model_name": getattr(self.encoder.config, "_name_or_path", ""),
                "head_type": "linear",
                "hidden_size": self.encoder.config.hidden_size,
                "architecture": "candidate_only_encoder_mean_pool_linear_scalar",
            }
        if metadata:
            config["training"] = metadata
        with open(os.path.join(save_directory, "fe_model_config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "fe_model_state.pt"))

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        """Load a complete new-format FE checkpoint."""
        config_path = os.path.join(save_directory, "fe_model_config.json")
        state_path = os.path.join(save_directory, "fe_model_state.pt")
        if not os.path.exists(config_path) or not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Not a complete FE checkpoint: {save_directory}. "
                "Expected fe_model_config.json and fe_model_state.pt."
            )
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        model = cls(model_name=save_directory, **kwargs)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    def mean_pool(self, last_hidden_state: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        return self.mean_pool(output.last_hidden_state, attention_mask)

    def score_text_batch(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        return self.evaluation_head(self.encode(input_ids, attention_mask))

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                group_sizes: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, epsilon: float = 0.2,
                scale: float = 5.0, loss_normalization: str = "items",
                loss: str = "logistic", regression_loss_name: str = "huber",
                huber_delta: float = 1.0,
                chosen_input_ids: Optional[torch.Tensor] = None,
                chosen_attention_mask: Optional[torch.Tensor] = None,
                rejected_input_ids: Optional[torch.Tensor] = None,
                rejected_attention_mask: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        if chosen_input_ids is not None or rejected_input_ids is not None:
            if chosen_input_ids is None or chosen_attention_mask is None:
                raise ValueError("chosen pair inputs must be provided together")
            if rejected_input_ids is None or rejected_attention_mask is None:
                raise ValueError("rejected pair inputs must be provided together")
            chosen_scores = self.score_text_batch(chosen_input_ids, chosen_attention_mask)
            rejected_scores = self.score_text_batch(rejected_input_ids, rejected_attention_mask)
            differences = chosen_scores - rejected_scores
            epsilon = getattr(self, "ranking_epsilon", epsilon)
            scale = getattr(self, "ranking_scale", scale)
            loss = getattr(self, "ranking_loss", loss)
            loss_name = canonicalize_loss_name(loss)
            pair_loss = (
                torch.relu(epsilon - differences)
                if loss_name == "hinge"
                else torch.nn.functional.softplus(-scale * differences)
            )
            if weights is not None and loss_name == "weighted_logistic":
                pair_loss = pair_loss * weights.to(pair_loss.device, pair_loss.dtype)
            return {
                "logits": differences,
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores,
                "loss": pair_loss.mean(),
            } if labels is not None else {
                "logits": differences,
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores,
            }

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")
        flat_scores = self.score_text_batch(input_ids, attention_mask)

        if group_sizes is None:
            output = {"flat_scores": flat_scores, "logits": flat_scores}
            if labels is not None:
                regression_loss_name = getattr(
                    self, "regression_loss_name", regression_loss_name
                )
                huber_delta = getattr(self, "regression_huber_delta", huber_delta)
                output["loss"] = regression_loss(
                    flat_scores, labels, loss=regression_loss_name,
                    huber_delta=huber_delta,
                )
            return output

        scores = pad_group_scores(flat_scores, group_sizes)
        output = {"flat_scores": flat_scores, "scores": scores, "logits": scores}
        if labels is not None:
            output["loss"] = pairwise_ranking_loss_flat(
                flat_scores=flat_scores, labels=labels, group_sizes=group_sizes,
                loss=loss, epsilon=epsilon, scale=scale,
                normalization=loss_normalization,
            )
        return output
