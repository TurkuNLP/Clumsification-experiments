from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .losses import pairwise_ranking_loss_flat
from .utils import assert_uniform_floating_dtype, get_preferred_param_dtype


class ScoringHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
            nn.init.normal_(self.net.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

            nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pad_group_scores(flat_scores: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    sizes = group_sizes.detach().cpu().tolist()
    max_k = max(sizes)

    padded = flat_scores.new_zeros((len(sizes), max_k))

    start = 0
    for b, n in enumerate(sizes):
        padded[b, :n] = flat_scores[start:start + n]
        start += n

    return padded


def _load_auto_model_with_dtype(
    model_name: str,
    attn_implementation: str,
    param_dtype: torch.dtype,
):
    """
    Newer Transformers versions accept `dtype=...`.
    Older versions often expect `torch_dtype=...`.

    This keeps the refactor compatible without changing the intended dtype behavior.
    """
    try:
        return AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=param_dtype,
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=param_dtype,
        )


class LTRModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        attn_implementation: str = "sdpa",
        param_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.param_dtype = param_dtype or get_preferred_param_dtype()

        self.encoder = _load_auto_model_with_dtype(
            model_name=model_name,
            attn_implementation=attn_implementation,
            param_dtype=self.param_dtype,
        )

        self.encoder.config.use_cache = False

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        emb_dim = self.encoder.config.hidden_size

        self.scorer = ScoringHead(
            emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        encoder_device = next(self.encoder.parameters()).device
        self.scorer.to(device=encoder_device, dtype=self.param_dtype)

        self.to(dtype=self.param_dtype)

        assert_uniform_floating_dtype(
            self,
            expected_dtype=self.param_dtype,
            name="LTRModel before FSDP",
        )

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def score_flat(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.encode(input_ids, attention_mask)
        return self.scorer(emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_sizes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epsilon: float = 0.2,
        scale: float = 5.0,
        loss_normalization: str = "items",
        loss: str = "logistic",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        flat_scores = self.score_flat(input_ids, attention_mask)
        scores = pad_group_scores(flat_scores, group_sizes)

        output = {
            "flat_scores": flat_scores,
            "scores": scores,
            "logits": scores,
        }

        if labels is not None:
            output["loss"] = pairwise_ranking_loss_flat(
                flat_scores=flat_scores,
                labels=labels,
                group_sizes=group_sizes,
                loss=loss,
                epsilon=epsilon,
                scale=scale,
                normalization=loss_normalization,
            )

        return output