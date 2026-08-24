# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Token-normalized causal-language-model likelihood scorer.

The returned score is mean log probability per non-padding token.  This is
the negative of token-normalized NLL/PPL directionally: higher values indicate
more fluent text, which matches the FE and correlation interfaces.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from clumsification_code.evals.inference.base import TextScorer


class HFCausalLMPerplexityInferenceModel(TextScorer):
    """Score text with token-normalized causal-LM log likelihood."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
        device_map: Optional[str] = None,
    ) -> None:
        self.device = device
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if device_map:
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        if not device_map:
            self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float32)
        run_device = device or self.device
        scores = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Scoring HF PPL"):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(run_device) for key, value in tokens.items()}
            logits = self.model(**tokens).logits

            # Causal-LM logits at position t predict the token at t+1.
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = tokens["input_ids"][:, 1:]
            target_mask = tokens["attention_mask"][:, 1:].bool()
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp_min(1)
            scores.append((token_log_probs.sum(dim=1) / token_counts).cpu())

        result = torch.cat(scores).numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("HF PPL scorer returned non-finite scores.")
        return result


def load_hf_ppl_model(**kwargs) -> HFCausalLMPerplexityInferenceModel:
    """Construct the HF causal-LM likelihood adapter."""
    return HFCausalLMPerplexityInferenceModel(**kwargs)
