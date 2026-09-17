# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from clumsification_code.fe.checkpointing import load_fe_model


class FEInferenceModel:
    """Evaluation adapter around the canonical clumsification_code.fe model."""

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        attn_implementation: str = "flash_attention_2",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_dir = model_dir
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_fe_model(
            final_dir=model_dir,
            attn_implementation=attn_implementation,
            param_dtype=dtype,
        )
        self.training_objective = self._read_training_objective(model_dir)

        if getattr(self.model.encoder.config, "pad_token_id", None) is None:
            self.model.encoder.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _read_training_objective(model_dir: str) -> str | None:
        """Read objective metadata without changing legacy checkpoint loading."""
        for filename in ("fe_model_config.json", "fe_trainer_checkpoint.json"):
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
            value = config.get("training", {}).get("objective") if filename == "fe_model_config.json" else config.get("objective")
            if isinstance(value, str):
                return value
        return None

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float32)

        device = device or self.device
        scores = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
            batch_texts = texts[start : start + batch_size]

            tok = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                pad_to_multiple_of=8,
            ).to(device)

            batch_scores = self.model.score_text_batch(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )

            if self.training_objective == "binary":
                batch_scores = torch.sigmoid(batch_scores)

            if not torch.isfinite(batch_scores).all():
                raise RuntimeError(
                    f"Non-finite FE scores in batch starting at {start}. "
                    f"Texts: {batch_texts[:3]}"
                )

            scores.append(batch_scores.detach().cpu().float())

        return torch.cat(scores, dim=0).numpy()


def load_fe_inference_model(
    model_dir: str,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
    dtype: Optional[torch.dtype] = None,
) -> FEInferenceModel:
    return FEInferenceModel(
        model_dir=model_dir,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
