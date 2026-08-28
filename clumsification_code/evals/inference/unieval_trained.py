# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Inference adapter for locally trained encoder UniEval checkpoints."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from clumsification_code.evals.inference.base import TextScorer
from clumsification_code.unieval.modeling import UniEvalEncoderClassifier


class TrainedUniEvalInferenceModel(TextScorer):
    def __init__(self, model_dir: str, device: torch.device, dtype=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = UniEvalEncoderClassifier.load_checkpoint(model_dir, torch_dtype=dtype, trust_remote_code=True)
        self.model.to(device=device, dtype=dtype or torch.float32).eval()

    @torch.no_grad()
    def score_texts(self, texts: List[str], device=None, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float32)
        target = device or self.device
        values = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
            encoded = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt").to(target)
            values.append(self.model(**encoded)["scores"].cpu().float())
        return torch.cat(values).numpy()


def load_trained_unieval_model(model_dir: str, device: torch.device, dtype=None):
    return TrainedUniEvalInferenceModel(model_dir=model_dir, device=device, dtype=dtype)
