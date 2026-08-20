# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TextScorer(Protocol):
    """Minimal interface required by benchmark and TDT evaluation."""

    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        ...


@runtime_checkable
class PromptAwareTextScorer(TextScorer, Protocol):
    """Optional interface for GPTScore/G-Eval-style prompt-aware scorers."""

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        ...
