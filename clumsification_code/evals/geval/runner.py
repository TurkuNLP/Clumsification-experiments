# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch

from clumsification_code.evals.geval.scorer import GEvalScorer


class GEvalBenchmarkAdapter:
    """
    Thin adapter around GEvalScorer for benchmark_runner.py-style evaluation.

    This is mostly useful if you later want a wrapper that can inject task/aspect
    metadata before calling score_texts(...). For the current benchmark runner,
    GEvalScorer itself already exposes the required score_texts(...) method.
    """

    def __init__(self, scorer: GEvalScorer):
        self.scorer = scorer

    def score_texts(
        self,
        texts: Iterable[str],
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> np.ndarray:
        return self.scorer.score_texts(
            texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )


def build_geval_scorer_from_args(args: Any) -> GEvalScorer:
    return GEvalScorer.from_args(args)


def build_geval_benchmark_adapter_from_args(args: Any) -> GEvalBenchmarkAdapter:
    return GEvalBenchmarkAdapter(GEvalScorer.from_args(args))


def score_texts_for_benchmark(
    scorer: GEvalScorer,
    texts: Iterable[str],
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    max_length: int = 512,
) -> np.ndarray:
    return scorer.score_texts(
        texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
