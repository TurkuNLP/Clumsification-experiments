# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Candidate-only UniEval fluency adapter.

UniEval is distributed as a small repository rather than as a stable PyPI
package.  The adapter therefore accepts the checkout path explicitly and uses
the official ``metric.evaluator.get_evaluator`` API.  We use the summarization
evaluator's fluency dimension because that dimension requires only the
candidate text and is consequently comparable across this English suite.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from clumsification_code.evals.inference.base import TextScorer


class UniEvalFluencyInferenceModel(TextScorer):
    """Score candidate texts with the official UniEval fluency evaluator."""

    def __init__(
        self,
        *,
        repo_path: Optional[str] = None,
        max_length: int = 1024,
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
    ) -> None:
        if repo_path:
            repo = str(Path(repo_path).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

        try:
            from metric import evaluator as unieval_evaluator
        except ImportError as exc:
            raise ImportError(
                "UniEval requires the official UniEval checkout on the Python "
                "path. Pass --unieval-repo /path/to/UniEval and install its "
                "requirements.txt."
            ) from exc

        self._ensure_sentence_tokenizer(unieval_evaluator)

        self.max_length = max_length
        self.device = str(device)
        self.evaluator = unieval_evaluator.get_evaluator(
            "summarization",
            max_length=max_length,
            device=self.device,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _ensure_sentence_tokenizer(unieval_evaluator) -> None:
        """Avoid a runtime download requirement for NLTK's punkt_tab data."""
        original_tokenizer = unieval_evaluator.sent_tokenize
        try:
            import nltk

            nltk.data.find("tokenizers/punkt_tab/english/")
        except (ImportError, LookupError):
            def original_tokenizer(text: str) -> List[str]:
                cleaned = str(text).strip()
                if not cleaned:
                    return []
                return [part for part in re.split(r"(?<=[.!?])\s+", cleaned) if part]

        def safe_sentence_tokenizer(text: str) -> List[str]:
            """Guarantee the upstream evaluator has one unit to average."""
            sentences = original_tokenizer(text)
            if sentences:
                return sentences
            # A literal period is a minimal valid candidate for the upstream
            # question builder and prevents its zero-sentence division error.
            return ["."]

        # UniEval imported sent_tokenize into metric.evaluator's module
        # namespace, so replacing that reference keeps the official
        # sentence-level fluency logic intact without editing the checkout.
        unieval_evaluator.sent_tokenize = safe_sentence_tokenizer

    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Return one higher-is-better fluency score per candidate.

        The upstream evaluator handles model-level batching.  ``batch_size`` is
        accepted for compatibility with the shared interface; UniEval's own
        scorer controls its internal batch behavior.
        """
        del device, batch_size
        if not texts:
            return np.asarray([], dtype=np.float32)

        data = [
            {"source": "", "system_output": str(text)}
            for text in texts
        ]
        scores = self.evaluator.evaluate(
            data,
            dims=["fluency"],
            overall=False,
            print_result=False,
        )
        values = np.asarray([row["fluency"] for row in scores], dtype=np.float32)
        if values.shape != (len(texts),) or not np.isfinite(values).all():
            raise RuntimeError("UniEval returned missing or non-finite fluency scores.")
        return values


def load_unieval_fluency_model(
    *,
    repo_path: Optional[str],
    max_length: int,
    device,
    cache_dir: Optional[str] = None,
) -> UniEvalFluencyInferenceModel:
    """Construct the candidate-only UniEval fluency adapter."""
    return UniEvalFluencyInferenceModel(
        repo_path=repo_path,
        max_length=max_length,
        device=str(device),
        cache_dir=cache_dir,
    )
