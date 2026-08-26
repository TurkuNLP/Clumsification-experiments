"""Normalized loaders for the project's multilingual human evaluations.

These loaders deliberately preserve the annotation type supplied by each
dataset.  BASSE is scalar Likert data; the Norwegian release is pairwise
preference data with an explicit tie option.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, Optional

import datasets


BASSE_DATASET = "HiTZ/BASSE"
NORWEGIAN_FLUENCY_DATASET = "ltg/normistral-fluency-annotation"


def _finite_number(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_finite(values: object) -> Optional[float]:
    if not isinstance(values, (list, tuple)):
        values = [values]
    numbers = []
    for value in values:
        number = _finite_number(value)
        if number is not None:
            numbers.append(number)
    return sum(numbers) / len(numbers) if numbers else None


def iter_basse_records(
    language: str,
    split: str = "test",
    dataset_name: str = BASSE_DATASET,
) -> Iterator[Dict[str, object]]:
    """Yield BASSE fluency ratings as grammaticality-focused scalar records.

    ``language`` is the Hugging Face configuration name (``"es"`` or
    ``"eu"``).  The source dataset also contains other summary criteria; this
    loader intentionally selects only its fluency criterion for this project.
    """
    if language not in {"es", "eu"}:
        raise ValueError("BASSE language must be 'es' (Spanish) or 'eu' (Basque)")

    split_data = datasets.load_dataset(dataset_name, language, split=split)
    language_name = {"es": "Spanish", "eu": "Basque"}[language]
    for row_number, row in enumerate(split_data):
        text = str(row.get("summary", "")).strip()
        #Take the mean of the flunecy labels by annotators. Theres 2-3 for each entry
        score = _mean_finite(row.get("fluency"))
        if not text or score is None:
            continue
        yield {
            "id": f"BASSE:{language}:{split}:{row_number}",
            "source": row.get("document"),
            "text": text,
            "human_scores": [score],
            "human_score": score,
            "benchmark": "BASSE",
            "aspect": "fluency",
            "metadata_aspect": "fluency",
            "task": "Summarization",
            "original_data": f"{BASSE_DATASET}/{language}",
            "task_family": "summarization",
            "fluency_categories": ("grammaticality",),
            "label_type": "scalar",
            "language": language_name,
            "language_code": language,
            "model": row.get("model"),
            "prompt": row.get("prompt"),
            "annotator_scores": row.get("fluency"),
            "rating_scale": "5-point Likert",
        }


def iter_norwegian_preference_records(
    split: str = "test",
    dataset_name: str = NORWEGIAN_FLUENCY_DATASET,
) -> Iterator[Dict[str, object]]:
    """Yield Norwegian Bokmål native-speaker fluency preferences.

    The dataset's ``choice`` field is ``A is more fluent``, ``B is more
    fluent``, or ``Equally fluent``.  Ties are retained as ``tie=True`` and
    are not converted into an arbitrary winner.
    """
    split_data = datasets.load_dataset(dataset_name, split=split)
    for row_number, row in enumerate(split_data):
        response_a = str(row.get("response_a", "")).strip()
        response_b = str(row.get("response_b", "")).strip()
        choice = str(row.get("choice", "")).strip()
        if not response_a or not response_b:
            continue
        choice_lower = choice.lower()
        if choice_lower.startswith("a "):
            preferred, dispreferred, tie = response_a, response_b, False
        elif choice_lower.startswith("b "):
            preferred, dispreferred, tie = response_b, response_a, False
        elif choice_lower.startswith("equally"):
            preferred, dispreferred, tie = response_a, response_b, True
        else:
            raise ValueError(f"Unknown Norwegian preference label: {choice!r}")

        yield {
            "id": f"Norwegian-fluency:{split}:{row_number}",
            "source": row.get("prompt"),
            "preferred_text": preferred,
            "dispreferred_text": dispreferred,
            "tie": tie,
            "benchmark": "Norwegian fluency",
            "aspect": "fluency",
            "metadata_aspect": "fluency",
            "task": "General text generation",
            "original_data": NORWEGIAN_FLUENCY_DATASET,
            "task_family": "controlled_prose",
            "fluency_categories": (
                "grammaticality", "coherence", "clarity", "naturalness",
            ),
            "label_type": "preference",
            "language": "Norwegian Bokmål",
            "language_code": "nb",
            "choice": choice,
            "annotator_id": row.get("annotator_id"),
            "model_a": row.get("model_a"),
            "model_b": row.get("model_b"),
        }
