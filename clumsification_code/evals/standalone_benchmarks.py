# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Normalized loaders for evaluation sources outside NLG-eval."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional


# These are the repository-local evaluation files that are intentionally kept
# outside NLG-eval.  Centralizing them prevents different runners from silently
# using different copies of the same benchmark.
DEFAULT_ELLIPSE_PATH = Path("data/benchmarks/ELLIPSE.csv")
DEFAULT_ARGESSAY_PATH = Path("data/benchmarks/arg-essay.csv")
DEFAULT_COHESENTIA_PATH = Path("data/benchmarks/CohesentiaTestData.json")


def _number(value: object) -> Optional[float]:
    """Convert a CSV/JSON label to a finite float, or return None."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def iter_ellipse_records(path: Path) -> Iterator[Dict[str, object]]:
    """Yield ELLIPSE records with one normalized record per essay."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            text = (row.get("full_text") or "").strip()
            if not text:
                continue
            # ELLIPSE supplies several language-quality dimensions.  Only
            # grammar and cohesion are in the primary fluency suite.
            for aspect, column, category in (
                ("grammar", "Grammar", "grammaticality"),
                ("cohesion", "Cohesion", "coherence"),
            ):
                score = _number(row.get(column))
                if score is None:
                    continue
                yield {
                    "id": f"ELLIPSE:{row_number}",
                    "source": None,
                    "text": text,
                    "human_scores": [score],
                    "human_score": score,
                    "benchmark": "ELLIPSE",
                    "aspect": aspect,
                    "metadata_aspect": aspect,
                    "task": "Essay Evaluation",
                    "original_data": "ELLIPSE",
                    "task_family": "essays",
                    "fluency_categories": (category,),
                    "label_type": "scalar",
                }


def iter_argessay_records(path: Path) -> Iterator[Dict[str, object]]:
    """Yield ARG-ESSAY records for the selected proficiency dimensions."""
    columns = {
        "language_mastery": "STUD_LangMastery",
        "language_constructs": "STUD_LangConstructs",
    }
    text_columns = {
        "Student": ("human", "STUD"),
        "ChatGPT-3": ("gpt3", "GPT3"),
        "ChatGPT-4": ("gpt4", "GPT4"),
    }
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            for text_column, (system, score_prefix) in text_columns.items():
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                for aspect, student_column in columns.items():
                    suffix = student_column.split("_", 1)[1]
                    score = _number(row.get(f"{score_prefix}_{suffix}"))
                    if score is None:
                        continue
                    yield {
                        "id": f"ARG-ESSAY:{row_number}:{system}:{aspect}",
                        "source": None,
                        "text": text,
                        "human_scores": [score],
                        "human_score": score,
                        "benchmark": "ARG-ESSAY",
                        "aspect": aspect,
                        "metadata_aspect": aspect,
                        "task": "Essay Evaluation",
                        "original_data": "ARG-ESSAY",
                        "task_family": "essays",
                        "fluency_categories": ("grammaticality",),
                        "label_type": "scalar",
                        "system": system,
                    }


def iter_cohesentia_records(path: Path) -> Iterator[Dict[str, object]]:
    """Yield evaluation-only holistic and incremental CoheSentia records."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.values() if isinstance(data, dict) else data
    if not hasattr(entries, "__iter__"):  # pragma: no cover
        raise ValueError(f"Unexpected CoheSentia structure in {path}")
    # The type check above is intentionally broad; iteration below handles the
    # dictionary-values view and list forms used by released benchmark files.
    for index, entry in enumerate(entries):
        text = str(entry.get("Text", "")).strip()
        if not text:
            continue
        for aspect, key in (("coherence_holistic", "HolisticData"), ("coherence_incremental", "IncrementalData")):
            score = _number(entry.get(key, {}).get("consensus_score"))
            if score is None:
                continue
            yield {
                "id": f"CoheSentia:{index}:{aspect}",
                "source": None,
                "text": text,
                "human_scores": [score],
                "human_score": score,
                "benchmark": "CoheSentia",
                "aspect": aspect,
                "metadata_aspect": aspect,
                "task": "Story Evaluation",
                "original_data": "CoheSentia",
                "task_family": "controlled_prose",
                "fluency_categories": ("coherence",),
                "label_type": "scalar",
            }


def iter_standalone_records(
    *,
    ellipse_path: Optional[Path] = None,
    argessay_path: Optional[Path] = None,
    cohesentia_path: Optional[Path] = None,
) -> Iterator[Dict[str, object]]:
    """Yield records from whichever standalone sources were requested."""
    if ellipse_path is not None:
        yield from iter_ellipse_records(ellipse_path)
    if argessay_path is not None:
        yield from iter_argessay_records(argessay_path)
    if cohesentia_path is not None:
        yield from iter_cohesentia_records(cohesentia_path)
