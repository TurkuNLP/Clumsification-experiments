# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Streaming loader for the Themis NLG-eval JSONL benchmark file."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .benchmark_registry import BenchmarkSpec, get_nlg_eval_specs


DEFAULT_NLG_EVAL_PATH = Path("data/benchmarks/NLG-Eval.jsonl")
DEFAULT_NLG_EVAL_METADATA_PATH = Path("data/benchmarks/NLG-Eval_meta_info.csv")
REQUIRED_METADATA_COLUMNS = (
    "task", "benchmark", "original_data", "aspect_number", "aspect",
    "annotator_number", "eval_scale", "raw_volume", "retained_volume",
    "paper", "url",
)


def load_nlg_eval_metadata(path: Path = DEFAULT_NLG_EVAL_METADATA_PATH) -> List[Dict[str, str]]:
    """Load and validate the small metadata catalogue.

    The catalogue is deliberately validated before records are selected so
    that a typo in a benchmark definition cannot silently produce an empty
    evaluation subset.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file has no header: {path}")
        missing = set(REQUIRED_METADATA_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metadata file is missing columns: {sorted(missing)}")
        rows = list(reader)
    #Skip the header rows so start at 2
    for row_number, row in enumerate(rows, start=2):
        for field in REQUIRED_METADATA_COLUMNS:
            if not row.get(field, "").strip():
                raise ValueError(f"Empty metadata field {field!r} at row {row_number}")
        try:
            aspect_number = int(row["aspect_number"])
            raw_volume = int(row["raw_volume"])
            retained_volume = int(row["retained_volume"])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric metadata at row {row_number}") from exc
        aspect_names = [name.strip() for name in row["aspect"].split(",") if name.strip()]
        if aspect_number != len(aspect_names):
            raise ValueError(f"Aspect count mismatch at metadata row {row_number}")
        if retained_volume > raw_volume:
            raise ValueError(f"Retained volume exceeds raw volume at row {row_number}")
    return rows


def _aspect_base(aspect: str) -> str:
    # JSONL aspects start with a short name followed by a colon and rubric text.
    return aspect.strip().split(":", 1)[0].strip()


def _aspect_matches(actual: str, canonical: str) -> bool:
    # Some rubrics use forms such as "Fluency (or Grammaticality)".  Accept
    # those extensions while still requiring the canonical aspect as a prefix.
    actual_base = _aspect_base(actual)
    canonical = canonical.strip()
    return actual_base == canonical or actual_base.startswith(canonical + " ") or actual_base.startswith(canonical + "(")


def _as_float_list(value: object) -> List[float]:
    # Keep individual annotator scores so later agreement analyses remain
    # possible; the normalized record also stores their arithmetic mean.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [float(value)]
    elif isinstance(value, list):
        values = [float(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
    else:
        values = []
    return [value for value in values if math.isfinite(value)]


def _record_matches(record: Mapping[str, object], spec: BenchmarkSpec) -> bool:
    # Benchmark and aspect are always required.  Task and original_data are
    # optional disambiguators for benchmark names with multiple task subsets.
    if record.get("benchmark") != spec.benchmark:
        return False
    if not _aspect_matches(str(record.get("aspect", "")), spec.aspect):
        return False
    if spec.task is not None and record.get("task") != spec.task:
        return False
    if spec.original_data is not None and record.get("original_data") != spec.original_data:
        return False
    return True


def iter_nlg_eval_records(
    path: Path = DEFAULT_NLG_EVAL_PATH,
    specs: Optional[Sequence[BenchmarkSpec]] = None,
) -> Iterator[Dict[str, object]]:
    """Yield normalized records without loading the JSONL into memory.

    The line number is used as a stable local identifier because the source
    file has no universal record ID and records are evaluated in file order.
    """
    selected_specs = tuple(specs if specs is not None else get_nlg_eval_specs())
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_number}") from exc

            # A record normally matches one spec.  Stop after the first match
            # to prevent duplicate output if future registry entries overlap.
            for spec in selected_specs:
                if not _record_matches(raw, spec):
                    continue
                human_scores = _as_float_list(raw.get("human_score"))
                if not human_scores:
                    continue
                yield {
                    "id": f"nlg-eval:{line_number}",
                    "source": raw.get("source"),
                    "text": raw.get("target"),
                    "human_scores": human_scores,
                    "human_score": sum(human_scores) / len(human_scores),
                    "task": raw.get("task"),
                    "benchmark": raw.get("benchmark"),
                    "original_data": raw.get("original_data"),
                    "aspect": raw.get("aspect"),
                    "metadata_aspect": spec.aspect,
                    "spec_name": spec.name,
                    "task_family": spec.task_family,
                    "fluency_categories": spec.categories,
                    "label_type": "scalar",
                    "eval_scale": raw.get("eval_scale"),
                    "annotator": raw.get("annotator"),
                }
                break
