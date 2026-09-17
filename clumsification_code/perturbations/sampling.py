# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Deterministic sampling for LLM perturbation edit catalogs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence


SEVERITIES = ("weak", "medium", "strong")
ABSOLUTE_MAX_EDIT_COUNT = 5
_REQUIRED_FIELDS = {
    "edit_id",
    "target_dimensions",
    "edit_type",
    "example_edited",
    "example_clean",
}


@dataclass(frozen=True)
class EditCatalogEntry:
    edit_id: str
    target_dimensions: tuple[str, ...]
    edit_type: str
    example_edited: str
    example_clean: str
    instruction: str | None = None
    minimum_realization: str | None = None
    non_examples: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampledEditAssignment:
    target_dimensions: tuple[str, ...]
    edits: tuple[EditCatalogEntry, ...]
    severity: str
    seed: int


def _as_nonempty_string(value: object, field: str, line_no: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Catalog line {line_no}: {field} must be a non-empty string")
    return value


def _as_string_tuple(value: object, field: str, line_no: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Catalog line {line_no}: {field} must be a non-empty array")
    result = tuple(_as_nonempty_string(item, field, line_no) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"Catalog line {line_no}: {field} contains duplicates")
    return result


def _optional_string(value: object, field: str, line_no: int) -> str | None:
    if value is None:
        return None
    return _as_nonempty_string(value, field, line_no)


def _optional_string_tuple(value: object, field: str, line_no: int) -> tuple[str, ...]:
    if value is None:
        return ()
    return _as_string_tuple(value, field, line_no)


def load_edit_catalog(path: str | Path) -> tuple[EditCatalogEntry, ...]:
    """Load and validate a JSONL edit catalog in stable file order."""
    catalog_path = Path(path)
    entries: list[EditCatalogEntry] = []
    seen_ids: set[str] = set()
    with catalog_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Catalog line {line_no} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Catalog line {line_no} must be a JSON object")
            # The revised catalog calls the source illustration example_source.
            # Normalize it to the existing internal name for both catalog formats.
            if "example_source" in value:
                value["example_clean"] = value["example_source"]
            missing = sorted(_REQUIRED_FIELDS - set(value))
            if missing:
                raise ValueError(f"Catalog line {line_no} is missing fields: {missing}")
            edit_id = _as_nonempty_string(value["edit_id"], "edit_id", line_no)
            if edit_id in seen_ids:
                raise ValueError(f"Duplicate edit_id {edit_id!r} at catalog line {line_no}")
            seen_ids.add(edit_id)
            entries.append(
                EditCatalogEntry(
                    edit_id=edit_id,
                    target_dimensions=_as_string_tuple(value["target_dimensions"], "target_dimensions", line_no),
                    edit_type=_as_nonempty_string(value["edit_type"], "edit_type", line_no),
                    example_edited=_as_nonempty_string(value["example_edited"], "example_edited", line_no),
                    example_clean=_as_nonempty_string(value["example_clean"], "example_clean", line_no),
                    instruction=_optional_string(value.get("instruction"), "instruction", line_no),
                    minimum_realization=_optional_string(value.get("minimum_realization"), "minimum_realization", line_no),
                    non_examples=_optional_string_tuple(value.get("non_examples"), "non_examples", line_no),
                    applicability=_optional_string_tuple(value.get("applicability"), "applicability", line_no),
                )
            )
    if not entries:
        raise ValueError(f"Edit catalog is empty: {catalog_path}")
    return tuple(entries)


def sample_edit_count(
    text_length: int,
    *,
    seed: int,
    max_edits: int | None = None,
) -> int:
    """Sample a length-scaled edit count from a truncated normal distribution.

    The range is one through ``min(5, floor(text_length / 500))``. Short
    texts therefore still receive one edit. ``max_edits`` can impose a lower
    caller-specific cap when a method cannot realize the full shared range.
    """
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    if max_edits is not None and max_edits < 1:
        raise ValueError("max_edits must be at least 1 when provided")

    upper = min(ABSOLUTE_MAX_EDIT_COUNT, max(1, text_length // 500))
    if max_edits is not None:
        upper = min(upper, max_edits)
    if upper == 1:
        return 1

    mean = (1 + upper) / 2
    standard_deviation = (upper - 1) / 6
    rng = random.Random(seed)
    while True:
        draw = rng.normalvariate(mean, standard_deviation)
        if 1 <= draw <= upper:
            return min(upper, max(1, int(draw + 0.5)))


__all__ = [
    "ABSOLUTE_MAX_EDIT_COUNT",
    "SEVERITIES",
    "EditCatalogEntry",
    "SampledEditAssignment",
    "load_edit_catalog",
    "sample_edit_count",
]
