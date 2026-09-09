# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Frozen, balanced assignments for the two LLM perturbation workflows."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

from clumsification_code.data.io import read_jsonl, write_jsonl_atomic

from .sampling import EditCatalogEntry, SEVERITIES, load_edit_catalog


LLM_ASSIGNMENT_METHODS = ("llm_single", "llm_sampled")
DEFAULT_ASSIGNMENT_FILENAME = "perturbation_assignments.jsonl"


def _seed(seed: int, stream: str) -> int:
    raw = f"{seed}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _item_seed(seed: int, base_text_id: str, stream: str) -> int:
    return _seed(seed, f"{base_text_id}:{stream}")


def _derived_dimensions(edits: Sequence[EditCatalogEntry]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            dimension
            for entry in edits
            for dimension in entry.target_dimensions
        )
    )


def _balanced_deck(
    values: Sequence[str], *, count: int, seed: int
) -> list[str]:
    """Return a seeded sequence in which value counts differ by at most one."""
    if not values:
        raise ValueError("Cannot build a balanced deck from no values")
    rng = random.Random(seed)
    deck: list[str] = []
    while len(deck) < count:
        cycle = list(values)
        rng.shuffle(cycle)
        deck.extend(cycle)
    return deck[:count]


def _draw_distinct_operations(
    deck: list[str], counts: Sequence[int]
) -> list[tuple[str, ...]]:
    """Partition a balanced operation deck into distinct per-source selections."""
    remaining = Counter(deck)
    # The deck's seeded order breaks ties reproducibly while the remaining
    # counts ensure every operation is consumed exactly as often as planned.
    tie_order = {operation: index for index, operation in enumerate(dict.fromkeys(deck))}
    assignments: list[tuple[str, ...]] = []
    for count in counts:
        available = sorted(
            (operation for operation, amount in remaining.items() if amount),
            key=lambda operation: (-remaining[operation], tie_order[operation]),
        )
        if len(available) < count:
            raise ValueError("Unable to allocate distinct edit operations")
        selected = available[:count]
        for operation in selected:
            remaining[operation] -= 1
        assignments.append(tuple(selected))
    if any(remaining.values()):
        raise ValueError("Operation allocation did not consume its balanced deck")
    return assignments


def _sample_edit_count(text: str, *, seed: int) -> int:
    """Uniformly sample the documented length-conditioned LLM edit count."""
    upper = min(5, max(1, len(text.replace("\n", " ")) // 500))
    return random.Random(seed).randint(1, upper)


@dataclass(frozen=True)
class LLMAssignment:
    """One frozen LLM assignment for one source and one independent workflow."""

    base_text_id: str
    method: str
    edit_count: int
    edits: tuple[str, ...]
    severity: str
    target_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.base_text_id:
            raise ValueError("base_text_id must not be empty")
        if self.method not in LLM_ASSIGNMENT_METHODS:
            raise ValueError(f"Unsupported LLM assignment method: {self.method!r}")
        if self.edit_count < 1 or self.edit_count != len(self.edits):
            raise ValueError("edit_count must equal a non-empty edits sequence")
        if len(self.edits) != len(set(self.edits)):
            raise ValueError("LLM assignment edits must be distinct")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {self.severity!r}")
        if not self.target_dimensions:
            raise ValueError("target_dimensions must not be empty")

    def to_row(self) -> dict[str, object]:
        return {
            "base_text_id": self.base_text_id,
            "method": self.method,
            "edit_count": self.edit_count,
            "edits": list(self.edits),
            "severity": self.severity,
            "target_dimensions": list(self.target_dimensions),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "LLMAssignment":
        required = {
            "base_text_id", "method", "edit_count", "edits", "severity",
            "target_dimensions",
        }
        unknown = set(row) - required
        missing = required - set(row)
        if unknown or missing:
            raise ValueError(
                f"Assignment row fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        edits = row["edits"]
        dimensions = row["target_dimensions"]
        if not isinstance(edits, list) or not all(isinstance(value, str) for value in edits):
            raise ValueError("Assignment edits must be an array of strings")
        if not isinstance(dimensions, list) or not all(isinstance(value, str) for value in dimensions):
            raise ValueError("Assignment target_dimensions must be an array of strings")
        if isinstance(row["edit_count"], bool) or not isinstance(row["edit_count"], int):
            raise ValueError("Assignment edit_count must be an integer")
        if not all(isinstance(row[field], str) for field in ("base_text_id", "method", "severity")):
            raise ValueError("Assignment identifiers and severity must be strings")
        return cls(
            base_text_id=str(row["base_text_id"]),
            method=str(row["method"]),
            edit_count=int(row["edit_count"]),
            edits=tuple(edits),
            severity=str(row["severity"]),
            target_dimensions=tuple(dimensions),
        )


def plan_llm_assignments(
    records: Iterable[Mapping[str, object]],
    *,
    catalog: Sequence[EditCatalogEntry],
    seed: int = 42,
) -> tuple[LLMAssignment, ...]:
    """Plan balanced assignments for independent ``llm_single`` and sampled runs."""
    values = []
    for record in records:
        base_text_id = record.get("base_text_id")
        text = record.get("text")
        if not isinstance(base_text_id, str) or not base_text_id:
            raise ValueError("Every record needs a non-empty base_text_id")
        if not isinstance(text, str):
            raise ValueError(f"Source {base_text_id!r} has non-string text")
        values.append((base_text_id, text))
    if not values:
        raise ValueError("Cannot plan assignments for no source records")
    if len({base_text_id for base_text_id, _ in values}) != len(values):
        raise ValueError("Source records must have unique base_text_id values")
    if not catalog:
        raise ValueError("Edit catalog must not be empty")

    by_id = {entry.edit_id: entry for entry in catalog}
    if len(by_id) != len(catalog):
        raise ValueError("Edit catalog must have unique edit_id values")
    operation_ids = tuple(sorted(by_id))
    ordered = sorted(values, key=lambda value: value[0])
    random.Random(_seed(seed, "source-order")).shuffle(ordered)

    plans: list[LLMAssignment] = []
    for method in LLM_ASSIGNMENT_METHODS:
        counts = (
            [1] * len(ordered)
            if method == "llm_single"
            else [
                _sample_edit_count(text, seed=_item_seed(seed, base_text_id, "edit-count"))
                for base_text_id, text in ordered
            ]
        )
        operation_deck = _balanced_deck(
            operation_ids,
            count=sum(counts),
            seed=_seed(seed, f"{method}:operations"),
        )
        selected_operations = _draw_distinct_operations(operation_deck, counts)
        severity_deck = _balanced_deck(
            SEVERITIES,
            count=len(ordered),
            seed=_seed(seed, f"{method}:severity"),
        )
        for (base_text_id, _), edit_ids, severity in zip(
            ordered, selected_operations, severity_deck, strict=True
        ):
            entries = tuple(by_id[edit_id] for edit_id in edit_ids)
            plans.append(
                LLMAssignment(
                    base_text_id=base_text_id,
                    method=method,
                    edit_count=len(edit_ids),
                    edits=edit_ids,
                    severity=severity,
                    target_dimensions=_derived_dimensions(entries),
                )
            )
    return tuple(sorted(plans, key=lambda plan: (plan.method, plan.base_text_id)))


def load_llm_assignments(path: str | Path) -> tuple[LLMAssignment, ...]:
    """Load a complete, duplicate-free LLM assignment file."""
    assignments = tuple(LLMAssignment.from_row(row) for row in read_jsonl(path))
    keys = [(assignment.base_text_id, assignment.method) for assignment in assignments]
    if len(keys) != len(set(keys)):
        raise ValueError("Assignment file contains duplicate source/method rows")
    return assignments


def write_llm_assignments(
    path: str | Path,
    assignments: Iterable[LLMAssignment],
    *,
    overwrite: bool = False,
) -> Path:
    """Write assignments in stable source/method order."""
    ordered = sorted(assignments, key=lambda item: (item.method, item.base_text_id))
    return write_jsonl_atomic(path, (item.to_row() for item in ordered), overwrite=overwrite)


def plan_llm_assignment_file(
    records: Iterable[Mapping[str, object]],
    *,
    catalog_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[LLMAssignment, ...]:
    """Plan and write the lightweight LLM assignment manifest."""
    assignments = plan_llm_assignments(
        records, catalog=load_edit_catalog(catalog_path), seed=seed
    )
    write_llm_assignments(output_path, assignments, overwrite=overwrite)
    return assignments


__all__ = [
    "DEFAULT_ASSIGNMENT_FILENAME",
    "LLM_ASSIGNMENT_METHODS",
    "LLMAssignment",
    "load_llm_assignments",
    "plan_llm_assignment_file",
    "plan_llm_assignments",
    "write_llm_assignments",
]
