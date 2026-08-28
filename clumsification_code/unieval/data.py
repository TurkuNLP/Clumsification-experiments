# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Streaming loader and reproducibility audit for UniEval JSONL data.

The released files are intentionally treated as immutable input.  This module
does not clean, deduplicate, or rewrite rows; it only validates and reports on
them.  Keeping the audit separate is important because duplicate/conflicting
pseudo-labels are themselves useful experimental conditions.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Mapping


class UniEvalDatasetError(ValueError):
    """Raised when a UniEval row is malformed."""


def iter_unieval_rows(path: str | Path) -> Iterator[dict]:
    """Yield validated rows from a UniEval JSONL file without loading it all."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UniEvalDatasetError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise UniEvalDatasetError(f"{path}:{line_number}: row is not an object")
            if not isinstance(row.get("src"), str):
                raise UniEvalDatasetError(f"{path}:{line_number}: 'src' must be a string")
            if row.get("tgt") not in {"Yes", "No"}:
                raise UniEvalDatasetError(
                    f"{path}:{line_number}: 'tgt' must be exactly 'Yes' or 'No'"
                )
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_unieval_file(path: str | Path) -> dict:
    """Return deterministic counts and data-quality diagnostics for one file."""
    path = Path(path)
    labels = {"Yes": 0, "No": 0}
    unique_src: set[str] = set()
    src_labels: defaultdict[str, set[str]] = defaultdict(set)
    total_chars = 0
    max_chars = 0
    rows = 0
    for row in iter_unieval_rows(path):
        rows += 1
        label = row["tgt"]
        labels[label] += 1
        src = row["src"]
        unique_src.add(src)
        src_labels[src].add(label)
        length = len(src)
        total_chars += length
        max_chars = max(max_chars, length)
    conflicts = sum(len(values) > 1 for values in src_labels.values())
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": rows,
        "labels": labels,
        "unique_src": len(unique_src),
        "unique_pairs": len({(src, label) for src, values in src_labels.items() for label in values}),
        "conflicting_src": conflicts,
        "mean_src_chars": (total_chars / rows) if rows else 0.0,
        "max_src_chars": max_chars,
    }


def build_unieval_manifest(root: str | Path) -> dict:
    """Audit all JSONL files below *root*, sorted by relative path."""
    root = Path(root)
    files = sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl"))
    # The released UniEval files are JSONL despite their .json suffix.
    files = sorted(set(files))
    return {
        "schema": "unieval-audit-v1",
        "root": str(root),
        "files": [audit_unieval_file(path) for path in files],
    }
