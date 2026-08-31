# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Shared filesystem primitives for canonical and transitional dataset code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

def default_formatted_dataset_path(dataset_name: str) -> str:
    return os.path.join(
        "data",
        "hf_datasets",
        dataset_name,
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and report the exact malformed row."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {source}:{line_number}")
            rows.append(value)
    return rows


def _atomic_text_write(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {destination}")

    def write(handle: Any) -> None:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _atomic_text_write(destination, write)
    return destination


def write_jsonl_atomic(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {destination}")
    materialized = [dict(row) for row in rows]

    def write(handle: Any) -> None:
        for row in materialized:
            json.dump(row, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")

    _atomic_text_write(destination, write)
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_ds(ds_path: str):
    """Compatibility alias for historical callers."""
    return read_jsonl(ds_path)


__all__ = [
    "canonical_json_hash",
    "default_formatted_dataset_path",
    "read_ds",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "write_json_atomic",
    "write_jsonl_atomic",
]
