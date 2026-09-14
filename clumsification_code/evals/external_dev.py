# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Frozen sources and leakage checks for human-labeled external development."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .standalone_benchmarks import (
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_COHESENTIA_TRAIN_PATH,
    DEFAULT_ELLIPSE_TEST_PATH,
    DEFAULT_ELLIPSE_TRAIN_PATH,
)


EXPECTED_SPLITS = {
    "ellipse_train": {
        "path": DEFAULT_ELLIPSE_TRAIN_PATH,
        "records": 3911,
        "sha256": "782344e99668a3ff508d7410c0eb6e36da70f3b28f81c96e367f1ca04924b06c",
    },
    "ellipse_test": {
        "path": DEFAULT_ELLIPSE_TEST_PATH,
        "records": 2571,
        "sha256": "7e990c6392a9df9554d15bdd22f0b568d19095cd6676ad39cb1eaa69c977ed7a",
    },
    "cohesentia_train": {
        "path": DEFAULT_COHESENTIA_TRAIN_PATH,
        "records": 434,
        "sha256": "1e7a6041e8e9237ce2ad327fd828d0fe565c074d11aff6a9fde3ed71d1b0c4aa",
    },
    "cohesentia_test": {
        "path": DEFAULT_COHESENTIA_PATH,
        "records": 49,
        "sha256": "691b3b519c82cbe53fd8cae85cd67736535e31309a3afa2634b44c09a73a2331",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ellipse_ids(path: Path) -> set[str]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {(row.get("text_id_kaggle") or "").strip() for row in rows}
    if "" in ids:
        raise ValueError(f"{path}: ELLIPSE contains an empty text_id_kaggle")
    if len(ids) != len(rows):
        raise ValueError(f"{path}: ELLIPSE text_id_kaggle values are not unique")
    return ids


def _cohesentia_ids(path: Path) -> set[str]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entries: Iterable[Dict[str, Any]] = (
        payload.values() if isinstance(payload, dict) else payload
    )
    ids = [str(entry.get("StoryID", "")).strip() for entry in entries]
    if any(not item for item in ids):
        raise ValueError(f"{path}: CoheSentia contains an empty StoryID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: CoheSentia StoryID values are not unique")
    return set(ids)


def audit_external_dev_splits(*, verify_hashes: bool = True) -> Dict[str, Any]:
    """Fail closed when local dev/final files are missing, changed, or overlap."""
    report: Dict[str, Any] = {"splits": {}}
    for name, spec in EXPECTED_SPLITS.items():
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Required audited benchmark split is missing: {path}"
            )
        observed_hash = sha256_file(path)
        if verify_hashes and observed_hash != spec["sha256"]:
            raise ValueError(
                f"{name} checksum mismatch: expected {spec['sha256']}, "
                f"observed {observed_hash}"
            )
        ids = _ellipse_ids(path) if name.startswith("ellipse") else _cohesentia_ids(path)
        if len(ids) != spec["records"]:
            raise ValueError(
                f"{name} count mismatch: expected {spec['records']}, observed {len(ids)}"
            )
        report["splits"][name] = {
            "path": str(path),
            "records": len(ids),
            "sha256": observed_hash,
        }

    ellipse_overlap = _ellipse_ids(DEFAULT_ELLIPSE_TRAIN_PATH) & _ellipse_ids(
        DEFAULT_ELLIPSE_TEST_PATH
    )
    cohesentia_overlap = _cohesentia_ids(
        DEFAULT_COHESENTIA_TRAIN_PATH
    ) & _cohesentia_ids(DEFAULT_COHESENTIA_PATH)
    if ellipse_overlap:
        raise ValueError(f"ELLIPSE development/test overlap: {len(ellipse_overlap)} IDs")
    if cohesentia_overlap:
        raise ValueError(
            f"CoheSentia development/test overlap: {len(cohesentia_overlap)} IDs"
        )
    report["overlap"] = {"ellipse": 0, "cohesentia": 0}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Check counts and overlap without enforcing the frozen file hashes.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_external_dev_splits(
                verify_hashes=not args.skip_hash_verification
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
