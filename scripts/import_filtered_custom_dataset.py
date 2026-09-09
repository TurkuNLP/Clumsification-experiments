# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Build a canonical custom dataset from rows accepted by the vLLM filter.

The input is the row-preserving JSONL produced by
``filter_scripts/vllm_filter_texts_en_mass.py``.  Only rows whose embedded
filter response is a valid, internally consistent PASS assessment are written.
All other rows fail closed and are counted in an audit manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = 1
FILTER_FIELD = "passes_filters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Filtered JSONL input.")
    parser.add_argument("--dataset", required=True, help="Canonical custom-dataset name.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/custom_datasets"),
        help="Directory containing canonical custom datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing original.jsonl and import manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count rows without writing output files.",
    )
    return parser.parse_args()


def parse_assessment(value: Any) -> tuple[str, dict[str, Any] | None]:
    """Return a fail-closed status and a normalized assessment object."""
    if value is None:
        return "missing_assessment", None
    if isinstance(value, dict):
        assessment = value
    elif isinstance(value, str):
        try:
            assessment = json.loads(value.strip())
        except (json.JSONDecodeError, TypeError):
            return "malformed_assessment", None
    else:
        return "malformed_assessment", None

    if not isinstance(assessment, dict):
        return "malformed_assessment", None
    decision = assessment.get("decision")
    substantial = assessment.get("contains_substantial_high_quality_section")
    if decision not in {"PASS", "FAIL"} or not isinstance(substantial, bool):
        return "malformed_assessment", assessment
    if (decision == "PASS") != substantial:
        return "inconsistent_assessment", assessment
    return ("passed" if decision == "PASS" else "failed"), assessment


def source_id(row: dict[str, Any], line_number: int) -> str:
    custom_id = row.get("custom_id")
    head_id = row.get("head_id")
    if custom_id is not None and head_id is not None and str(custom_id) != str(head_id):
        raise ValueError(
            f"custom_id and head_id disagree on input line {line_number}: "
            f"{custom_id!r} != {head_id!r}"
        )
    value = custom_id if custom_id is not None else head_id
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError(f"Missing or invalid custom_id/head_id on input line {line_number}")
    return str(value)


def canonical_row(
    row: dict[str, Any], assessment: dict[str, Any], line_number: int
) -> dict[str, Any]:
    identifier = source_id(row, line_number)
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing or empty text on passing input line {line_number}")
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"schema_version", "custom_id", "head_id", "text", FILTER_FIELD}
    }
    metadata["filter_provenance"] = {
        "input_line_number": line_number,
        "assessment": assessment,
    }
    return {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "custom_id": identifier,
        "text": text,
    }


def validate_destination(dataset_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_path = dataset_dir / "original.jsonl"
    manifest_path = dataset_dir / "filter_import_manifest.json"
    derived = [
        dataset_dir / "perturbations",
        dataset_dir / "scores",
        dataset_dir / "split_assignments.jsonl",
    ]
    existing_derived = [str(path) for path in derived if path.exists()]
    if existing_derived:
        raise FileExistsError(
            "Refusing to replace a dataset with derived artifacts: "
            + ", ".join(existing_derived)
        )
    if not overwrite:
        existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; use --overwrite after review: " + ", ".join(existing)
            )
    return output_path, manifest_path


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Filtered input does not exist: {input_path}")
    if not args.dataset or args.dataset in {".", ".."} or "/" in args.dataset or "\\" in args.dataset:
        raise ValueError("--dataset must be one non-empty path component")

    dataset_dir = args.dataset_root / args.dataset
    output_path = dataset_dir / "original.jsonl"
    manifest_path = dataset_dir / "filter_import_manifest.json"
    if not args.dry_run:
        output_path, manifest_path = validate_destination(dataset_dir, args.overwrite)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "input_rows": 0,
        "passed": 0,
        "failed": 0,
        "missing_assessment": 0,
        "malformed_assessment": 0,
        "inconsistent_assessment": 0,
    }
    seen_ids: set[str] = set()
    temp_name: str | None = None
    output_handle = None
    try:
        if not args.dry_run:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".tmp", dir=str(dataset_dir)
            )
            output_handle = os.fdopen(descriptor, "w", encoding="utf-8")
        with input_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid input JSON on line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object on input line {line_number}")
                status, assessment = parse_assessment(row.get(FILTER_FIELD))
                counts[status] += 1
                if status != "passed":
                    continue
                assert assessment is not None
                normalized = canonical_row(row, assessment, line_number)
                identifier = normalized["custom_id"]
                if identifier in seen_ids:
                    raise ValueError(f"Duplicate passing custom_id {identifier!r} on line {line_number}")
                seen_ids.add(identifier)
                if output_handle is not None:
                    encoded = (
                        json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True)
                        + "\n"
                    )
                    output_handle.write(encoded)
        if counts["passed"] == 0:
            raise ValueError("No rows passed the filter; refusing to create an empty dataset")
        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
            output_handle.close()
            output_handle = None
            assert temp_name is not None
            os.replace(temp_name, output_path)
            temp_name = None
    finally:
        if output_handle is not None:
            output_handle.close()
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    manifest = {
        "schema_version": 1,
        "operation": "import_vllm_filtered_custom_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset,
        "filter_field": FILTER_FIELD,
        "acceptance_rule": (
            'embedded JSON has decision="PASS" and '
            "contains_substantial_high_quality_section=true"
        ),
        "input": {"path": str(input_path)},
        "output": {
            "path": str(output_path.resolve()),
        },
        "counts": counts,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(convert(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
