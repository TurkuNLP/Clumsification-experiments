# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Audit released UniEval pseudo-data and write a reproducibility manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.unieval.data import build_unieval_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/unprocessed_datasets/unieval_pseudo_data",
        help="Directory containing UniEval JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="data/unprocessed_datasets/unieval_pseudo_data.audit.json",
        help="Manifest path; source data is never modified.",
    )
    args = parser.parse_args()
    manifest = build_unieval_manifest(args.data_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    totals = {"rows": 0, "yes": 0, "no": 0, "conflicts": 0}
    for item in manifest["files"]:
        totals["rows"] += item["rows"]
        totals["yes"] += item["labels"]["Yes"]
        totals["no"] += item["labels"]["No"]
        totals["conflicts"] += item["conflicting_src"]
    print(f"Audited {len(manifest['files'])} files / {totals['rows']} rows")
    print(f"Labels: Yes={totals['yes']} No={totals['no']}; conflicting inputs={totals['conflicts']}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
