# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Audit the configured English scalar evaluation suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

from .benchmark_registry import get_nlg_eval_specs
from .nlg_eval_loader import (
    DEFAULT_NLG_EVAL_METADATA_PATH,
    DEFAULT_NLG_EVAL_PATH,
    iter_nlg_eval_records,
    load_nlg_eval_metadata,
)
from .standalone_benchmarks import iter_standalone_records
from .standalone_benchmarks import (
    DEFAULT_ARGESSAY_PATH,
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_ELLIPSE_PATH,
)


def build_audit_report(
    *,
    nlg_eval_path: Path = DEFAULT_NLG_EVAL_PATH,
    metadata_path: Path = DEFAULT_NLG_EVAL_METADATA_PATH,
    ellipse_path: Path | None = DEFAULT_ELLIPSE_PATH,
    argessay_path: Path | None = DEFAULT_ARGESSAY_PATH,
    cohesentia_path: Path | None = DEFAULT_COHESENTIA_PATH,
) -> Dict[str, object]:
    """Build a machine-readable count and mapping report."""
    metadata = load_nlg_eval_metadata(metadata_path)
    specs = get_nlg_eval_specs()
    metadata_by_benchmark = Counter({})
    for row in metadata:
        metadata_by_benchmark[row["benchmark"]] += int(row["retained_volume"])

    selected_counts = Counter()
    category_counts = Counter()
    task_counts = Counter()
    for record in iter_nlg_eval_records(nlg_eval_path, specs):
        selected_counts[record["metadata_aspect"] + "::" + str(record["benchmark"])] += 1
        for category in record["fluency_categories"]:
            category_counts[category] += 1
        task_counts[record["task_family"]] += 1

    standalone_counts = Counter()
    for record in iter_standalone_records(
        ellipse_path=ellipse_path,
        argessay_path=argessay_path,
        cohesentia_path=cohesentia_path,
    ):
        standalone_counts[str(record["benchmark"])] += 1
        for category in record["fluency_categories"]:
            category_counts[category] += 1
        task_counts[record["task_family"]] += 1

    return {
        "metadata_rows": len(metadata),
        "metadata_retained_records": sum(metadata_by_benchmark.values()),
        "registry_specs": len(specs),
        "nlg_eval_selected_records": sum(selected_counts.values()),
        "nlg_eval_selected_by_dimension": dict(sorted(selected_counts.items())),
        "standalone_records": dict(sorted(standalone_counts.items())),
        "category_records": dict(sorted(category_counts.items())),
        "task_family_records": dict(sorted(task_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlg-eval-path", type=Path, default=DEFAULT_NLG_EVAL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_NLG_EVAL_METADATA_PATH)
    parser.add_argument("--ellipse-path", type=Path, default=DEFAULT_ELLIPSE_PATH)
    parser.add_argument("--argessay-path", type=Path, default=DEFAULT_ARGESSAY_PATH)
    parser.add_argument("--cohesentia-path", type=Path, default=DEFAULT_COHESENTIA_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    options = vars(args).copy()
    options.pop("output")
    report = build_audit_report(**options)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
