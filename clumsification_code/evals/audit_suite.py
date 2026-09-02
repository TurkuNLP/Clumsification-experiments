# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Audit the configured English scalar evaluation suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from .benchmark_registry import get_nlg_eval_specs
from .nlg_eval_loader import (
    DEFAULT_NLG_EVAL_METADATA_PATH,
    DEFAULT_NLG_EVAL_PATH,
    iter_nlg_eval_records,
    load_nlg_eval_metadata,
)
from .standalone_benchmarks import iter_standalone_records
from .standalone_benchmarks import (
    DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_ELLIPSE_PATH,
    DEFAULT_MTEB_SUMMEVAL_DATASET,
)


DEFAULT_MANIFEST_PATH = Path("data/benchmarks/english_fluency_suite_manifest.json")

# These are deliberate research exclusions, not loader failures. Keeping them
# in the manifest makes it possible to audit why a benchmark is absent.
EXCLUDED_BENCHMARKS = {
    "FUDGE": "Excluded by the project scope.",
    "dialogue_response": "Dialogue responses are a separate register and task.",
    "LENS": "Overall Quality combines fluency with meaning preservation and simplification.",
    "BAGEL/SFRES/SFHOT": "Excluded to preserve Themis as a possible direct baseline.",
}


def build_audit_report(
    *,
    nlg_eval_path: Path = DEFAULT_NLG_EVAL_PATH,
    metadata_path: Path = DEFAULT_NLG_EVAL_METADATA_PATH,
    ellipse_path: Path | None = DEFAULT_ELLIPSE_PATH,
    human_chatgpt_essays_path: Path | None = DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
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
        selected_counts[str(record["spec_name"])] += 1
        for category in record["fluency_categories"]:
            category_counts[category] += 1
        task_counts[record["task_family"]] += 1

    standalone_counts = Counter()
    for record in iter_standalone_records(
        ellipse_path=ellipse_path,
        human_chatgpt_essays_path=human_chatgpt_essays_path,
        cohesentia_path=cohesentia_path,
    ):
        standalone_counts[str(record["benchmark"])] += 1
        for category in record["fluency_categories"]:
            category_counts[category] += 1
        task_counts[record["task_family"]] += 1

    included_specs = []
    for spec in specs:
        included_specs.append(
            {
                "name": spec.name,
                "source": "nlg_eval",
                "benchmark": spec.benchmark,
                "aspect": spec.aspect,
                "task_family": spec.task_family,
                "fluency_categories": list(spec.categories),
                "task_filter": spec.task,
                "original_data_filter": spec.original_data,
                "record_count": selected_counts.get(spec.name, 0),
                "label_type": "scalar",
            }
        )

    return {
        "manifest_version": 1,
        "suite": "english_fluency",
        "source_paths": {
            "nlg_eval": str(nlg_eval_path),
            "nlg_eval_metadata": str(metadata_path),
            "ellipse": str(ellipse_path) if ellipse_path else None,
            "human_chatgpt_essays": (
                str(human_chatgpt_essays_path)
                if human_chatgpt_essays_path
                else None
            ),
            "cohesentia": str(cohesentia_path) if cohesentia_path else None,
            "mteb_summeval": DEFAULT_MTEB_SUMMEVAL_DATASET,
        },
        "metadata_rows": len(metadata),
        "metadata_retained_records": sum(metadata_by_benchmark.values()),
        "registry_specs": len(specs),
        "nlg_eval_selected_records": sum(selected_counts.values()),
        "included_nlg_eval_specs": included_specs,
        "standalone_records": dict(sorted(standalone_counts.items())),
        "category_records": dict(sorted(category_counts.items())),
        "task_family_records": dict(sorted(task_counts.items())),
        "excluded_benchmarks": EXCLUDED_BENCHMARKS,
        "preference_datasets": {
            "JFLEG": "separate preference adapter",
            "MultiBLiMP English": "separate preference adapter",
            "Story Cloze": "secondary coherence diagnostic",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlg-eval-path", type=Path, default=DEFAULT_NLG_EVAL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_NLG_EVAL_METADATA_PATH)
    parser.add_argument("--ellipse-path", type=Path, default=DEFAULT_ELLIPSE_PATH)
    parser.add_argument(
        "--human-chatgpt-essays-path",
        "--argessay-path",
        dest="human_chatgpt_essays_path",
        type=Path,
        default=DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
    )
    parser.add_argument("--cohesentia-path", type=Path, default=DEFAULT_COHESENTIA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
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
