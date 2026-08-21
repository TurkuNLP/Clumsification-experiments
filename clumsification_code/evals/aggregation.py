# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Aggregation of per-dimension benchmark correlations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping


def _finite_mean(values: Iterable[object]) -> float:
    """Return a macro-average while ignoring undefined correlations."""
    finite = []
    for value in values:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return sum(finite) / len(finite) if finite else float("nan")


def _add_group_result(
    output: Dict[str, Any],
    prefix: str,
    group: str,
    summaries: List[Mapping[str, Any]],
) -> None:
    """Add both rank statistics and the contributing-dimension count."""
    safe_group = str(group).replace(" ", "_")
    output[f"{prefix}__{safe_group}__spearman_rho"] = _finite_mean(
        item.get("spearman_rho") for item in summaries
    )
    output[f"{prefix}__{safe_group}__kendall_tau"] = _finite_mean(
        item.get("kendall_tau") for item in summaries
    )
    output[f"{prefix}__{safe_group}__n_dimensions"] = len(summaries)


def aggregate_dimension_results(
    summaries: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Macro-average dimensions by task family, category, and overall.

    A multi-construct dimension is intentionally included in every category
    listed by the registry.  This makes category results interpretable, while
    the overall result first averages the category means so larger categories
    cannot dominate it.
    """
    items = list(summaries)
    output: Dict[str, Any] = {}

    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[str(item["task_family"])].append(item)
        for category in item.get("categories", ()):
            by_category[str(category)].append(item)

    for task_family, group in sorted(by_task.items()):
        _add_group_result(output, "aggregate__task_family", task_family, group)
    for category, group in sorted(by_category.items()):
        _add_group_result(output, "aggregate__category", category, group)

    category_spearman = [
        output[f"aggregate__category__{category.replace(' ', '_')}__spearman_rho"]
        for category in sorted(by_category)
    ]
    category_kendall = [
        output[f"aggregate__category__{category.replace(' ', '_')}__kendall_tau"]
        for category in sorted(by_category)
    ]
    output["aggregate__overall__spearman_rho"] = _finite_mean(category_spearman)
    output["aggregate__overall__kendall_tau"] = _finite_mean(category_kendall)
    output["aggregate__overall__n_categories"] = len(by_category)
    output["aggregate__overall__n_dimensions"] = len(items)
    return output

