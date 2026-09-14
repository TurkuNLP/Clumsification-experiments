# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets

from clumsification_code.evals import benchmark_data as data
from clumsification_code.evals.aggregation import aggregate_dimension_results
from clumsification_code.evals.benchmark_registry import get_nlg_eval_specs
from clumsification_code.evals.nlg_eval_loader import (
    DEFAULT_NLG_EVAL_PATH,
    iter_nlg_eval_records,
)
from clumsification_code.evals.standalone_benchmarks import (
    DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_COHESENTIA_TRAIN_PATH,
    DEFAULT_ELLIPSE_PATH,
    DEFAULT_ELLIPSE_TRAIN_PATH,
    iter_cohesentia_records,
    iter_ellipse_records,
    iter_standalone_records,
)
from clumsification_code.evals.external_dev import audit_external_dev_splits
from clumsification_code.evals.inference.base import TextScorer
from clumsification_code.evals.multilingual_benchmarks import (
    iter_basse_records,
    iter_norwegian_preference_records,
)
from clumsification_code.evals.metrics import (
    correlation_bundle,
    flatten_preference_metrics,
    preference_metrics,
)


def maybe_set_prompt_context(model: TextScorer, task_name: str, aspect: str) -> None:
    setter = getattr(model, "set_prompt_context", None)
    if callable(setter):
        setter(task_name, aspect)


def score_scalar_aspect(
    *,
    model: TextScorer,
    device,
    texts: List[str],
    labels: List[float],
    task_name: str,
    aspect: str,
    result_name: str,
    batch_size: int,
    max_length: int,
    group_ids: Optional[List[str]] = None,
    bootstrap_samples: int = 0,
) -> Dict[str, Any]:
    maybe_set_prompt_context(model, task_name, aspect)

    preds = model.score_texts(
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    return correlation_bundle(
        labels,
        preds,
        result_name,
        group_ids=group_ids,
        bootstrap_samples=bootstrap_samples,
    )


def eval_pairwise_preference_dataset(
    *,
    name: str,
    model: TextScorer,
    device,
    preferred_texts: List[str],
    dispreferred_texts: List[str],
    task_name: str,
    aspect: str,
    batch_size: int,
    max_length: int,
    human_ties: Optional[List[bool]] = None,
    group_ids: Optional[List[str]] = None,
    bootstrap_samples: int = 0,
) -> Optional[Dict[str, Any]]:
    if len(preferred_texts) != len(dispreferred_texts):
        raise ValueError(
            f"{name}: preferred/dispreferred length mismatch: "
            f"{len(preferred_texts)} vs {len(dispreferred_texts)}"
        )
    if human_ties is not None and len(human_ties) != len(preferred_texts):
        raise ValueError(f"{name}: human tie mask length mismatch")
    if group_ids is not None and len(group_ids) != len(preferred_texts):
        raise ValueError(f"{name}: group_ids length mismatch")

    clean_text = getattr(data, "clean_text", None)
    if not callable(clean_text):
        clean_text = getattr(data, "_clean_text")

    pairs = []
    tie_values = human_ties if human_ties is not None else [False] * len(preferred_texts)
    group_values = group_ids if group_ids is not None else [None] * len(preferred_texts)
    for p, d, tie, group_id in zip(
        preferred_texts, dispreferred_texts, tie_values, group_values
    ):
        p = clean_text(p)
        d = clean_text(d)
        if p and d:
            pairs.append((p, d, bool(tie), group_id))

    if not pairs:
        print(f"{name}: no valid preference pairs.")
        return None

    preferred_texts = [p for p, _, _, _ in pairs]
    dispreferred_texts = [d for _, d, _, _ in pairs]
    human_ties = [tie for _, _, tie, _ in pairs]
    retained_group_ids = [group_id for _, _, _, group_id in pairs]

    maybe_set_prompt_context(model, task_name, aspect)

    n = len(preferred_texts)
    all_scores = model.score_texts(
        texts=preferred_texts + dispreferred_texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    metrics = preference_metrics(
        preferred_scores=all_scores[:n],
        dispreferred_scores=all_scores[n:],
        human_ties=human_ties,
        name=name,
        group_ids=retained_group_ids if group_ids is not None else None,
        bootstrap_samples=bootstrap_samples,
    )

    print(
        f"  {name}: n={metrics['n']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"tie rate={metrics['tie_rate']:.4f}"
    )

    return metrics


def run_external_dev_suite(
    *,
    model: TextScorer,
    device,
    batch_size: int,
    max_length: int,
    ellipse_path=DEFAULT_ELLIPSE_TRAIN_PATH,
    cohesentia_path=DEFAULT_COHESENTIA_TRAIN_PATH,
    include_story_cloze_diagnostic: bool = False,
    max_records_per_dimension: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the human-labeled, non-test checkpoint-selection panel.

    This route is deliberately separate from ``run_standard_benchmark_suite``.
    It scores only author-defined non-test data and never invokes a final-suite
    scoring loader. The provenance audit reads frozen final identifiers and
    checksums only to enforce disjointness. Story Cloze train is optional and
    diagnostic-only.
    """
    if max_records_per_dimension is not None and max_records_per_dimension < 1:
        raise ValueError("max_records_per_dimension must be positive when provided")
    if Path(ellipse_path).resolve() != DEFAULT_ELLIPSE_TRAIN_PATH.resolve():
        raise ValueError(
            "External development requires the audited ELLIPSE train path: "
            f"{DEFAULT_ELLIPSE_TRAIN_PATH}"
        )
    if Path(cohesentia_path).resolve() != DEFAULT_COHESENTIA_TRAIN_PATH.resolve():
        raise ValueError(
            "External development requires the audited CoheSentia train path: "
            f"{DEFAULT_COHESENTIA_TRAIN_PATH}"
        )

    audit_report = audit_external_dev_splits()
    results: Dict[str, Any] = {
        "external_dev__provenance": audit_report,
        "external_dev__selection_datasets": ["ELLIPSE", "JFLEG", "CoheSentia"],
    }

    scalar_groups = {
        "ELLIPSE_train": list(
            iter_ellipse_records(ellipse_path, split_name="train")
        ),
        "CoheSentia_train": list(
            iter_cohesentia_records(cohesentia_path, split_name="train")
        ),
    }
    for dataset_name, dataset_records in scalar_groups.items():
        by_aspect: Dict[str, List[Dict[str, Any]]] = {}
        for record in dataset_records:
            by_aspect.setdefault(str(record["aspect"]), []).append(record)
        for aspect, records in by_aspect.items():
            if max_records_per_dimension is not None:
                records = records[:max_records_per_dimension]
            result_name = f"external_dev__{dataset_name}__{aspect}"
            results.update(
                score_scalar_aspect(
                    model=model,
                    device=device,
                    texts=[str(record["text"]) for record in records],
                    labels=[float(record["human_score"]) for record in records],
                    task_name=str(records[0]["task_family"]),
                    aspect=aspect,
                    result_name=result_name,
                    batch_size=batch_size,
                    max_length=max_length,
                    group_ids=[str(record["source_id"]) for record in records],
                    bootstrap_samples=1000,
                )
            )
            results[f"{result_name}__n_sources"] = len(
                {str(record["source_id"]) for record in records}
            )

    jfleg_records = data.load_jfleg_preference_records(split="validation")
    jfleg_sources = {str(record["source_id"]) for record in jfleg_records}
    if len(jfleg_records) != 2593 or len(jfleg_sources) != 719:
        raise ValueError(
            "Pinned JFLEG validation split changed after filtering: expected "
            f"2593 pairs from 719 sources, observed {len(jfleg_records)} pairs "
            f"from {len(jfleg_sources)} sources"
        )
    if max_records_per_dimension is not None:
        jfleg_records = jfleg_records[:max_records_per_dimension]
    jfleg_name = "external_dev__JFLEG_validation__correction_preference"
    jfleg_metrics = eval_pairwise_preference_dataset(
        name=jfleg_name,
        model=model,
        device=device,
        preferred_texts=[str(record["preferred_text"]) for record in jfleg_records],
        dispreferred_texts=[str(record["dispreferred_text"]) for record in jfleg_records],
        task_name="jfleg",
        aspect="grammar",
        batch_size=batch_size,
        max_length=max_length,
        group_ids=[str(record["source_id"]) for record in jfleg_records],
        bootstrap_samples=1000,
    )
    results.update(flatten_preference_metrics(jfleg_name, jfleg_metrics))
    results[f"{jfleg_name}__n_sources"] = len(
        {str(record["source_id"]) for record in jfleg_records}
    )
    results[f"{jfleg_name}__revision"] = data.JFLEG_REVISION

    if include_story_cloze_diagnostic:
        preferred, dispreferred = data.load_story_cloze_preference_pairs(split="train")
        if max_records_per_dimension is not None:
            preferred = preferred[:max_records_per_dimension]
            dispreferred = dispreferred[:max_records_per_dimension]
        diagnostic_name = "diagnostic__StoryCloze_train__ending_preference"
        diagnostic_metrics = eval_pairwise_preference_dataset(
            name=diagnostic_name,
            model=model,
            device=device,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            task_name="story_cloze",
            aspect="coherence",
            batch_size=batch_size,
            max_length=max_length,
        )
        results.update(
            flatten_preference_metrics(diagnostic_name, diagnostic_metrics)
        )

    return results


def run_standard_benchmark_suite(
    *,
    model: TextScorer,
    device,
    batch_size: int,
    max_length: int,
    nlg_eval_path=DEFAULT_NLG_EVAL_PATH,
    ellipse_path=DEFAULT_ELLIPSE_PATH,
    human_chatgpt_essays_path=DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
    cohesentia_path=DEFAULT_COHESENTIA_PATH,
    skip_preferences: bool = False,
    max_records_per_dimension: Optional[int] = None,
    include_multilingual: bool = True,
) -> Dict[str, Any]:
    """Run the registry-backed English scalar and preference suite.

    Dataset selection is now defined by the benchmark registry.  Records are
    materialized only after streaming filters have reduced the 3.3 GB NLG-eval
    file to the selected dimensions.
    """
    if max_records_per_dimension is not None and max_records_per_dimension < 1:
        raise ValueError("max_records_per_dimension must be positive when provided")
    all_results: Dict[str, Any] = {}
    dimension_summaries: List[Dict[str, Any]] = []
    specs = get_nlg_eval_specs()
    spec_records: Dict[str, List[Dict[str, Any]]] = {spec.name: [] for spec in specs}

    # Preference benchmarks are intentionally kept outside scalar aggregation.
    # They measure pairwise ordering and therefore report tie-aware and strict
    # accuracy instead of Spearman/Kendall correlations.
    def add_preference_result(name: str, metrics: Optional[Dict[str, Any]]) -> None:
        if metrics is not None:
            all_results.update(flatten_preference_metrics(name, metrics))

    if not skip_preferences:
        preferred, dispreferred = data.load_jfleg_preference_pairs(split="test")
        add_preference_result(
        "JFLEG_test_correction_preference",
        eval_pairwise_preference_dataset(
            name="JFLEG_test_correction_preference",
            model=model,
            device=device,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            task_name="jfleg",
            aspect="grammar",
            batch_size=batch_size,
            max_length=max_length,
        ),
    )

        preferred, dispreferred = data.load_multiblimp_english_preference_pairs()
        add_preference_result(
        "MultiBLiMP_eng_minimal_pair_preference",
        eval_pairwise_preference_dataset(
            name="MultiBLiMP_eng_minimal_pair_preference",
            model=model,
            device=device,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            task_name="multiblimp",
            aspect="acceptability",
            batch_size=batch_size,
            max_length=max_length,
        ),
    )

        # Story Cloze is retained as a secondary coherence diagnostic because its
        # labels also depend on commonsense plausibility.
        preferred, dispreferred = data.load_story_cloze_preference_pairs(split="eval")
        add_preference_result(
        "StoryCloze_eval_ending_preference",
        eval_pairwise_preference_dataset(
            name="StoryCloze_eval_ending_preference",
            model=model,
            device=device,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            task_name="story_cloze",
            aspect="coherence",
            batch_size=batch_size,
            max_length=max_length,
        ),
    )

    # One pass over NLG-eval is substantially cheaper than rescanning the large
    # JSONL file once for every benchmark/aspect specification.
    for record in iter_nlg_eval_records(path=nlg_eval_path, specs=specs):
        records = spec_records[str(record["spec_name"])]
        if max_records_per_dimension is None or len(records) < max_records_per_dimension:
            records.append(record)

    for spec in specs:
        records = spec_records[spec.name]
        if not records:
            print(f"{spec.name}: no valid records")
            continue
        labels = [float(record["human_score"]) for record in records]
        texts = [str(record["text"]) for record in records]
        result_name = f"{spec.name}"
        dimension_result = score_scalar_aspect(
                model=model,
                device=device,
                texts=texts,
                labels=labels,
                task_name=spec.task_family,
                aspect=spec.aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        all_results.update(dimension_result)
        dimension_summaries.append(
            {
                "name": result_name,
                "task_family": spec.task_family,
                "categories": spec.categories,
                "spearman_rho": dimension_result.get(f"{result_name}_spearman_rho"),
                "kendall_tau": dimension_result.get(f"{result_name}_kendall_tau"),
            }
        )

    # These sources intentionally remain outside NLG-eval but expose the same
    # normalized record fields and therefore use the same scoring helper. This
    # includes original MTEB SummEval, which stays distinct from SummEval-OP.
    standalone_records = list(
        iter_standalone_records(
            ellipse_path=ellipse_path,
            human_chatgpt_essays_path=human_chatgpt_essays_path,
            cohesentia_path=cohesentia_path,
        )
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in standalone_records:
        key = f"{record['benchmark']}__{record['aspect']}"
        records = grouped.setdefault(key, [])
        if max_records_per_dimension is None or len(records) < max_records_per_dimension:
            records.append(record)

    for group_name, records in grouped.items():
        labels = [float(record["human_score"]) for record in records]
        texts = [str(record["text"]) for record in records]
        result_name = group_name
        dimension_result = score_scalar_aspect(
                model=model,
                device=device,
                texts=texts,
                labels=labels,
                task_name=str(records[0]["task_family"]),
                aspect=str(records[0]["aspect"]),
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        all_results.update(dimension_result)
        dimension_summaries.append(
            {
                "name": result_name,
                "task_family": records[0]["task_family"],
                "categories": records[0]["fluency_categories"],
                "spearman_rho": dimension_result.get(f"{result_name}_spearman_rho"),
                "kendall_tau": dimension_result.get(f"{result_name}_kendall_tau"),
            }
        )

    multilingual_summaries: List[Dict[str, Any]] = []
    if include_multilingual:
        for language_code in ("eu", "es"):
            records = list(iter_basse_records(language=language_code))
            if max_records_per_dimension is not None:
                records = records[:max_records_per_dimension]
            if not records:
                continue
            language_name = str(records[0]["language"]).lower()
            result_name = f"multilingual__basse__{language_name}__grammaticality"
            dimension_result = score_scalar_aspect(
                model=model,
                device=device,
                texts=[str(record["text"]) for record in records],
                labels=[float(record["human_score"]) for record in records],
                task_name="summarization",
                aspect="fluency",
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
            all_results.update(dimension_result)
            multilingual_summaries.append({
                "name": result_name,
                "task_family": "summarization",
                "categories": ("grammaticality",),
                "language": language_name,
                "track": "multilingual",
                "spearman_rho": dimension_result.get(f"{result_name}_spearman_rho"),
                "kendall_tau": dimension_result.get(f"{result_name}_kendall_tau"),
            })

        norwegian = list(iter_norwegian_preference_records())
        if max_records_per_dimension is not None:
            norwegian = norwegian[:max_records_per_dimension]
        if norwegian:
            result_name = "multilingual__norwegian__holistic_fluency_preference"
            metrics = eval_pairwise_preference_dataset(
                name=result_name,
                model=model,
                device=device,
                preferred_texts=[str(record["preferred_text"]) for record in norwegian],
                dispreferred_texts=[str(record["dispreferred_text"]) for record in norwegian],
                human_ties=[bool(record["tie"]) for record in norwegian],
                task_name="general_text_generation",
                aspect="fluency",
                batch_size=batch_size,
                max_length=max_length,
            )
            add_preference_result(result_name, metrics)

        all_results.update(aggregate_dimension_results(
            multilingual_summaries,
            prefix="aggregate__multilingual",
        ))

    all_results.update(aggregate_dimension_results(dimension_summaries))
    return all_results
