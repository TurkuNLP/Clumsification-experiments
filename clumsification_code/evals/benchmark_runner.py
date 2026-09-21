# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import numpy as np

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
from clumsification_code.data.flattening import (
    flatten_pairwise_dataset,
    flatten_regression_dataset,
)
from clumsification_code.data.hf_dataset import load_formatted_dataset_dict
from clumsification_code.evals.metrics import (
    correlation_bundle,
    flatten_preference_metrics,
    preference_metrics,
)
from clumsification_code.fe.metrics import binary_metrics


class _SuiteScoreCache:
    """Score each distinct text/request once across labels in a suite run."""

    def __init__(self, model: TextScorer) -> None:
        self.model = model
        self._prompt_setter = getattr(model, "set_prompt_context", None)
        self._context: Any = None
        self._scores: Dict[tuple[Any, str], float] = {}

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        if callable(self._prompt_setter):
            self._prompt_setter(task_name, aspect)
            context_key = getattr(self.model, "score_cache_context", None)
            # Scorers with a fixed prompt can share scores between aspects;
            # otherwise the task/aspect pair conservatively identifies a request.
            self._context = (
                context_key() if callable(context_key) else (task_name, aspect)
            )

    def score_texts(
        self, texts: List[str], device=None, batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        keys = [(self._context, str(candidate)) for candidate in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self._scores))
        if missing:
            values = np.asarray(
                self.model.score_texts(
                    texts=[candidate for _, candidate in missing],
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
            )
            if values.shape != (len(missing),):
                raise ValueError("Scorer returned the wrong number of scores")
            self._scores.update(
                (key, float(value)) for key, value in zip(missing, values)
            )
        return np.asarray([self._scores[key] for key in keys])


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
        print(f"{name}: no valid preference pairs.", flush=True)
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
        f"tie rate={metrics['tie_rate']:.4f}",
        flush=True,
    )

    return metrics


def run_formatted_dataset_suite(
    *,
    model: TextScorer,
    device,
    dataset_path: str,
    split: str = "test",
    training_method: str = "regression",
    score_name: Optional[str] = None,
    pair_policy: str = "all_unequal_layers",
    batch_size: int,
    max_length: int,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one split of a saved formatted HF dataset."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("formatted dataset split must be one of: train, dev, test")
    if training_method not in {"regression", "pairwise", "binary"}:
        raise ValueError(
            "formatted dataset training method must be regression, pairwise, or binary"
        )
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")

    dataset_dict = load_formatted_dataset_dict(dataset_path)
    dataset = dataset_dict[split]
    if max_records is not None:
        dataset = dataset.select(range(min(max_records, len(dataset))))

    result_prefix = f"formatted__{Path(dataset_path).name}__{split}"

    if training_method == "regression":
        if not score_name:
            raise ValueError("score_name is required for formatted regression evaluation")
        flat = flatten_regression_dataset(dataset, score_name)
        result = score_scalar_aspect(
            model=model,
            device=device,
            texts=[str(text) for text in flat["text"]],
            labels=[float(label) for label in flat["label"]],
            task_name="formatted_dataset",
            aspect="regression",
            result_name=result_prefix,
            batch_size=batch_size,
            max_length=max_length,
            group_ids=[str(source_id) for source_id in flat["source_id"]],
        )
        result[f"{result_prefix}__n"] = len(flat)
        result[f"{result_prefix}__score_name"] = score_name
        return result

    if training_method == "pairwise":
        flat = flatten_pairwise_dataset(dataset, policy=pair_policy)
        metrics = eval_pairwise_preference_dataset(
            name=result_prefix,
            model=model,
            device=device,
            preferred_texts=[str(text) for text in flat["chosen_text"]],
            dispreferred_texts=[str(text) for text in flat["rejected_text"]],
            task_name="formatted_dataset",
            aspect="pairwise_quality",
            batch_size=batch_size,
            max_length=max_length,
            group_ids=[str(source_id) for source_id in flat["source_id"]],
        )
        result = flatten_preference_metrics(result_prefix, metrics)
        result[f"{result_prefix}__n"] = len(flat)
        result[f"{result_prefix}__pair_policy"] = pair_policy
        return result

    missing = {"text"} - set(dataset.column_names)
    if "label" not in dataset.column_names and "target" not in dataset.column_names:
        missing.add("label or target")
    if missing:
        raise ValueError(f"Binary formatted split is missing column(s): {sorted(missing)}")
    label_column = "label" if "label" in dataset.column_names else "target"
    predictions = model.score_texts(
        texts=[str(text) for text in dataset["text"]],
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    metrics = binary_metrics(
        SimpleNamespace(predictions=predictions, label_ids=dataset[label_column]),
        predictions_are_logits=False,
    )
    result = {f"{result_prefix}_{key}": value for key, value in metrics.items()}
    result[f"{result_prefix}__n"] = len(dataset)
    return result


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
    model = _SuiteScoreCache(model)
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
    model = _SuiteScoreCache(model)
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
    #    preferred, dispreferred = data.load_story_cloze_preference_pairs(split="eval")
    #    add_preference_result(
    #    "StoryCloze_eval_ending_preference",
    #    eval_pairwise_preference_dataset(
    #        name="StoryCloze_eval_ending_preference",
    #        model=model,
    #        device=device,
    #        preferred_texts=preferred,
    #        dispreferred_texts=dispreferred,
    #        task_name="story_cloze",
    #        aspect="coherence",
    #        batch_size=batch_size,
    #        max_length=max_length,
    #    ),
    #)

    # One pass over NLG-eval is substantially cheaper than rescanning the large
    # JSONL file once for every benchmark/aspect specification.
    for record in iter_nlg_eval_records(path=nlg_eval_path, specs=specs):
        records = spec_records[str(record["spec_name"])]
        if max_records_per_dimension is None or len(records) < max_records_per_dimension:
            records.append(record)

    for spec in specs:
        records = spec_records[spec.name]
        if not records:
            print(f"{spec.name}: no valid records", flush=True)
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
