# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

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
    DEFAULT_ARGESSAY_PATH,
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_ELLIPSE_PATH,
    iter_standalone_records,
)
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
) -> Dict[str, Any]:
    maybe_set_prompt_context(model, task_name, aspect)

    preds = model.score_texts(
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    return correlation_bundle(labels, preds, result_name)


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
) -> Optional[Dict[str, Any]]:
    if len(preferred_texts) != len(dispreferred_texts):
        raise ValueError(
            f"{name}: preferred/dispreferred length mismatch: "
            f"{len(preferred_texts)} vs {len(dispreferred_texts)}"
        )
    if human_ties is not None and len(human_ties) != len(preferred_texts):
        raise ValueError(f"{name}: human tie mask length mismatch")

    clean_text = getattr(data, "clean_text", None)
    if not callable(clean_text):
        clean_text = getattr(data, "_clean_text")

    pairs = []
    for p, d in zip(preferred_texts, dispreferred_texts):
        p = clean_text(p)
        d = clean_text(d)
        if p and d:
            pairs.append((p, d))

    if not pairs:
        print(f"{name}: no valid preference pairs.")
        return None

    preferred_texts = [p for p, _ in pairs]
    dispreferred_texts = [d for _, d in pairs]

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
    )

    print(
        f"  {name}: n={metrics['n']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"tie rate={metrics['tie_rate']:.4f}"
    )

    return metrics


def run_legacy_benchmark_suite(
    *,
    model: TextScorer,
    device,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    """Shared benchmark suite for FE, GPTScore, MetricX, and G-Eval adapters."""

    all_results: Dict[str, Any] = {}

    def update_preference_results(bench_name: str, metrics: Optional[Dict[str, Any]]) -> None:
        if metrics is not None:
            all_results.update(flatten_preference_metrics(bench_name, metrics))

    print("=" * 60)
    print("Preference-style HF benchmarks")
    print("=" * 60)

    # JFLEG
    # Current loader returns: preferred, dispreferred
    preferred, dispreferred = data.load_jfleg_preference_pairs(split="test")
    bench_name = "JFLEG_test_correction_preference"

    metrics = eval_pairwise_preference_dataset(
        name=bench_name,
        model=model,
        device=device,
        preferred_texts=preferred,
        dispreferred_texts=dispreferred,
        task_name="jfleg",
        aspect="grammar",
        batch_size=batch_size,
        max_length=max_length,
    )
    update_preference_results(bench_name, metrics)

    # MultiBLiMP
    # Current loader returns: preferred, dispreferred
    preferred, dispreferred = data.load_multiblimp_english_preference_pairs()
    bench_name = "MultiBLiMP_eng_minimal_pair_preference"

    metrics = eval_pairwise_preference_dataset(
        name=bench_name,
        model=model,
        device=device,
        preferred_texts=preferred,
        dispreferred_texts=dispreferred,
        task_name="multiblimp",
        aspect="acceptability",
        batch_size=batch_size,
        max_length=max_length,
    )
    update_preference_results(bench_name, metrics)

    # Story Cloze
    # Current loader returns: preferred, dispreferred
    preferred, dispreferred = data.load_story_cloze_preference_pairs(split="eval")
    bench_name = "StoryCloze_eval_ending_preference"

    metrics = eval_pairwise_preference_dataset(
        name=bench_name,
        model=model,
        device=device,
        preferred_texts=preferred,
        dispreferred_texts=dispreferred,
        task_name="story_cloze",
        aspect="coherence",
        batch_size=batch_size,
        max_length=max_length,
    )
    update_preference_results(bench_name, metrics)

    print()
    print("=" * 60)
    print("Scalar human-score benchmarks")
    print("=" * 60)

    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]

    for aspect in ["fluency", "coherence", "consistency"]:
        labels = [x for y in ds[aspect] for x in y]
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=summeval_texts,
                labels=labels,
                task_name="summeval",
                aspect=aspect,
                result_name=f"summeval_{aspect}",
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    ellipse_ds = data.load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    for aspect, result_name, label_key in [
        ("overall", "ellipse_overall", "overall"),
        ("cohesion", "ellipse_cohesion", "cohesion"),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=ellipse_ds["text"],
                labels=[float(x) for x in ellipse_ds[label_key]],
                task_name="ellipse",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # USR - Topical Chat
    tc_texts, tc_overall_labels = data.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Overall",
    )
    _, tc_natural_labels = data.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Natural",
    )

    for aspect, result_name, labels in [
        ("overall", "tc_overall", tc_overall_labels),
        ("natural", "tc_natural", tc_natural_labels),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=tc_texts,
                labels=labels,
                task_name="usr_topical_chat",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # USR - Persona Chat
    pc_texts, pc_overall_labels = data.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Overall",
    )
    _, pc_natural_labels = data.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Natural",
    )

    for aspect, result_name, labels in [
        ("overall", "pc_overall", pc_overall_labels),
        ("natural", "pc_natural", pc_natural_labels),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=pc_texts,
                labels=labels,
                task_name="usr_persona_chat",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # OpenMEVA
    meva_texts_roc, meva_labels_roc = data.load_data_openmeva(
        "data/benchmarks/mans_roc.json"
    )
    meva_texts_wp, meva_labels_wp = data.load_data_openmeva(
        "data/benchmarks/mans_wp.json"
    )

    all_results.update(
        score_scalar_aspect(
            model=model,
            device=device,
            texts=meva_texts_roc + meva_texts_wp,
            labels=meva_labels_roc + meva_labels_wp,
            task_name="openmeva",
            aspect="overall",
            result_name="OpenMEVA_overall",
            batch_size=batch_size,
            max_length=max_length,
        )
    )

    # WebNLG
    webnlg_texts, webnlg_labels = data.load_data_webnlg(
        "data/benchmarks/web_nlg_2020_human_evals_en.json"
    )

    all_results.update(
        score_scalar_aspect(
            model=model,
            device=device,
            texts=webnlg_texts,
            labels=[float(x) for x in webnlg_labels],
            task_name="webnlg",
            aspect="fluency",
            result_name="WebNLG_fluency",
            batch_size=batch_size,
            max_length=max_length,
        )
    )

    # HANNA
    hanna_ds = data.load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")

    for aspect, result_name, label_key in [
        ("coherence", "HANNA_coherence", "coherence"),
        ("complexity", "HANNA_complexity", "complexity"),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=hanna_ds["text"],
                labels=[float(x) for x in hanna_ds[label_key]],
                task_name="hanna",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # ARG-ESSAY
    arge_ds = data.load_argessay_data("data/benchmarks/arg-essay.csv")

    for aspect, result_name, label_key in [
        (
            "language_mastery",
            "ARG-ESSAY_language_mastery",
            "language_mastery",
        ),
        ("complexity", "ARG-ESSAY_complexity", "complexity"),
        ("vocabulary", "ARG-ESSAY_vocabulary", "vocabulary"),
        (
            "language_constructs",
            "ARG-ESSAY_language_constructs",
            "language_constructs",
        ),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=arge_ds["text"],
                labels=[float(x) for x in arge_ds[label_key]],
                task_name="argessay",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # Human Ratings of NLG
    hr_ds = data.load_human_ratings_of_nlg_data(
        "data/benchmarks/human_ratings_of_nlg.csv"
    )

    for aspect, result_name, label_key in [
        ("quality", "HumanRatings_quality", "quality"),
        ("naturalness", "HumanRatings_naturalness", "naturalness"),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=hr_ds["text"],
                labels=[float(x) for x in hr_ds[label_key]],
                task_name="human_ratings_of_nlg",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    # FED
    turn_ds, whole_ds = data.load_fed_data("data/benchmarks/fed_data.json")

    for aspect, result_name, label_key in [
        ("fluency", "FED_turn_fluency", "fluent"),
        ("overall", "FED_turn_overall", "overall"),
    ]:
        all_results.update(
            score_scalar_aspect(
                model=model,
                device=device,
                texts=turn_ds["text"],
                labels=[float(x) for x in turn_ds[label_key]],
                task_name="fed_turn",
                aspect=aspect,
                result_name=result_name,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    all_results.update(
        score_scalar_aspect(
            model=model,
            device=device,
            texts=whole_ds["text"],
            labels=[float(x) for x in whole_ds["overall"]],
            task_name="fed_whole_dialogue",
            aspect="overall",
            result_name="FED_whole_overall",
            batch_size=batch_size,
            max_length=max_length,
        )
    )

    return all_results


def run_standard_benchmark_suite(
    *,
    model: TextScorer,
    device,
    batch_size: int,
    max_length: int,
    nlg_eval_path=DEFAULT_NLG_EVAL_PATH,
    ellipse_path=DEFAULT_ELLIPSE_PATH,
    argessay_path=DEFAULT_ARGESSAY_PATH,
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
            argessay_path=argessay_path,
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
