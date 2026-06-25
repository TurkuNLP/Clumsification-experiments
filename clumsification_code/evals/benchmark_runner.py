from __future__ import annotations

from typing import Any, Dict, List, Optional

import datasets

from clumsification_code.evals import benchmark_data as data
from clumsification_code.evals.inference.base import TextScorer
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
) -> Optional[Dict[str, Any]]:
    if len(preferred_texts) != len(dispreferred_texts):
        raise ValueError(
            f"{name}: preferred/dispreferred length mismatch: "
            f"{len(preferred_texts)} vs {len(dispreferred_texts)}"
        )

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
        name=name,
    )

    print(
        f"  {name}: n={metrics['n']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"tie rate={metrics['tie_rate']:.4f}"
    )

    return metrics


def run_standard_benchmark_suite(
    *,
    model: TextScorer,
    device,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    """Shared benchmark suite for LTR, GPTScore, MetricX, and G-Eval adapters."""

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