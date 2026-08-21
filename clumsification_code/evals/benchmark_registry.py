# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical benchmark/aspect selections for the English fluency suite.

The metadata CSV is the catalogue of available benchmark entries.  This module
contains only the research decision about which entries are in scope and how
they map to the fluency categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark/aspect selection from NLG-eval."""

    name: str
    benchmark: str
    aspect: str
    task_family: str
    categories: Tuple[str, ...]
    task: Optional[str] = None
    original_data: Optional[str] = None
    include: bool = True
    notes: str = ""


def _spec(
    name: str,
    benchmark: str,
    aspect: str,
    task_family: str,
    *categories: str,
    task: Optional[str] = None,
    original_data: Optional[str] = None,
    notes: str = "",
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        benchmark=benchmark,
        aspect=aspect,
        task_family=task_family,
        categories=tuple(categories),
        task=task,
        original_data=original_data,
        notes=notes,
    )


# Aspect names below are the short canonical names in NLG-Eval_meta_info.csv.
# The JSONL contains the same names followed by the full rubric description. The loader resolves that representation difference.
NLG_EVAL_FLUENCY_SPECS: Tuple[BenchmarkSpec, ...] = (
    _spec("chiang_cohesiveness", "chiang LLM Evaluation", "Cohesiveness", "story_generation", "coherence"),
    _spec("chiang_grammaticality", "chiang LLM Evaluation", "Grammaticality", "story_generation", "grammaticality"),
    _spec("coeval_grammaticality", "CoEval", "Grammaticality", "story_generation", "grammaticality"),
    _spec("coeval_coherence", "CoEval", "Coherence", "story_generation", "coherence"),
    _spec("coeval_clarity", "CoEval", "Clarity", "story_generation", "clarity"),
    _spec("hanna_coherence", "Hanna", "Coherence", "story_generation", "coherence"),
    _spec("nextchapter_fluency", "nextchapter", "Fluency", "story_generation", "grammaticality"),
    _spec("nextchapter_coherence", "nextchapter", "Coherence", "story_generation", "coherence"),
    _spec("pplm_fluency", "PPLM", "Fluency", "controlled_prose", "grammaticality", "clarity"),
    _spec("e2e_naturalness", "E2E NLG", "Naturalness", "data_to_text", "naturalness"),
    _spec("inlg16_naturalness", "INLG16", "Naturalness", "data_to_text", "naturalness"),
    _spec("rankme_naturalness", "RankMe", "Naturalness", "data_to_text", "naturalness"),
    _spec("webnlg2017_fluency", "webnlg_2017", "Fluency", "data_to_text", "naturalness"),
    _spec("webnlg2017_grammaticality", "webnlg_2017", "Grammaticality", "data_to_text", "grammaticality"),
    _spec("webnlg2020_fluency", "webnlg_2020", "Fluency", "data_to_text", "coherence", "clarity", "naturalness"),
    _spec("webnlg2020_text_structure", "webnlg_2020", "Text Structure", "data_to_text", "grammaticality", "coherence"),
    _spec("protagolabs_gec_grammaticality", "protagolabs", "Grammaticality", "gec_paraphrasing", "grammaticality", task="Grammatical Error Correction", original_data="BEA19"),
    _spec("tmu_gfm_grammaticality", "TMU-GFM", "Grammaticality", "gec_paraphrasing", "grammaticality", "clarity", task="Grammatical Error Correction"),
    _spec("tmu_gfm_fluency", "TMU-GFM", "Fluency", "gec_paraphrasing", "naturalness", task="Grammatical Error Correction"),
    _spec("parabank_fluency", "parabank", "Fluency", "gec_paraphrasing", "grammaticality"),
    _spec("protagolabs_summary_fluency", "protagolabs", "Fluency", "summarization", "grammaticality", task="Summarization", original_data="cnn/dm"),
    _spec("protagolabs_summary_coherence", "protagolabs", "Coherence", "summarization", "coherence", task="Summarization", original_data="cnn/dm"),
    _spec("dialsumm_eval_fluency", "DialSummEval", "Fluency", "summarization", "grammaticality"),
    _spec("dialsumm_eval_coherence", "DialSummEval", "Coherence", "summarization", "coherence"),
    _spec("summeval_op_fluency", "SummEval-OP", "Fluency", "summarization", "grammaticality", "clarity"),
    _spec("summeval_op_coherence", "SummEval-OP", "Coherence", "summarization", "coherence"),
    _spec("asset_fluency", "ASSET", "Fluency", "simplification", "grammaticality"),
    _spec("human_likert_fluency", "Human-likert", "Fluency", "simplification", "grammaticality"),
    _spec("lens_overall_quality", "LENS", "Overall Quality", "simplification", "grammaticality", notes="Contains fluency plus meaning preservation and simplification; review before primary inclusion."),
    _spec("metaeval_fluency", "metaeval", "Fluency", "simplification", "grammaticality", "naturalness"),
    _spec("protagolabs_simplification_fluency", "protagolabs", "Fluency", "simplification", "grammaticality", "clarity", task="Text Simplification", original_data="Newsela"),
    _spec("samsa_grammaticality", "SAMSA", "Grammaticality", "simplification", "grammaticality"),
)


def get_nlg_eval_specs(include_review_required: bool = False) -> Tuple[BenchmarkSpec, ...]:
    """Return the accepted NLG-eval selections."""
    if include_review_required:
        return NLG_EVAL_FLUENCY_SPECS
    return tuple(spec for spec in NLG_EVAL_FLUENCY_SPECS if not spec.notes)
