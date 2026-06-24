from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence


GEVAL_QE_PROMPT_VERSION = "geval_qe_no_reference_v1"

Message = Dict[str, str]


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str


DEFAULT_QE_RUBRIC: Sequence[RubricCriterion] = (
    RubricCriterion(
        "Fluency",
        "Grammar, spelling, punctuation, sentence structure, and readability.",
    ),
    RubricCriterion(
        "Naturalness",
        "Whether the text sounds like coherent human-written English.",
    ),
    RubricCriterion(
        "Coherence",
        "Logical flow, consistency, and absence of contradictions within the text.",
    ),
    RubricCriterion(
        "Clarity",
        "Whether the meaning is understandable and not unnecessarily confusing.",
    ),
    RubricCriterion(
        "Lexical and syntactic quality",
        "Appropriate word choice and sentence variety.",
    ),
    RubricCriterion(
        "Overall standalone quality",
        "Holistic quality given no source, reference, or user prompt.",
    ),
)


ASPECT_RUBRICS: Mapping[str, Sequence[RubricCriterion]] = {
    "fluency": (
        RubricCriterion(
            "Fluency",
            "Grammar, spelling, punctuation, sentence structure, and readability.",
        ),
        RubricCriterion(
            "Naturalness",
            "Whether the text sounds like coherent human-written English.",
        ),
        RubricCriterion(
            "Surface correctness",
            "Absence of awkward, ungrammatical, malformed, or hard-to-read language.",
        ),
    ),
    "coherence": (
        RubricCriterion(
            "Coherence",
            "Logical flow, consistency, topic continuity, and absence of self-contradictions.",
        ),
        RubricCriterion(
            "Clarity",
            "Whether the text is understandable as standalone writing.",
        ),
    ),
    "consistency": (
        RubricCriterion(
            "Internal consistency",
            "Whether the text contradicts itself internally.",
        ),
        RubricCriterion(
            "Caution about missing context",
            "Do not judge factual consistency against a source, reference, or prompt that is not provided.",
        ),
    ),
    "naturalness": (
        RubricCriterion(
            "Naturalness",
            "Whether the text sounds like plausible human-written English.",
        ),
        RubricCriterion(
            "Idiomaticity",
            "Appropriate word choice, phrasing, and sentence rhythm.",
        ),
    ),
    "overall": DEFAULT_QE_RUBRIC,
    "quality": DEFAULT_QE_RUBRIC,
}


# Optional task/aspect mapping. The current benchmark runner only passes raw text
# into score_texts(...), so GEvalScorer normally uses the default no-reference
# rubric. These mappings are here for future runner variants that may pass task
# or aspect metadata.
TASK_ASPECT_MAPPING: Mapping[str, Mapping[str, str]] = {
    "summeval": {
        "fluency": "fluency",
        "coherence": "coherence",
        "consistency": "consistency",
    },
    "ellipse": {
        "overall": "overall",
        "cohesion": "coherence",
    },
    "usr": {
        "overall": "overall",
        "natural": "naturalness",
    },
    "openmeva": {
        "overall": "overall",
    },
    "webnlg": {
        "fluency": "fluency",
    },
    "hanna": {
        "coherence": "coherence",
        "complexity": "quality",
    },
    "argessay": {
        "language_mastery": "fluency",
        "complexity": "quality",
        "vocabulary": "quality",
        "language_constructs": "fluency",
    },
    "humanratings": {
        "quality": "quality",
        "naturalness": "naturalness",
    },
    "fed": {
        "fluent": "fluency",
        "overall": "overall",
    },
    "e2e": {
        "naturalness": "naturalness",
        "quality": "quality",
    },
}


def truncate_text(text: object, max_input_chars: int) -> str:
    text = "" if text is None else str(text)

    if max_input_chars is None or max_input_chars <= 0:
        return text

    if len(text) <= max_input_chars:
        return text

    return text[:max_input_chars] + "\n\n[TRUNCATED]"


def normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def rubric_for(
    *,
    task: Optional[str] = None,
    aspect: Optional[str] = None,
) -> Sequence[RubricCriterion]:
    task_key = normalize_key(task)
    aspect_key = normalize_key(aspect)

    if task_key and aspect_key:
        mapped_aspect = TASK_ASPECT_MAPPING.get(task_key, {}).get(aspect_key)
        if mapped_aspect and mapped_aspect in ASPECT_RUBRICS:
            return ASPECT_RUBRICS[mapped_aspect]

    if aspect_key and aspect_key in ASPECT_RUBRICS:
        return ASPECT_RUBRICS[aspect_key]

    return DEFAULT_QE_RUBRIC


def render_rubric(criteria: Sequence[RubricCriterion]) -> str:
    return "\n".join(
        f"{idx}. {criterion.name}: {criterion.description}"
        for idx, criterion in enumerate(criteria, start=1)
    )


def build_system_prompt() -> str:
    return (
        "You are an expert evaluator of natural language generation quality. "
        "You must provide calibrated, reproducible, no-reference quality judgments "
        "for standalone text. You do not know the original task, source document, "
        "reference answer, or user prompt. Therefore, do not judge factual accuracy "
        "against missing context. Judge only what can be assessed from the candidate "
        "text itself."
    )


def build_user_prompt(
    text: object,
    *,
    max_input_chars: int = 12000,
    task: Optional[str] = None,
    aspect: Optional[str] = None,
) -> str:
    text = truncate_text(text, max_input_chars=max_input_chars)
    rubric = render_rubric(rubric_for(task=task, aspect=aspect))

    return f"""
Evaluate the following candidate text using a G-Eval-style rubric.

Evaluation criteria:
{rubric}

Important constraints:
- This is a no-reference quality-estimation setting.
- Do not penalize the text for missing facts that require external context.
- Do not reward unsupported factual claims beyond their writing quality.
- Prefer concise but well-formed text over verbose incoherent text.
- Use the full score range when appropriate.
- Return JSON only.

Score scale:
1 = very poor quality; many severe errors; hard to understand.
2 = poor quality; noticeable errors or awkwardness; weak coherence.
3 = acceptable quality; understandable but with flaws.
4 = good quality; mostly fluent, clear, and coherent.
5 = excellent quality; highly fluent, natural, clear, and coherent.

Return exactly this JSON shape:
{{
  "score": number,
  "reasoning": string
}}

Candidate text:
<candidate>
{text}
</candidate>
""".strip()


def build_messages(
    text: object,
    *,
    max_input_chars: int = 12000,
    task: Optional[str] = None,
    aspect: Optional[str] = None,
) -> List[Message]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(
                text,
                max_input_chars=max_input_chars,
                task=task,
                aspect=aspect,
            ),
        },
    ]


def build_response_format_json_schema() -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "geval_quality_score",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "A no-reference quality score from 1 to 5.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief rationale for the score. Keep concise.",
                    },
                },
                "required": ["score", "reasoning"],
            },
        },
    }