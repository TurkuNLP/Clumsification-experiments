# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from clumsification_code.prompts import load_prompt_data, load_prompt_spec


Message = Dict[str, str]

_RUBRIC_DATA = load_prompt_data("evaluation/rubrics/geval_no_reference.json")
_PROMPT_SPEC = load_prompt_spec("evaluation/protocols/geval_json.json")

GEVAL_QE_PROMPT_VERSION = _PROMPT_SPEC.version


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str


def _criteria(rows: Sequence[Sequence[str]]) -> tuple[RubricCriterion, ...]:
    return tuple(RubricCriterion(name=row[0], description=row[1]) for row in rows)


DEFAULT_QE_RUBRIC: Sequence[RubricCriterion] = _criteria(_RUBRIC_DATA["default"])

ASPECT_RUBRICS: Mapping[str, Sequence[RubricCriterion]] = {
    name: _criteria(rows) for name, rows in _RUBRIC_DATA["aspects"].items()
}
for _alias, _target in _RUBRIC_DATA["aspect_aliases"].items():
    ASPECT_RUBRICS[_alias] = (
        DEFAULT_QE_RUBRIC if _target == "default" else ASPECT_RUBRICS[_target]
    )

TASK_ASPECT_MAPPING: Mapping[str, Mapping[str, str]] = _RUBRIC_DATA[
    "task_aspect_mapping"
]


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
    return _PROMPT_SPEC.messages[0].content


def build_user_prompt(
    text: object,
    *,
    max_input_chars: int = 12000,
    task: Optional[str] = None,
    aspect: Optional[str] = None,
) -> str:
    return _render_messages(
        text,
        max_input_chars=max_input_chars,
        task=task,
        aspect=aspect,
    )[1]["content"]


def build_messages(
    text: object,
    *,
    max_input_chars: int = 12000,
    task: Optional[str] = None,
    aspect: Optional[str] = None,
) -> List[Message]:
    return _render_messages(
        text,
        max_input_chars=max_input_chars,
        task=task,
        aspect=aspect,
    )


def _render_messages(
    text: object,
    *,
    max_input_chars: int,
    task: Optional[str],
    aspect: Optional[str],
) -> List[Message]:
    candidate_text = truncate_text(text, max_input_chars=max_input_chars)
    rubric = render_rubric(rubric_for(task=task, aspect=aspect))
    return _PROMPT_SPEC.render_messages(
        {"candidate_text": candidate_text, "rubric": rubric}
    )


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
