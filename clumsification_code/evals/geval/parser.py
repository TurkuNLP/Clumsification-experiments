from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, Optional


class RetryableParseError(ValueError):
    """
    Raised when the model response could not be parsed into a valid score.

    The scorer treats this as retryable because malformed JSON or missing fields
    are usually transient model/provider issues.
    """


@dataclass(frozen=True)
class ParsedScore:
    score: float
    payload: Dict[str, Any]
    raw_content: str


_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    match = _CODE_FENCE_RE.fullmatch(raw)
    if match:
        return match.group(1).strip()
    return raw


def extract_json_object(raw: str) -> Dict[str, Any]:
    """
    Extract a JSON object from raw model output.

    Expected path: raw is already valid JSON due to response_format.
    Fallback path: extract the first {...} block to survive occasional provider
    formatting deviations.
    """
    if raw is None:
        raise RetryableParseError("Empty response content")

    raw = _strip_code_fence(str(raw).strip())

    if not raw:
        raise RetryableParseError("Empty response content")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict):
        return obj

    first = raw.find("{")
    last = raw.rfind("}")

    if first >= 0 and last > first:
        candidate = raw[first : last + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RetryableParseError(
                f"Could not parse JSON object from response: {raw[:500]!r}"
            ) from exc

        if isinstance(obj, dict):
            return obj

    raise RetryableParseError(
        f"Could not parse JSON object from response: {raw[:500]!r}"
    )


def parse_score_response(
    raw_content: str,
    *,
    score_min: float = 1.0,
    score_max: float = 5.0,
    clamp: bool = True,
) -> ParsedScore:
    if score_min >= score_max:
        raise ValueError(f"score_min must be < score_max, got {score_min} >= {score_max}")

    obj = extract_json_object(raw_content)

    if "score" not in obj:
        raise RetryableParseError(f"G-Eval response missing 'score': {obj}")

    try:
        score = float(obj["score"])
    except Exception as exc:
        raise RetryableParseError(f"G-Eval score is not numeric: {obj.get('score')!r}") from exc

    if not math.isfinite(score):
        raise RetryableParseError(f"G-Eval score is non-finite: {score}")

    if clamp:
        score = min(max(score, float(score_min)), float(score_max))
    elif not (score_min <= score <= score_max):
        raise RetryableParseError(
            f"G-Eval score outside range [{score_min}, {score_max}]: {score}"
        )

    return ParsedScore(score=score, payload=obj, raw_content=raw_content)


def is_retryable_parse_error(exc: BaseException) -> bool:
    return isinstance(exc, RetryableParseError)