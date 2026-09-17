"""Decode model text and validate results before canonical persistence."""
from __future__ import annotations

from collections import Counter
import re
import json
from typing import Any

from clumsification_code.data.candidate_identity import make_candidate_id
from clumsification_code.data.schemas import CandidateRecord

from .generation_config import GenerationRequest
from .schemas import ChatCompletion, PerturbationInput, PerturbationResult, SkippedGeneration, SkippedPerturbation


class GenerationValidationError(ValueError):
    """A method returned output that cannot become a canonical candidate."""


def parse_completion(output: Any, *, thinking: bool = False) -> ChatCompletion:
    """Recover only identifiable final text; truncated metadata need not lose text."""
    if not getattr(output, "outputs", None):
        return ChatCompletion("", failure_reason="empty_output")
    completion = output.outputs[0]
    raw = str(getattr(completion, "text", ""))
    finish = getattr(completion, "finish_reason", None)
    metadata = {"finish_reason": finish,
                "generated_tokens": len(getattr(completion, "token_ids", []) or []),
                "actual_prompt_tokens": len(getattr(output, "prompt_token_ids", []) or [])}
    final = getattr(completion, "content", None)
    if isinstance(final, str):
        raw = final
    elif "</think>" in raw and (thinking or not raw.lstrip().startswith(("{", "```"))):
        raw = raw.split("</think>", 1)[1]
    elif thinking or ("<think>" in raw and not raw.lstrip().startswith(("{", "```"))):
        return ChatCompletion("", metadata, "reasoning_only")
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json|text)?\s*\n?", "", raw)[:-3].strip()
    recovered = False
    if raw.startswith("{"):
        try:
            value, _ = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            raw = value["text"]
            counts = value.get("applied_edits")
            if isinstance(counts, dict) and all(isinstance(k,str) and type(v) is int and v >= 0 for k,v in counts.items()):
                metadata["reported_applied_edits"] = counts
            else:
                metadata["statistics_unavailable"] = True
            recovered = True
        elif value is None:
            # Only recover a fully closed first text string; never repair cut text.
            match = re.match(r'^\{\s*"text"\s*:\s*', raw)
            if match:
                try:
                    text, _ = json.JSONDecoder().raw_decode(raw[match.end():])
                    if isinstance(text, str):
                        raw, recovered = text, True
                        metadata["statistics_unavailable"] = True
                except json.JSONDecodeError:
                    pass
            if not recovered:
                return ChatCompletion("", metadata, "unparseable_output")
        else:
            return ChatCompletion("", metadata, "unparseable_output")
    if finish == "length" and not recovered:
        return ChatCompletion("", metadata, "truncated_output")
    if finish in ("abort", "error"):
        return ChatCompletion("", metadata, "engine_abort")
    if recovered and finish == "length":
        metadata["metadata_truncated"] = True
    return ChatCompletion(raw, metadata, None if raw.strip() else "empty_output")


def parse_vllm_text(output: Any) -> str:
    return parse_completion(output).text


def collect_batch_results(
    request: GenerationRequest,
    batch: list[PerturbationInput],
    results: list[PerturbationResult],
    candidate_counts: dict[str, int],
) -> tuple[list[CandidateRecord], dict[str, dict[str, Any]]]:
    """Check identity/acceptance and separate successes from retryable failures."""
    batch_input_by_parent = {str(item.candidate_id): item for item in batch}
    expected_ids = Counter(batch_input_by_parent.keys())
    actual_ids = Counter(str(result.parent_candidate_id) for result in results)
    if actual_ids != expected_ids:
        raise ValueError("Generation batch did not return exactly one result per input")
    batch_candidates: list[CandidateRecord] = []
    batch_failures: dict[str, dict[str, Any]] = {}
    for result in results:
        parent_id = str(result.parent_candidate_id)
        item = batch_input_by_parent[parent_id]
        failure = _output_failure(result, item, allow_unchanged=request.allow_unchanged)
        expected = (
            request.dataset_name, request.method, request.perturbation_source,
            request.run_id, request.source_layer, request.source_method, request.source_run_id,
            request.target_layer,
        )
        actual = (
            result.dataset_name, result.perturbation_method,
            result.perturbation_source, result.run_id, result.source_layer,
            result.source_method, result.source_run_id, result.target_layer,
        )
        if actual != expected or result.base_text_id != item.base_text_id:
            raise GenerationValidationError("Generated result provenance does not match the request")
        if result.perturbation_source == "LLM":
            counts = result.metadata.get("reported_applied_edits")
            result.metadata["statistics_source"] = "model_self_report" if isinstance(counts, dict) else "unavailable"
            result.metadata["missing_reported_edits"] = [edit for edit in result.perturbation_edits if not isinstance(counts, dict) or edit not in counts]
        if failure is not None:
            batch_failures[parent_id] = failure
            continue
        candidate_index = candidate_counts[parent_id]
        candidate_counts[parent_id] += 1
        batch_candidates.append(
            CandidateRecord(
                dataset_name=result.dataset_name,
                base_text_id=result.base_text_id,
                candidate_id=make_candidate_id(
                    dataset_name=result.dataset_name,
                    perturbation_method=request.method,
                    run_id=request.run_id,
                    base_text_id=result.base_text_id,
                    target_layer=request.target_layer,
                    parent_candidate_id=parent_id,
                    candidate_index=candidate_index,
                ),
                candidate_index=candidate_index,
                text=result.text,
                perturbation_method=request.method,
                perturbation_source=request.perturbation_source,
                run_id=request.run_id,
                source_layer=request.source_layer,
                source_method=request.source_method,
                source_run_id=request.source_run_id,
                target_layer=request.target_layer,
                parent_candidate_id=parent_id,
                perturbation_edits=tuple(result.perturbation_edits),
                target_dimensions=tuple(result.target_dimensions),
                severity=result.severity,
                edit_count=result.edit_count,
                generator=result.generator,
                seed=result.seed,
                prompt_version=result.prompt_version,
                metadata=dict(result.metadata),
            )
        )
    return batch_candidates, batch_failures


def _output_failure(
    result: PerturbationResult, item: PerturbationInput, *, allow_unchanged: bool
) -> dict[str, Any] | None:
    """Apply the existing text-acceptance rules without changing the result."""
    parent_id = str(result.parent_candidate_id)
    failure: dict[str, Any] | None = None
    if result.metadata.get("generation_failure"):
        return {"parent_candidate_id": parent_id, "reason": result.metadata["generation_failure"]}
    if isinstance(result.text, SkippedGeneration):
        failure = {
            "parent_candidate_id": parent_id,
            "reason": "over_length",
            "prompt_tokens": result.text.prompt_tokens,
            "required_tokens": result.text.required_tokens,
        }
    elif isinstance(result.text, SkippedPerturbation):
        failure = {
            "parent_candidate_id": parent_id,
            "reason": result.text.reason,
            "attempts": result.text.attempts,
        }
    elif not isinstance(result.text, str) or not result.text.strip():
        failure = {"parent_candidate_id": parent_id, "reason": "empty_output"}
    elif not allow_unchanged and " ".join(result.text.split()) == " ".join(str(item.text).split()):
        failure = {"parent_candidate_id": parent_id, "reason": "unchanged_output"}
    max_output_chars = result.metadata.get("max_output_chars")
    if max_output_chars is not None and result.perturbation_source != "LLM":
        if (
            isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or max_output_chars < 1
        ):
            raise GenerationValidationError("max_output_chars must be a positive integer")
        if failure is None and len(result.text) > max_output_chars:
            failure = {
                "parent_candidate_id": parent_id,
                "reason": "max_output_chars_exceeded",
                "output_chars": len(result.text),
                "max_output_chars": max_output_chars,
            }
    return failure
