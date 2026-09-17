# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Sampled-operation LLM perturbation method and prompt renderer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence
import math

from .assignment_plan import LLMAssignment, load_llm_assignments
from .sampling import (
    EditCatalogEntry,
    SampledEditAssignment,
    load_edit_catalog,
)
from .schemas import ChatCompletion, GenerationRuntime, PerturbationInput, PerturbationResult


SAMPLED_METHOD = "llm_sampled"
SINGLE_METHOD = "llm_single"
PROMPT_VERSION = "llm-sampled-v3-json1"
SINGLE_PROMPT_VERSION = "llm-single-v3-json1"

_SYSTEM_PROMPT = """You are a controlled fluency-degrading editor. Rewrite the source so it is substantially less fluent while preserving its propositional content. Fluency concerns grammaticality, coherence, clarity, and naturalness. Perturbation edits come in three different severities. Weak means that the awkwardness is noticeable to a proficient reader but remains easy to follow. Medium means that the awkwardness is noticeable to an ordinary reader and at least some passages require rereading. Strong means that the he awkwardness is noticeable even to a beginner and multiple passages require rereading. The edited text's meaning and facts must be recoverable from the edited text, but doing so requires extra effort. Preserve existing source errors unless a requested operation directly targets that span. Return only the edited text followed by the necessary statistics."""

_SEVERITY_GUIDANCE = {
    "weak": "The awkwardness is noticeable to a proficient reader but remains easy to follow.",
    "medium": "The awkwardness is noticeable to an ordinary reader and at least some passages require rereading.",
    "strong": "The awkwardness is noticeable even to a beginner and multiple passages require rereading",
}


@dataclass(frozen=True)
class SampledPromptRequest:
    messages: list[dict[str, str]]
    assignment: SampledEditAssignment
    prompt_version: str = PROMPT_VERSION


def _stable_item_seed(base_seed: int, item: Mapping[str, Any], index: int) -> int:
    identity = item.get("candidate_id") or item.get("custom_id") or item.get("_source_index") or index
    raw = f"{base_seed}:{identity}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _render_operation(entry: EditCatalogEntry, index: int) -> str:
    instruction = entry.instruction or (
        f"Apply the edit type '{entry.edit_type}' while preserving the source meaning."
    )
    extra = ""
    if entry.applicability:
        extra += " Suitable context: " + "; ".join(entry.applicability) + "."
    if entry.minimum_realization:
        extra += f" Minimum realization: {entry.minimum_realization}"
    if entry.non_examples:
        extra += " Do not count these near-misses: " + "; ".join(entry.non_examples) + "."
    return (
        f"{index}. {entry.edit_type} ({entry.edit_id})\n"
        f"Instruction: {instruction}{extra}\n"
        f"Illustration — source: {entry.example_clean}\n"
        f"Illustration — edited: {entry.example_edited}"
    )


def render_sampled_messages(
    item: Mapping[str, Any],
    assignment: SampledEditAssignment,
    *,
    max_length: int | None = None,
) -> list[dict[str, str]]:
    """Render one operation-conditioned chat request."""
    text = str(item.get("text", "")).replace("\n", " ")
    if max_length is None:
        max_length = int(item.get("max_length") or min(int(len(text) * 1.1), len(text) + 500))
    operations = "\n\n".join(
        _render_operation(entry, index)
        for index, entry in enumerate(assignment.edits, start=1)
    )
    dimensions = ", ".join(assignment.target_dimensions)
    task = f"""Target dimensions: {dimensions}
Target severity: {assignment.severity}.

Required operations:
{operations}

Requirements:
- Apply exactly {len(assignment.edits)} edits.
- Apply every required operation at least once.
- Use at least one qualifying change in {max(1, math.floor(len(assignment.edits)/2))} distinct sentences.
- Edits must correspond to severity and also be genuine changes; isolated neutral synonym substitutions do not count.
- Do not add facts, omit propositions, change timeline or polarity, translate, or repair unrelated source errors.
- Do not copy the illustration text or its entities into the source.
- The edited text must not be more than 100 characters longer.
- Return JSON with "text" first and "applied_edits" second. The character preference applies only to "text".
- In "applied_edits", include every requested edit_id with an integer count of qualifying changes actually made. Use 0 when not realized; do not claim an edit just because it was requested.
- Count realized instances, not the number of requested operation types. Escape quotes and newlines in the JSON text string. Do not include explanations.

Source text:
{text}"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


class SampledLLMMethod:
    """Deterministic edit sampler and prompt renderer for ``llm_sampled``."""

    name = SAMPLED_METHOD
    perturbation_source = "LLM"
    prompt_version = PROMPT_VERSION

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        catalog_path = self.config.get(
            "edit_catalog", "data/perturbation_prompts/english/edit_types.jsonl"
        )
        self.catalog = load_edit_catalog(catalog_path)
        self.seed = int(self.config.get("seed", 42))
        self._request_cache = {}
        self._planned_assignments = self._load_planned_assignments()

    def _load_planned_assignments(self) -> dict[str, LLMAssignment]:
        path = self.config.get("assignment_file")
        if path is None:
            raise ValueError(
                "LLM generation requires assignment_file created by "
                "plan_llm_assignments.py"
            )
        if not isinstance(path, (str, bytes)):
            raise ValueError("assignment_file must be a path string")
        assignments = [
            assignment
            for assignment in load_llm_assignments(path)
            if assignment.method == self.name
        ]
        by_base_text_id = {
            assignment.base_text_id: assignment for assignment in assignments
        }
        if len(by_base_text_id) != len(assignments):
            raise ValueError(
                f"Assignment file has duplicate {self.name!r} rows for one source"
            )
        catalog_by_id = {entry.edit_id: entry for entry in self.catalog}
        for assignment in assignments:
            try:
                entries = tuple(catalog_by_id[edit_id] for edit_id in assignment.edits)
            except KeyError as exc:
                raise ValueError(
                    f"Assignment for {assignment.base_text_id!r} references an "
                    f"unknown edit type {exc.args[0]!r}"
                ) from exc
            dimensions = tuple(dict.fromkeys(
                dimension for entry in entries for dimension in entry.target_dimensions
            ))
            if assignment.target_dimensions != dimensions:
                raise ValueError(
                    f"Assignment dimensions do not match edits for "
                    f"{assignment.base_text_id!r}"
                )
        return by_base_text_id

    def _planned_assignment_for_item(
        self, item: Mapping[str, Any], *, seed: int
    ) -> SampledEditAssignment:
        base_text_id = item.get("base_text_id")
        if not isinstance(base_text_id, str) or not base_text_id:
            raise ValueError("Planned LLM generation requires base_text_id")
        assignment = self._planned_assignments.get(base_text_id)
        if assignment is None:
            raise ValueError(
                f"No {self.name!r} assignment exists for source {base_text_id!r}"
            )
        catalog_by_id = {entry.edit_id: entry for entry in self.catalog}
        return SampledEditAssignment(
            target_dimensions=assignment.target_dimensions,
            edits=tuple(catalog_by_id[edit_id] for edit_id in assignment.edits),
            severity=assignment.severity,
            seed=seed,
        )

    def assignment_for_item(self, item: Mapping[str, Any], *, index: int = 0) -> SampledEditAssignment:
        seed = _stable_item_seed(self.seed, item, index)
        return self._planned_assignment_for_item(item, seed=seed)

    def build_requests(self, items: Sequence[Mapping[str, Any]]) -> list[SampledPromptRequest]:
        requests = []
        tolerance = int(self.config.get("max_output_char_tolerance", 256))
        for index, item in enumerate(items):
            key = (item.get("candidate_id"), item.get("base_text_id"), str(item.get("text", "")), item.get("max_length"))
            if key in self._request_cache:
                requests.append(self._request_cache[key])
                continue
            assignment = self.assignment_for_item(item, index=index)
            text = str(item.get("text", "")).replace("\n", " ")
            base_limit = int(
                item.get("max_length")
                or min(int(len(text) * 1.1), len(text) + 500)
            )
            requests.append(
                SampledPromptRequest(
                    messages=render_sampled_messages(
                        item, assignment, max_length=base_limit + tolerance
                    ),
                    assignment=assignment,
                    prompt_version=self.prompt_version,
                )
            )
            self._request_cache[key] = requests[-1]
        return requests

    def build_prompts(self, items: Sequence[Mapping[str, Any]]) -> list[list[dict[str, str]]]:
        return [request.messages for request in self.build_requests(items)]

    def generate(
        self,
        items: list[PerturbationInput],
        runtime: GenerationRuntime,
    ) -> list[PerturbationResult]:
        requests = self.build_requests(
            [
                item.metadata
                | {
                    "base_text_id": item.base_text_id,
                    "candidate_id": item.candidate_id,
                    "text": item.text,
                }
                for item in items
            ]
        )
        model, outputs = runtime.run_chat(
            self.config, [request.messages for request in requests],
            source_texts=[str(item.text).replace("\n", " ") for item in items],
            request_ids=[str(item.candidate_id) for item in items],
            requested_edits=[[edit.edit_id for edit in request.assignment.edits] for request in requests],
        )
        return [
            PerturbationResult(
                dataset_name=item.dataset_name,
                base_text_id=item.base_text_id,
                text=output.text if isinstance(output, ChatCompletion) else output,
                source_layer=item.source_layer,
                source_method=item.source_method,
                source_run_id=item.source_run_id,
                parent_candidate_id=item.candidate_id,
                target_layer=int(self.config["target_layer"]),
                perturbation_method=self.name,
                perturbation_source=self.perturbation_source,
                run_id=str(self.config["run_id"]),
                perturbation_edits=[entry.edit_id for entry in request.assignment.edits],
                target_dimensions=list(request.assignment.target_dimensions),
                severity=request.assignment.severity,
                edit_count=len(request.assignment.edits),
                generator=model,
                seed=request.assignment.seed,
                prompt_version=request.prompt_version,
                method_config=dict(self.config),
                metadata={
                    **{k: v for k, v in item.metadata.items() if k.startswith("pilot_")},
                    "length_delta_chars": len(output.text if isinstance(output, ChatCompletion) else output) - len(item.text)
                        if isinstance(output, (str, ChatCompletion)) else None,
                    **(output.metadata if isinstance(output, ChatCompletion) else {}),
                    **({"generation_failure": output.failure_reason}
                       if isinstance(output, ChatCompletion) and output.failure_reason else {}),
                },
            )
            for item, output, request in zip(items, outputs, requests)
        ]


class SingleLLMMethod(SampledLLMMethod):
    """One-operation counterpart to the length-scaled ``llm_sampled`` method."""

    name = SINGLE_METHOD
    prompt_version = SINGLE_PROMPT_VERSION

    def assignment_for_item(
        self, item: Mapping[str, Any], *, index: int = 0
    ) -> SampledEditAssignment:
        seed = _stable_item_seed(self.seed, item, index)
        planned = self._planned_assignment_for_item(item, seed=seed)
        if len(planned.edits) != 1:
            raise ValueError(
                f"llm_single assignment for {item.get('base_text_id')!r} "
                "must contain exactly one edit"
            )
        return planned


__all__ = [
    "PROMPT_VERSION",
    "SINGLE_METHOD",
    "SINGLE_PROMPT_VERSION",
    "SAMPLED_METHOD",
    "SingleLLMMethod",
    "SampledLLMMethod",
    "SampledPromptRequest",
    "render_sampled_messages",
]
