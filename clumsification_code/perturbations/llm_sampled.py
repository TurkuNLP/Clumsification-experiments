# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Sampled-operation LLM perturbation method and prompt renderer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from clumsification_code.data.io import canonical_json_hash, sha256_file

from .sampling import (
    EditCatalogEntry,
    SampledEditAssignment,
    load_edit_catalog,
    sample_edit_assignment,
)
from .schemas import GenerationRuntime, PerturbationInput, PerturbationResult


SAMPLED_METHOD = "llm_sampled"
PROMPT_VERSION = "llm-sampled-v1"

_SYSTEM_PROMPT = """You are a controlled fluency-perturbation editor. Rewrite the source so it is substantially less fluent while preserving its propositional content. Fluency concerns grammaticality, coherence, clarity, and naturalness. Preserve every claim, entity, number, polarity, temporal relation, causal relation, degree of certainty, and speaker attitude. Preserve existing source errors unless a requested operation directly targets that span. Return only the edited text."""

_SEVERITY_GUIDANCE = {
    "weak": "The awkwardness is noticeable to a proficient reader but remains easy to follow.",
    "medium": "The awkwardness is conspicuous to an ordinary reader and at least some passages require extra processing, while the original propositions remain recoverable.",
    "strong": "The awkwardness is unmistakable even to a beginner and multiple passages require rereading, while the original propositions remain recoverable.",
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
    if entry.minimum_realization:
        extra += f" Minimum realization: {entry.minimum_realization}"
    if entry.non_examples:
        extra += " Do not count these near-misses: " + "; ".join(entry.non_examples) + "."
    return (
        f"{index}. {entry.edit_type} ({entry.edit_id})\n"
        f"Instruction: {instruction}{extra}\n"
        f"Illustration — edited: {entry.example_edited}\n"
        f"Illustration — clean: {entry.example_clean}"
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
Target severity: {assignment.severity}. {_SEVERITY_GUIDANCE[assignment.severity]}

Required operations:
{operations}

Requirements:
- Apply every required operation at least once.
- Use at least one qualifying change in each of 3 distinct sentences when the source contains that many sentences.
- Clause-level or discourse-level changes are required; isolated neutral synonym substitutions do not count.
- Do not add facts, omit propositions, change timeline or polarity, translate, or repair unrelated source errors.
- Do not copy the illustration text or its entities into the source.
- The maximum output length is {max_length} characters.

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

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        catalog_path = self.config.get(
            "edit_catalog", "data/perturbation_prompts/english/edit_types.jsonl"
        )
        self.catalog = load_edit_catalog(catalog_path)
        self.catalog_hash = sha256_file(catalog_path)
        self.target_dimensions = tuple(
            self.config.get("target_dimensions", ("Clarity", "Naturalness"))
        )
        self.n_edits = int(self.config.get("n_edits", self.config.get("edits_per_candidate", 3)))
        self.severity = self.config.get("severity", "medium")
        self.seed = int(self.config.get("seed", 42))
        self.weights = self.config.get("weights")
        self.require_dimension_coverage = bool(self.config.get("require_dimension_coverage", True))

    def assignment_for_item(self, item: Mapping[str, Any], *, index: int = 0) -> SampledEditAssignment:
        seed = _stable_item_seed(self.seed, item, index)
        return sample_edit_assignment(
            self.catalog,
            target_dimensions=self.target_dimensions,
            n_edits=self.n_edits,
            severity=self.severity,
            seed=seed,
            weights=self.weights,
            require_dimension_coverage=self.require_dimension_coverage,
        )

    def build_requests(self, items: Sequence[Mapping[str, Any]]) -> list[SampledPromptRequest]:
        return [
            SampledPromptRequest(
                messages=render_sampled_messages(item, assignment_for_item := self.assignment_for_item(item, index=index)),
                assignment=assignment_for_item,
            )
            for index, item in enumerate(items)
        ]

    def build_prompts(self, items: Sequence[Mapping[str, Any]]) -> list[list[dict[str, str]]]:
        return [request.messages for request in self.build_requests(items)]

    def generate(
        self,
        items: list[PerturbationInput],
        runtime: GenerationRuntime,
    ) -> list[PerturbationResult]:
        requests = self.build_requests(
            [item.metadata | {"text": item.text} for item in items]
        )
        model, outputs = runtime.run_chat(
            self.config, [request.messages for request in requests]
        )
        return [
            PerturbationResult(
                dataset_name=item.dataset_name,
                base_text_id=item.base_text_id,
                text=output,
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
                prompt_hash=canonical_json_hash(request.messages),
                catalog_hash=self.catalog_hash,
                method_config=dict(self.config),
                metadata={
                    "max_output_chars": int(
                        item.metadata.get("max_length")
                        or min(int(len(item.text.replace("\n", " ")) * 1.1), len(item.text.replace("\n", " ")) + 500)
                    )
                },
            )
            for item, output, request in zip(items, outputs, requests)
        ]


__all__ = [
    "PROMPT_VERSION",
    "SAMPLED_METHOD",
    "SampledLLMMethod",
    "SampledPromptRequest",
    "render_sampled_messages",
]
