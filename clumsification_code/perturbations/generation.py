# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Reusable service for generating one canonical perturbation layer."""
from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from typing import Any, Sequence

from clumsification_code.data.candidate_identity import (
    make_candidate_id,
    make_original_candidate_id,
)
from clumsification_code.data.io import canonical_json_hash
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import (
    CandidateRecord,
    GenerationSpec,
    LayerManifestEntry,
)

from .registry import get_method_spec
from .schemas import ChatRunner, GenerationRuntime, PerturbationInput


class GenerationValidationError(ValueError):
    """A method returned output that cannot become a canonical candidate."""


def _parse_vllm_text(output: Any) -> str:
    text = str(output.outputs[0].text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip("\n'")


def run_vllm(
    model_path: str,
    prompts: list[list[dict[str, str]]],
    temperature: float,
    max_tokens: int,
) -> Sequence[str]:
    """Execute an LLM batch; heavy optional dependencies stay lazy."""
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("LLM methods require vLLM and torch") from exc
    llm = LLM(
        model=model_path,
        max_model_len=max_tokens * 2,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        language_model_only=True,
    )
    outputs = llm.chat(
        prompts,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
    )
    return [_parse_vllm_text(output) for output in outputs]


class PerturbationGenerationService:
    """Load canonical parents, execute a method, and persist its candidates."""

    def __init__(
        self,
        repository: DatasetRepository,
        *,
        llm_runner: ChatRunner | None = None,
    ):
        self.repository = repository
        self.llm_runner = llm_runner or run_vllm

    def load_source_items(
        self,
        *,
        source_layer: int,
        source_method: str | None,
        source_run_id: str | None,
        limit: int | None = None,
    ) -> list[PerturbationInput]:
        if isinstance(source_layer, bool) or not isinstance(source_layer, int) or source_layer < 0:
            raise ValueError("source_layer must be a non-negative integer")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        if source_layer == 0:
            if source_method is not None or source_run_id is not None:
                raise ValueError(
                    "source_method/source_run_id must be omitted for original inputs"
                )
            items = [
                PerturbationInput(
                    dataset_name=self.repository.dataset_name,
                    base_text_id=record.base_text_id,
                    text=record.text,
                    parent_candidate_id=make_original_candidate_id(
                        dataset_name=self.repository.dataset_name,
                        base_text_id=record.base_text_id,
                    ),
                    metadata={**record.metadata},
                )
                for record in self.repository.read_originals()
            ]
        else:
            if not source_method or not source_run_id:
                raise ValueError(
                    "source_method and source_run_id are required for perturbed inputs"
                )
            entry = self.repository.get_layer(source_method, source_run_id, source_layer)
            items = [
                PerturbationInput(
                    dataset_name=record.dataset_name,
                    base_text_id=record.base_text_id,
                    text=record.text,
                    source_layer=source_layer,
                    source_method=source_method,
                    source_run_id=source_run_id,
                    parent_candidate_id=record.candidate_id,
                    metadata={**record.metadata, "candidate_id": record.candidate_id},
                )
                for record in self.repository.read_candidates(entry)
            ]
        selected = items if limit is None else items[:limit]
        if not selected:
            raise ValueError("Source layer is empty")
        return selected

    def generate_layer(
        self,
        *,
        source_layer: int,
        source_method: str | None,
        source_run_id: str | None,
        method: str,
        run_id: str = "default",
        target_layer: int | None = None,
        config: dict[str, Any] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
    ) -> LayerManifestEntry:
        resolved_source_run_id = None if source_layer == 0 else source_run_id
        resolved_target = source_layer + 1 if target_layer is None else int(target_layer)
        method_config = dict(config or {})
        GenerationSpec(
            method=method,
            run_id=run_id,
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=resolved_source_run_id,
            target_layer=resolved_target,
            limit=limit,
            config=method_config,
        ).validate()
        spec = get_method_spec(method)
        allow_unchanged = method_config.get("allow_unchanged", False)
        if not isinstance(allow_unchanged, bool):
            raise ValueError("allow_unchanged must be true or false")
        method_config.update(
            {
                "language": method_config.get("language", "english"),
                "seed": int(method_config.get("seed", 42)),
                "target_layer": resolved_target,
                "run_id": run_id,
            }
        )
        persisted_config = {
            key: value for key, value in method_config.items() if key != "store"
        }
        canonical_json_hash(persisted_config)
        destination = self.repository.layer_path(method, run_id, resolved_target)
        if not overwrite:
            identity = (method, run_id, resolved_target)
            if destination.exists() or any(
                entry.identity == identity for entry in self.repository.list_layers()
            ):
                raise FileExistsError(
                    "Canonical generation destination already exists for "
                    f"method={method!r}, run_id={run_id!r}, "
                    f"target_layer={resolved_target}"
                )
        items = self.load_source_items(
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=resolved_source_run_id,
            limit=limit,
        )
        adapter = spec.create(method_config)
        results = list(
            adapter.generate(items, GenerationRuntime(chat_runner=self.llm_runner))
        )
        parent_base_ids: dict[str, str] = {}
        for item in items:
            candidate_id = item.candidate_id
            if candidate_id is None or candidate_id in parent_base_ids:
                raise ValueError("Generation inputs must have unique candidate identities")
            parent_base_ids[candidate_id] = item.base_text_id
        candidate_counts: dict[str, int] = defaultdict(int)
        candidates = []
        input_by_parent = {str(item.candidate_id): item for item in items}
        for result in results:
            parent_id = result.parent_candidate_id
            if parent_id not in parent_base_ids:
                raise ValueError("Generated result references an unknown parent candidate")
            if result.base_text_id != parent_base_ids[parent_id]:
                raise ValueError("Generated result and parent have different base_text_id values")
            expected = (
                self.repository.dataset_name,
                method,
                spec.perturbation_source,
                run_id,
                source_layer,
                source_method,
                resolved_source_run_id,
                resolved_target,
            )
            actual = (
                result.dataset_name,
                result.perturbation_method,
                result.perturbation_source,
                result.run_id,
                result.source_layer,
                result.source_method,
                result.source_run_id,
                result.target_layer,
            )
            if actual != expected:
                raise GenerationValidationError(
                    "Generated result provenance does not match the request"
                )
            if not isinstance(result.text, str) or not result.text.strip():
                raise GenerationValidationError(
                    f"Method returned empty output for parent {parent_id!r}"
                )
            source_text = input_by_parent[parent_id].text
            if not allow_unchanged and result.text.strip() == source_text.strip():
                raise GenerationValidationError(
                    f"Method returned unchanged output for parent {parent_id!r}"
                )
            max_output_chars = result.metadata.get("max_output_chars")
            if max_output_chars is not None:
                if (
                    isinstance(max_output_chars, bool)
                    or not isinstance(max_output_chars, int)
                    or max_output_chars < 1
                ):
                    raise GenerationValidationError(
                        "max_output_chars must be a positive integer"
                    )
                if len(result.text) > max_output_chars:
                    raise GenerationValidationError(
                        f"Output for parent {parent_id!r} exceeds its "
                        f"{max_output_chars}-character limit"
                    )
            candidate_index = candidate_counts[parent_id]
            candidate_counts[parent_id] += 1
            metadata = dict(result.metadata)
            if result.edit_count is not None:
                metadata["edit_count"] = result.edit_count
            candidates.append(
                CandidateRecord(
                    dataset_name=result.dataset_name,
                    base_text_id=result.base_text_id,
                    candidate_id=make_candidate_id(
                        dataset_name=result.dataset_name,
                        perturbation_method=method,
                        run_id=run_id,
                        base_text_id=result.base_text_id,
                        target_layer=resolved_target,
                        parent_candidate_id=parent_id,
                        candidate_index=candidate_index,
                    ),
                    candidate_index=candidate_index,
                    text=result.text,
                    perturbation_method=method,
                    perturbation_source=spec.perturbation_source,
                    run_id=run_id,
                    source_layer=source_layer,
                    source_method=source_method,
                    source_run_id=resolved_source_run_id,
                    target_layer=resolved_target,
                    parent_candidate_id=parent_id,
                    perturbation_edits=tuple(result.perturbation_edits),
                    target_dimensions=tuple(result.target_dimensions),
                    severity=result.severity,
                    generator=result.generator,
                    seed=result.seed,
                    prompt_version=result.prompt_version,
                    prompt_hash=result.prompt_hash,
                    catalog_hash=result.catalog_hash,
                    metadata=metadata,
                )
            )
        if set(candidate_counts) != set(parent_base_ids):
            missing = len(set(parent_base_ids) - set(candidate_counts))
            raise ValueError(f"Generated layer has no candidate for {missing} input parent(s)")
        return self.repository.write_candidate_layer(
            candidates,
            method=method,
            run_id=run_id,
            target_layer=resolved_target,
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=resolved_source_run_id,
            config=persisted_config,
            input_count=len(items),
            overwrite=overwrite,
        )


def load_source_items(
    dataset: str,
    *,
    source_layer: int,
    source_method: str | None,
    source_run_id: str | None = None,
    limit: int | None = None,
    dataset_root: str | Path = "data/custom_datasets",
) -> list[PerturbationInput]:
    repository = DatasetRepository.from_root(dataset_root, dataset)
    return PerturbationGenerationService(repository).load_source_items(
        source_layer=source_layer,
        source_method=source_method,
        source_run_id=source_run_id,
        limit=limit,
    )


def generate_layer(
    dataset: str,
    *,
    source_layer: int,
    source_method: str | None,
    source_run_id: str | None = None,
    method: str,
    run_id: str = "default",
    target_layer: int | None = None,
    config: dict[str, Any] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    dataset_root: str | Path = "data/custom_datasets",
    llm_runner: ChatRunner | None = None,
) -> Path:
    repository = DatasetRepository.from_root(dataset_root, dataset)
    entry = PerturbationGenerationService(
        repository, llm_runner=llm_runner
    ).generate_layer(
        source_layer=source_layer,
        source_method=source_method,
        source_run_id=source_run_id,
        method=method,
        run_id=run_id,
        target_layer=target_layer,
        config=config,
        limit=limit,
        overwrite=overwrite,
    )
    return repository.dataset_dir / entry.path


__all__ = [
    "ChatRunner",
    "GenerationValidationError",
    "PerturbationGenerationService",
    "generate_layer",
    "load_source_items",
    "run_vllm",
]
