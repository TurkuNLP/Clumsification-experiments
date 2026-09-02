# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Reusable service for generating one canonical perturbation layer."""
from __future__ import annotations

from collections import defaultdict
import gc
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
from .schemas import ChatRunner, GenerationRuntime, PerturbationInput, SkippedGeneration


class GenerationValidationError(ValueError):
    """A method returned output that cannot become a canonical candidate."""


LEGACY_TEXT_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384)
LEGACY_PROMPT_OVERHEAD = 512


def _legacy_bucket_params(text_bucket: int) -> tuple[int, int]:
    """Return the legacy context and output limits for a text bucket."""
    thinking = max(1024, int(text_bucket * 0.45))
    total = LEGACY_PROMPT_OVERHEAD + text_bucket + text_bucket + thinking + 128
    if total > 4000:
        thinking = 4096
        total = LEGACY_PROMPT_OVERHEAD + text_bucket + text_bucket + thinking + 128
    model_len = 1
    while model_len < total:
        model_len *= 2
    return model_len, text_bucket + thinking + 256


def _source_text_from_prompt(messages: list[dict[str, str]]) -> str:
    content = str(messages[-1].get("content", "")) if messages else ""
    for marker in ("Source text:\n", "Now, edit this text: \n"):
        if marker in content:
            return content.rsplit(marker, 1)[1]
    return content


def _automatic_context_buckets(max_model_len: int) -> tuple[int, ...]:
    """Return ascending power-of-two buckets ending at ``max_model_len``."""
    if max_model_len < 1:
        raise ValueError("max_model_len must be a positive integer")
    buckets: list[int] = []
    bucket = 4096
    while bucket < max_model_len:
        buckets.append(bucket)
        bucket *= 2
    if not buckets or buckets[-1] != max_model_len:
        buckets.append(max_model_len)
    return tuple(buckets)


def estimate_chat_prompt_tokens(
    model: str,
    messages: list[dict[str, str]],
    *,
    tokenizer: Any | None = None,
) -> int:
    """Count tokens in a rendered chat request using the model tokenizer."""
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("LLM length estimation requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        encoded = tokenizer(
            "\n".join(str(message.get("content", "")) for message in messages),
            add_special_tokens=True,
        )["input_ids"]
    return len(encoded)


def plan_context_buckets(
    model: str,
    prompts: list[list[dict[str, str]]],
    *,
    max_model_len: int,
    max_tokens: int,
    tokenizer: Any | None = None,
    safety_margin: int = 256,
    bucket_policy: str = "legacy_text_length",
) -> tuple[dict[int, list[int]], list[dict[str, int]]]:
    """Assign prompts to automatic context buckets and report over-limit items.

    Returned bucket values contain original prompt indices. Over-limit entries
    contain their index and measured lengths, allowing the execution layer to
    skip them without changing the order of eligible requests.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if safety_margin < 0:
        raise ValueError("safety_margin must be non-negative")
    if bucket_policy not in {"legacy_text_length", "rendered_prompt"}:
        raise ValueError("bucket_policy must be 'legacy_text_length' or 'rendered_prompt'")
    planned: dict[int, list[int]] = {}
    skipped: list[dict[str, int]] = []
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("LLM length estimation requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    for index, messages in enumerate(prompts):
        if bucket_policy == "rendered_prompt":
            if hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            else:
                rendered = tokenizer(
                    "\n".join(str(message.get("content", "")) for message in messages),
                    add_special_tokens=True,
                )["input_ids"]
            prompt_tokens = len(rendered)
            required_tokens = prompt_tokens + max_tokens + safety_margin
            bucket = next(
                (boundary for boundary in _automatic_context_buckets(max_model_len)
                 if boundary >= required_tokens),
                None,
            )
            if bucket is None:
                skipped.append({"prompt_index": index, "prompt_tokens": prompt_tokens,
                                "required_tokens": required_tokens})
            else:
                planned.setdefault(bucket, []).append(index)
            continue
        text_tokens = len(tokenizer(_source_text_from_prompt(messages), add_special_tokens=True)["input_ids"])
        text_bucket = next((boundary for boundary in LEGACY_TEXT_BUCKETS if text_tokens <= boundary), None)
        if text_bucket is None:
            text_bucket = LEGACY_TEXT_BUCKETS[-1]
            model_len, bucket_tokens = _legacy_bucket_params(text_bucket)
            skipped.append({"prompt_index": index, "prompt_tokens": text_tokens, "required_tokens": model_len + 1})
            continue
        model_len, bucket_tokens = _legacy_bucket_params(text_bucket)
        required_tokens = model_len
        if required_tokens > max_model_len:
            skipped.append(
                {
                    "prompt_index": index,
                    "prompt_tokens": text_tokens,
                    "required_tokens": required_tokens,
                }
            )
        else:
            planned.setdefault(model_len, []).append(index)
    return {bucket: indices for bucket, indices in planned.items() if indices}, skipped


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
    *,
    max_model_len: int = 32768,
    safety_margin: int = 256,
    bucket_policy: str = "legacy_text_length",
) -> Sequence[str]:
    """Execute prompts sequentially in automatically sized vLLM buckets."""
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("LLM methods require vLLM and torch") from exc
    plan, skipped = plan_context_buckets(
        model_path,
        prompts,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        safety_margin=safety_margin,
        bucket_policy=bucket_policy,
    )
    run_vllm.last_context_stats = {
        "bucket_counts": {str(bucket): len(indices) for bucket, indices in plan.items()},
        "skipped_over_length_count": len(skipped),
    }
    print(
        "[vLLM] context buckets: "
        + ", ".join(f"{bucket}={len(indices)}" for bucket, indices in plan.items())
        + f"; skipped_over_limit={len(skipped)}",
        flush=True,
    )
    results: list[str | SkippedGeneration | None] = [None] * len(prompts)
    for entry in skipped:
        results[entry["prompt_index"]] = SkippedGeneration(
            prompt_tokens=entry["prompt_tokens"],
            required_tokens=entry["required_tokens"],
        )
    for bucket, indices in plan.items():
        llm = LLM(
            model=model_path,
            max_model_len=bucket,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
            language_model_only=True,
        )
        bucket_outputs = llm.chat(
            [prompts[index] for index in indices],
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
        )
        for index, output in zip(indices, bucket_outputs):
            results[index] = _parse_vllm_text(output)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if any(output is None for output in results):
        raise RuntimeError("Bucketed vLLM execution did not produce every output")
    return [output if isinstance(output, SkippedGeneration) else str(output) for output in results]


# Marker consumed by GenerationRuntime without changing the public injected
# runner contract used by tests and downstream callers.
run_vllm.supports_context_buckets = True


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
        # Resolve the historical name before constructing paths/provenance.
        method = spec.name
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
        if spec.perturbation_source == "LLM":
            max_model_len = int(method_config.get("max_model_len", 32768))
            max_tokens = int(method_config.get("max_tokens", 2048))
            safety_margin = int(method_config.get("context_safety_margin", 256))
            if max_model_len < 1 or max_tokens < 1:
                raise ValueError("LLM context and output limits must be positive")
            persisted_config["max_model_len"] = max_model_len
            persisted_config["max_tokens"] = max_tokens
            persisted_config["context_safety_margin"] = safety_margin
            bucket_policy = str(method_config.get("bucket_policy", "legacy_text_length"))
            if bucket_policy not in {"legacy_text_length", "rendered_prompt"}:
                raise ValueError(
                    "bucket_policy must be 'legacy_text_length' or 'rendered_prompt'"
                )
            persisted_config["bucket_policy"] = bucket_policy
            persisted_config["context_buckets"] = list(
                _automatic_context_buckets(max_model_len)
            )
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
        skipped_over_length: list[dict[str, Any]] = []
        input_by_parent = {str(item.candidate_id): item for item in items}
        for result in results:
            parent_id = result.parent_candidate_id
            if parent_id not in parent_base_ids:
                raise ValueError("Generated result references an unknown parent candidate")
            if isinstance(result.text, SkippedGeneration):
                skipped_over_length.append(
                    {
                        "parent_candidate_id": parent_id,
                        "prompt_tokens": result.text.prompt_tokens,
                        "required_tokens": result.text.required_tokens,
                    }
                )
                continue
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
        skipped_ids = {entry["parent_candidate_id"] for entry in skipped_over_length}
        if set(candidate_counts) | skipped_ids != set(parent_base_ids):
            missing = len(set(parent_base_ids) - set(candidate_counts) - skipped_ids)
            raise ValueError(f"Generated layer has no candidate for {missing} input parent(s)")
        if skipped_over_length:
            persisted_config["skipped_over_length_count"] = len(skipped_over_length)
            persisted_config["skipped_over_length"] = skipped_over_length
        execution_stats = getattr(self.llm_runner, "last_context_stats", None)
        if spec.perturbation_source == "LLM" and isinstance(execution_stats, dict):
            persisted_config.update(
                {
                    key: value
                    for key, value in execution_stats.items()
                    if key == "bucket_counts"
                }
            )
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
    "SkippedGeneration",
    "estimate_chat_prompt_tokens",
    "generate_layer",
    "load_source_items",
    "plan_context_buckets",
    "run_vllm",
]
