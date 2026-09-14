# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Reusable service for generating one canonical perturbation layer."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import re
from pathlib import Path
from typing import Any, Sequence

from clumsification_code.data.candidate_identity import (
    make_candidate_id,
    make_original_candidate_id,
)
from clumsification_code.data.io import read_json, write_json_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import (
    CandidateRecord,
    GenerationSpec,
    LayerManifestEntry,
)

from .registry import get_method_spec
from .schemas import (
    ChatRunner,
    GenerationRuntime,
    PerturbationInput,
    SkippedGeneration,
    SkippedPerturbation,
)


class GenerationValidationError(ValueError):
    """A method returned output that cannot become a canonical candidate."""


LEGACY_TEXT_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384)
LEGACY_PROMPT_OVERHEAD = 512
_ACTIVE_VLLM_ENGINE: Any | None = None
_ACTIVE_VLLM_ENGINE_KEY: tuple[str, int, int] | None = None

# These fields describe the outcome of a generation attempt rather than its
# reproducible request configuration.  They must not make a retry of the same
# layer look incompatible with its original invocation.
_ATTEMPT_AUDIT_FIELDS = frozenset(
    {
        "skipped_over_length_count",
        "skipped_over_length",
        "skipped_invalid_output_count",
        "skipped_invalid_output",
        "retried_input_count",
        "retry_attempt_count",
        "bucket_counts",
        "retry_history",
        "retry_round",
        "unresolved_failure_count",
        "unresolved_failures",
        "n_jobs",
        "attempted_parent_ids",
        "failed_parent_ids",
        "completed_input_count",
        "generation_complete",
        "max_output_char_tolerance",
    }
)


def _request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the immutable request portion of a persisted config."""
    return {key: value for key, value in config.items() if key not in _ATTEMPT_AUDIT_FIELDS}


def _legacy_bucket_params(text_bucket: int) -> tuple[int, int]:
    """Return legacy context sizing and a no-thinking output allowance."""
    # Keep the original reserved headroom when sizing the context. Besides
    # reasoning, that headroom protects the variable sampled-operation prompt
    # from overflowing its bucket. It is deliberately not made available to
    # generation now that thinking is disabled.
    reserved_headroom = max(1024, int(text_bucket * 0.45))
    total = LEGACY_PROMPT_OVERHEAD + text_bucket + text_bucket + reserved_headroom + 128
    if total > 4000:
        reserved_headroom = 4096
        total = LEGACY_PROMPT_OVERHEAD + text_bucket + text_bucket + reserved_headroom + 128
    output_tokens = text_bucket + 256
    model_len = 1
    while model_len < total:
        model_len *= 2
    return model_len, output_tokens


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
    tokenizer: Any | None = None,
) -> tuple[dict[int, list[int]], list[dict[str, int]]]:
    """Assign prompts using the original text-length bucket implementation.

    Returned bucket values contain original prompt indices. Over-limit entries
    contain their index and measured lengths, allowing the execution layer to
    skip them without changing the order of eligible requests.
    """
    groups, skipped = _plan_legacy_bucket_groups(
        model, prompts, max_model_len=max_model_len, tokenizer=tokenizer
    )
    return {
        model_len: list(group["indices"])
        for model_len, group in groups.items()
    }, skipped


def _plan_legacy_bucket_groups(
    model: str,
    prompts: list[list[dict[str, str]]],
    *,
    max_model_len: int,
    tokenizer: Any | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, int]]]:
    """Return merged legacy buckets including their automatic output limits."""
    if max_model_len < 1:
        raise ValueError("max_model_len must be a positive integer")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("LLM length estimation requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_buckets: dict[int, list[tuple[int, int]]] = {}
    for index, messages in enumerate(prompts):
        text_tokens = len(
            tokenizer(
                _source_text_from_prompt(messages), add_special_tokens=False
            )["input_ids"]
        )
        text_bucket = next((boundary for boundary in LEGACY_TEXT_BUCKETS if text_tokens <= boundary), None)
        if text_bucket is None:
            text_bucket = 1
            while text_bucket < text_tokens:
                text_bucket *= 2
        raw_buckets.setdefault(text_bucket, []).append((index, text_tokens))

    groups: dict[int, dict[str, Any]] = {}
    skipped: list[dict[str, int]] = []
    for text_bucket in sorted(raw_buckets):
        model_len, output_tokens = _legacy_bucket_params(text_bucket)
        entries = raw_buckets[text_bucket]
        if model_len > max_model_len:
            skipped.extend(
                {
                    "prompt_index": index,
                    "prompt_tokens": text_tokens,
                    "required_tokens": model_len,
                }
                for index, text_tokens in entries
            )
            continue
        group = groups.setdefault(
            model_len,
            {"indices": [], "text_buckets": [], "max_tokens": 0},
        )
        group["indices"].extend(index for index, _ in entries)
        group["text_buckets"].append(text_bucket)
        group["max_tokens"] = max(group["max_tokens"], output_tokens)
    return groups, skipped


def _parse_vllm_text(output: Any) -> str:
    text = str(output.outputs[0].text)
    if "<think>" in text and "</think>" not in text:
        return ""
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
    seed: int = 42,
) -> Sequence[str]:
    """Execute prompts sequentially in automatically sized vLLM buckets."""
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("LLM methods require vLLM and torch") from exc
    groups, skipped = _plan_legacy_bucket_groups(
        model_path,
        prompts,
        max_model_len=max_model_len,
    )
    run_vllm.last_context_stats = {
        "bucket_counts": {
            str(bucket): len(group["indices"])
            for bucket, group in groups.items()
        },
        "skipped_over_length_count": len(skipped),
    }
    print(
        "[vLLM] context buckets: "
        + ", ".join(
            f"{bucket}={len(group['indices'])} (max_tokens={group['max_tokens']})"
            for bucket, group in groups.items()
        )
        + f"; skipped_over_limit={len(skipped)}",
        flush=True,
    )
    results: list[str | SkippedGeneration | None] = [None] * len(prompts)
    for entry in skipped:
        results[entry["prompt_index"]] = SkippedGeneration(
            prompt_tokens=entry["prompt_tokens"],
            required_tokens=entry["required_tokens"],
        )
    global _ACTIVE_VLLM_ENGINE, _ACTIVE_VLLM_ENGINE_KEY
    for bucket, group in groups.items():
        indices = group["indices"]
        tensor_parallel_size = max(1, torch.cuda.device_count())
        engine_key = (model_path, bucket, tensor_parallel_size)
        if _ACTIVE_VLLM_ENGINE_KEY != engine_key:
            if "llm" in locals():
                del llm
            _ACTIVE_VLLM_ENGINE = None
            _ACTIVE_VLLM_ENGINE_KEY = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _ACTIVE_VLLM_ENGINE = LLM(
                model=model_path,
                max_model_len=bucket,
                tensor_parallel_size=tensor_parallel_size,
                language_model_only=True,
            )
            _ACTIVE_VLLM_ENGINE_KEY = engine_key
        llm = _ACTIVE_VLLM_ENGINE
        bucket_outputs = llm.chat(
            [prompts[index] for index in indices],
            sampling_params=SamplingParams(
                max_tokens=group["max_tokens"], temperature=temperature, seed=seed
            ),
            chat_template_kwargs={"enable_thinking": False},
        )
        for index, output in zip(indices, bucket_outputs):
            results[index] = _parse_vllm_text(output)
    if any(output is None for output in results):
        raise RuntimeError("Bucketed vLLM execution did not produce every output")
    return [output if isinstance(output, SkippedGeneration) else str(output) for output in results]


# Marker consumed by GenerationRuntime without changing the public injected
# runner contract used by tests and downstream callers.
run_vllm.supports_context_buckets = True
run_vllm.supports_persistent_bucket_batches = True


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
        source_partitions: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[PerturbationInput]:
        if isinstance(source_layer, bool) or not isinstance(source_layer, int) or source_layer < 0:
            raise ValueError("source_layer must be a non-negative integer")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        selected_partitions = (
            set(source_partitions) if source_partitions is not None else None
        )
        if selected_partitions is not None and (
            not selected_partitions
            or any(not isinstance(value, str) or not value for value in selected_partitions)
        ):
            raise ValueError("source_partitions must be a non-empty sequence of names")
        originals = self.repository.read_originals()
        partition_by_original = self.repository.read_split_assignments()
        if selected_partitions is not None and partition_by_original is None:
            raise FileNotFoundError(
                "source_partitions requires split_assignments.jsonl"
            )

        def is_selected(base_text_id: str) -> bool:
            if selected_partitions is None:
                return True
            assert partition_by_original is not None
            partition = partition_by_original.get(base_text_id)
            if not isinstance(partition, str) or not partition:
                raise ValueError(f"Original source {base_text_id!r} has no valid split assignment")
            return partition in selected_partitions

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
                for record in originals
                if is_selected(record.base_text_id)
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
                if is_selected(record.base_text_id)
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
        source_partitions: tuple[str, ...] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
        retry_failed: bool = False,
    ) -> LayerManifestEntry:
        resolved_source_run_id = None if source_layer == 0 else source_run_id
        resolved_target = source_layer + 1 if target_layer is None else int(target_layer)
        method_config = dict(config or {})
        if source_partitions is not None and isinstance(source_partitions, str):
            raise ValueError("source_partitions must be a sequence of names, not a string")
        normalized_partitions = tuple(source_partitions or ())
        GenerationSpec(
            method=method,
            run_id=run_id,
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=resolved_source_run_id,
            target_layer=resolved_target,
            limit=limit,
            source_partitions=normalized_partitions,
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
        # ``seed`` defines the deterministic request and remains immutable
        # across resumed runs. ``sampling_seed`` controls candidate generation
        # and may advance for a retry without changing that request.
        generation_seed = int(method_config.get("sampling_seed", method_config["seed"]))
        persisted_config = {
            key: value for key, value in method_config.items() if key != "store"
        }
        if normalized_partitions:
            persisted_config["source_partitions"] = list(normalized_partitions)
        if spec.perturbation_source == "LLM":
            max_model_len = int(method_config.get("max_model_len", 32768))
            if max_model_len < 1:
                raise ValueError("LLM context limit must be positive")
            persisted_config["max_model_len"] = max_model_len
            max_retries = int(method_config.get("max_retries", 0))
            if max_retries != 0:
                raise ValueError(
                    "LLM generation now gives each item one chance per submission; "
                    "max_retries must be 0"
                )
            method_config["max_retries"] = max_retries
            persisted_config["max_retries"] = max_retries
            batch_size = int(method_config.get("batch_size", 512))
            if batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            method_config["batch_size"] = batch_size
            persisted_config["batch_size"] = batch_size
            output_char_tolerance = int(
                method_config.get("max_output_char_tolerance", 256)
            )
            if output_char_tolerance < 0:
                raise ValueError("max_output_char_tolerance must be non-negative")
            method_config["max_output_char_tolerance"] = output_char_tolerance
            persisted_config["max_output_char_tolerance"] = output_char_tolerance
        elif spec.perturbation_source == "trad":
            max_attempts = int(method_config.get("max_attempts", 100))
            if max_attempts < 1:
                raise ValueError("max_attempts must be at least 1")
            method_config["max_attempts"] = max_attempts
            persisted_config["max_attempts"] = max_attempts
        json.dumps(persisted_config, ensure_ascii=False, allow_nan=False, sort_keys=True)
        destination = self.repository.layer_path(method, run_id, resolved_target)
        identity = (method, run_id, resolved_target)
        existing_entry = next(
            (entry for entry in self.repository.list_layers() if entry.identity == identity),
            None,
        )
        if retry_failed and overwrite:
            raise ValueError("retry_failed and overwrite cannot be used together")
        progress_path = destination.with_suffix(".progress.json")
        progress = (
            read_json(progress_path)
            if progress_path.exists() and not overwrite
            else {}
        )
        if not isinstance(progress, dict):
            raise ValueError(f"Invalid generation progress file: {progress_path}")
        resumable = spec.perturbation_source == "LLM" and not overwrite
        if not retry_failed:
            if (destination.exists() or existing_entry is not None) and not resumable:
                raise FileExistsError(
                    "Canonical generation destination already exists for "
                    f"method={method!r}, run_id={run_id!r}, "
                    f"target_layer={resolved_target}"
                )
        elif existing_entry is None and not progress_path.exists():
            raise FileNotFoundError(
                "retry_failed requires an existing canonical generation layer for "
                f"method={method!r}, run_id={run_id!r}, target_layer={resolved_target}"
            )
        elif existing_entry is not None and (
            existing_entry.source_layer != source_layer
            or existing_entry.source_method != source_method
            or existing_entry.source_run_id != resolved_source_run_id
            or _request_config(existing_entry.config) != _request_config(persisted_config)
        ):
            raise ValueError(
                "retry_failed request does not match the existing layer's immutable "
                "source or generation configuration"
            )
        if existing_entry is not None and (
            existing_entry.source_layer != source_layer
            or existing_entry.source_method != source_method
            or existing_entry.source_run_id != resolved_source_run_id
            or _request_config(existing_entry.config) != _request_config(persisted_config)
        ):
            raise ValueError(
                "Resume request does not match the existing layer's immutable "
                "source or generation configuration"
            )

        all_items = self.load_source_items(
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=resolved_source_run_id,
            source_partitions=normalized_partitions or None,
            limit=limit,
        )
        existing_candidates: list[CandidateRecord] = []
        retry_round = 0
        if existing_entry is not None and not overwrite:
            if existing_entry.input_count != len(all_items):
                raise ValueError(
                    "retry_failed source selection does not match the existing layer's "
                    "input count (check --source-partitions and --limit)"
                )
            existing_candidates = list(self.repository.read_candidates(existing_entry))
            existing_parent_ids = [record.parent_candidate_id for record in existing_candidates]
            if len(existing_parent_ids) != len(set(existing_parent_ids)):
                raise ValueError("Existing retry layer has duplicate parent candidates")
        if retry_failed:
            retry_round = len((existing_entry.config if existing_entry else {}).get("retry_history", [])) + 1
            # A resumed run must explore a different model trajectory without
            # changing the immutable assignment seed.
            generation_seed += retry_round
            method_config["sampling_seed"] = generation_seed
        existing_parent_ids = {record.parent_candidate_id for record in existing_candidates}
        attempted_parent_ids = set(progress.get("attempted_parent_ids", []))
        failed_parent_ids = set(progress.get("failed_parent_ids", []))
        if existing_entry is not None:
            attempted_parent_ids.update(existing_entry.config.get("attempted_parent_ids", []))
            failed_parent_ids.update(existing_entry.config.get("failed_parent_ids", []))
        attempted_parent_ids.update(existing_parent_ids)
        # The canonical layer is committed before the advisory progress file.
        # If cancellation leaves that sidecar stale, a canonical success wins.
        failed_parent_ids.difference_update(existing_parent_ids)
        if retry_failed:
            selected_parent_ids = failed_parent_ids
        else:
            selected_parent_ids = {
                str(item.candidate_id) for item in all_items
            } - attempted_parent_ids
        items = [item for item in all_items if str(item.candidate_id) in selected_parent_ids]
        if not items:
            if existing_entry is None:
                raise ValueError("No inputs are eligible for this generation mode")
            return existing_entry
        parent_base_ids: dict[str, str] = {}
        for item in items:
            candidate_id = item.candidate_id
            if candidate_id is None or candidate_id in parent_base_ids:
                raise ValueError("Generation inputs must have unique candidate identities")
            parent_base_ids[candidate_id] = item.base_text_id
        input_by_parent = {str(item.candidate_id): item for item in items}
        adapter = spec.create(method_config)
        runtime = GenerationRuntime(chat_runner=self.llm_runner)
        context_bucket_counts: dict[str, int] = defaultdict(int)
        generation_batches: list[list[PerturbationInput]] | None = None

        # Keep consecutive checkpoint batches inside one context bucket.  The
        # built-in runner can therefore retain that bucket's vLLM engine while
        # still committing at most ``batch_size`` items at a time.
        if (
            spec.perturbation_source == "LLM"
            and getattr(self.llm_runner, "supports_persistent_bucket_batches", False)
            and hasattr(adapter, "build_prompts")
        ):
            prompt_items = [
                item.metadata
                | {
                    "base_text_id": item.base_text_id,
                    "candidate_id": item.candidate_id,
                    "text": item.text,
                }
                for item in items
            ]
            prompts = adapter.build_prompts(prompt_items)
            planned_groups, planned_skipped = _plan_legacy_bucket_groups(
                str(method_config.get("model") or method_config.get("model_path")),
                prompts,
                max_model_len=int(method_config.get("max_model_len", 32768)),
            )
            ordered_indices = [
                index
                for group in planned_groups.values()
                for index in group["indices"]
            ]
            ordered_indices.extend(entry["prompt_index"] for entry in planned_skipped)
            if len(ordered_indices) != len(items) or len(set(ordered_indices)) != len(items):
                raise RuntimeError("Context-bucket planning did not preserve every input")
            generation_batches = []
            for group in planned_groups.values():
                bucket_items = [items[index] for index in group["indices"]]
                generation_batches.extend(
                    bucket_items[start : start + batch_size]
                    for start in range(0, len(bucket_items), batch_size)
                )
            skipped_items = [items[entry["prompt_index"]] for entry in planned_skipped]
            generation_batches.extend(
                skipped_items[start : start + batch_size]
                for start in range(0, len(skipped_items), batch_size)
            )

        def record_context_stats() -> None:
            stats = getattr(self.llm_runner, "last_context_stats", None)
            if not isinstance(stats, dict):
                return
            for bucket, count in stats.get("bucket_counts", {}).items():
                context_bucket_counts[str(bucket)] += int(count)

        candidate_counts: dict[str, int] = defaultdict(int)
        for record in existing_candidates:
            candidate_counts[record.parent_candidate_id] = max(
                candidate_counts[record.parent_candidate_id], record.candidate_index + 1
            )
        candidates: list[CandidateRecord] = []
        persisted_failures = [
            *(existing_entry.config.get("unresolved_failures", []) if existing_entry else []),
            *progress.get("failures", []),
        ]
        failure_records: dict[str, dict[str, Any]] = {
            str(entry["parent_candidate_id"]): dict(entry)
            for entry in persisted_failures
            if (
                isinstance(entry, dict)
                and "parent_candidate_id" in entry
                and str(entry["parent_candidate_id"]) in failed_parent_ids
            )
        }
        last_entry = existing_entry
        effective_batch_size = batch_size if spec.perturbation_source == "LLM" else len(items)
        if generation_batches is None:
            generation_batches = [
                items[start : start + effective_batch_size]
                for start in range(0, len(items), effective_batch_size)
            ]
        for batch in generation_batches:
            batch_parent_ids = {str(item.candidate_id) for item in batch}
            batch_input_by_parent = {str(item.candidate_id): item for item in batch}
            results = list(adapter.generate(batch, runtime))
            record_context_stats()
            if {str(result.parent_candidate_id) for result in results} != batch_parent_ids:
                raise ValueError("Generation batch did not return exactly one result per input")
            batch_candidates: list[CandidateRecord] = []
            batch_failures: dict[str, dict[str, Any]] = {}
            for result in results:
                parent_id = str(result.parent_candidate_id)
                item = batch_input_by_parent[parent_id]
                failure: dict[str, Any] | None = None
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
                elif not allow_unchanged and result.text.strip() == str(item.text).strip():
                    failure = {"parent_candidate_id": parent_id, "reason": "unchanged_output"}
                max_output_chars = result.metadata.get("max_output_chars")
                if max_output_chars is not None:
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
                expected = (
                    self.repository.dataset_name, method, spec.perturbation_source,
                    run_id, source_layer, source_method, resolved_source_run_id,
                    resolved_target,
                )
                actual = (
                    result.dataset_name, result.perturbation_method,
                    result.perturbation_source, result.run_id, result.source_layer,
                    result.source_method, result.source_run_id, result.target_layer,
                )
                if actual != expected or result.base_text_id != item.base_text_id:
                    raise GenerationValidationError("Generated result provenance does not match the request")
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
                        edit_count=result.edit_count,
                        generator=result.generator,
                        seed=result.seed,
                        prompt_version=result.prompt_version,
                        metadata=dict(result.metadata),
                    )
                )
            candidates.extend(batch_candidates)
            attempted_parent_ids.update(batch_parent_ids)
            failed_parent_ids.difference_update(batch_parent_ids)
            failed_parent_ids.update(batch_failures)
            for parent_id in batch_parent_ids:
                failure_records.pop(parent_id, None)
            failure_records.update(batch_failures)
            merged_candidates = [*existing_candidates, *candidates]
            completed_count = len(attempted_parent_ids)
            checkpoint_config = dict(persisted_config)
            checkpoint_config.update(
                {
                    "attempted_parent_ids": sorted(attempted_parent_ids),
                    "failed_parent_ids": sorted(failed_parent_ids),
                    "completed_input_count": completed_count,
                    "generation_complete": completed_count == len(all_items),
                    "unresolved_failure_count": len(failed_parent_ids),
                }
            )
            if failure_records:
                checkpoint_config["unresolved_failures"] = [
                    failure_records[key] for key in sorted(failure_records)
                ]
            if context_bucket_counts:
                checkpoint_config["bucket_counts"] = dict(context_bucket_counts)
            if retry_failed:
                retry_history = list(
                    (existing_entry.config if existing_entry else {}).get("retry_history", [])
                )
                checkpoint_config["retry_history"] = [
                    *retry_history,
                    {
                        "round": retry_round,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "effective_seed": generation_seed,
                        "attempted_input_count": len(items),
                        "recovered_output_count": len(candidates),
                        "remaining_failure_count": len(failed_parent_ids),
                    },
                ]
                checkpoint_config["retry_round"] = retry_round
            progress_value = {
                "schema_version": 1,
                "attempted_parent_ids": sorted(attempted_parent_ids),
                "failed_parent_ids": sorted(failed_parent_ids),
                "failures": [failure_records[key] for key in sorted(failure_records)],
                "input_count": len(all_items),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            last_entry = self.repository.write_candidate_layer(
                merged_candidates,
                method=method,
                run_id=run_id,
                target_layer=resolved_target,
                source_layer=source_layer,
                source_method=source_method,
                source_run_id=resolved_source_run_id,
                config=checkpoint_config,
                input_count=len(all_items),
                overwrite=True,
            )
            # The canonical layer is committed first. If cancellation lands
            # between these two atomic writes, its manifest remains sufficient
            # to resume without losing a successful result.
            write_json_atomic(progress_path, progress_value, overwrite=True)
            print(
                f"Checkpoint: attempted={completed_count}/{len(all_items)}, "
                f"successful={len(merged_candidates)}, failed={len(failed_parent_ids)}",
                flush=True,
            )
        assert last_entry is not None
        return last_entry


def load_source_items(
    dataset: str,
    *,
    source_layer: int,
    source_method: str | None,
    source_run_id: str | None = None,
    source_partitions: tuple[str, ...] | None = None,
    limit: int | None = None,
    dataset_root: str | Path = "data/custom_datasets",
) -> list[PerturbationInput]:
    repository = DatasetRepository.from_root(dataset_root, dataset)
    return PerturbationGenerationService(repository).load_source_items(
        source_layer=source_layer,
        source_method=source_method,
        source_run_id=source_run_id,
        source_partitions=source_partitions,
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
    source_partitions: tuple[str, ...] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
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
        source_partitions=source_partitions,
        limit=limit,
        overwrite=overwrite,
        retry_failed=retry_failed,
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
