"""Validated settings shared by generation, validation, and checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from clumsification_code.data.schemas import GenerationSpec

from .registry import MethodSpec, get_method_spec


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
        "n_jobs", "batch_size", "tensor_parallel_size", "gpu_memory_utilization",
        "max_num_seqs", "max_num_batched_tokens", "enable_prefix_caching", "enable_chunked_prefill",
        "device_groups", "accelerator", "replicas", "storage", "journal_path", "failure_counts", "completed_input_count", "source_fingerprint",
        "generation_fingerprint", "elapsed_seconds", "output_snapshot",
        "attempted_parent_ids",
        "failed_parent_ids",
        "completed_input_count",
        "generation_complete",
        "all_outputs_ready",
        "max_output_char_tolerance",
    }
)


def request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the immutable request portion of a persisted config."""
    return {key: value for key, value in config.items() if key not in _ATTEMPT_AUDIT_FIELDS}


@dataclass(frozen=True)
class GenerationRequest:
    """Resolved request settings; mutable progress lives in GenerationStore."""

    dataset_name: str
    spec: MethodSpec
    run_id: str
    source_layer: int
    source_method: str | None
    source_run_id: str | None
    target_layer: int
    source_partitions: tuple[str, ...]
    limit: int | None
    method_config: dict[str, Any]
    persisted_config: dict[str, Any]
    allow_unchanged: bool
    batch_size: int
    generation_seed: int

    @property
    def method(self) -> str:
        return self.spec.name

    @property
    def perturbation_source(self) -> str:
        return self.spec.perturbation_source

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        """Canonical identity and parent-layer settings for repository writes."""
        return {
            "method": self.method,
            "run_id": self.run_id,
            "target_layer": self.target_layer,
            "source_layer": self.source_layer,
            "source_method": self.source_method,
            "source_run_id": self.source_run_id,
        }


def prepare_request(
    *,
    dataset_name: str,
    source_layer: int,
    source_method: str | None,
    source_run_id: str | None,
    method: str,
    run_id: str,
    target_layer: int | None,
    config: dict[str, Any] | None,
    source_partitions: tuple[str, ...] | None,
    limit: int | None,
) -> GenerationRequest:
    """Validate the public request and apply the existing method defaults."""
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
    batch_size = 0  # Traditional methods retain their single-batch execution.
    if spec.perturbation_source == "LLM":
        defaults = {
            "enable_thinking": True, "thinking_token_cap": 0, "answer_reserve_tokens": 256,
            "output_metadata_tokens": 256, "structured_output": True,
            "tensor_parallel_size": 1, "max_num_seqs": 64, "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
            "dtype": "bfloat16", "enable_prefix_caching": True, "enable_chunked_prefill": True,
        }
        for name, default in defaults.items():
            method_config.setdefault(name, default)
        for name in ("enable_thinking", "structured_output", "enable_prefix_caching", "enable_chunked_prefill"):
            if type(method_config[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("thinking_token_cap", "answer_reserve_tokens", "output_metadata_tokens"):
            if type(method_config[name]) is not int or method_config[name] < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens"):
            if type(method_config[name]) is not int or method_config[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < method_config["gpu_memory_utilization"] < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if method_config.get("source_buckets") is not None:
            values = method_config["source_buckets"]
            if not values or any(type(v) is not int or v < 1 for v in values) or sorted(set(values)) != values:
                raise ValueError("source_buckets must be increasing positive integers")
        persisted_config.update(method_config)
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
    return GenerationRequest(
        dataset_name=dataset_name,
        spec=spec,
        run_id=run_id,
        source_layer=source_layer,
        source_method=source_method,
        source_run_id=resolved_source_run_id,
        target_layer=resolved_target,
        source_partitions=normalized_partitions,
        limit=limit,
        method_config=method_config,
        persisted_config=persisted_config,
        allow_unchanged=allow_unchanged,
        batch_size=batch_size,
        generation_seed=generation_seed,
    )
