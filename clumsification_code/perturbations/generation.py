"""Generate canonical perturbation layers: select, execute, validate, checkpoint.

Length planning, inference, output validation, and persisted progress each have
one owner. Public entry points remain here for scripts and downstream callers.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import closing, nullcontext
from typing import Any

from clumsification_code.data.candidate_identity import make_original_candidate_id
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import LayerManifestEntry

from .generation_config import prepare_request
from .generation_store import GenerationStore
from .batch_store import BatchGenerationStore
from .length_planning import (
    estimate_chat_prompt_tokens,
    iter_generation_batches,
    plan_context_buckets,
)
from .output_parsing import GenerationValidationError, collect_batch_results
from .schemas import ChatRunner, GenerationRuntime, PerturbationInput, SkippedGeneration
from .vllm_runner import VLLMRunner, run_vllm


class PerturbationGenerationService:
    """Load canonical parents, execute a method, and persist its candidates."""

    def __init__(
        self,
        repository: DatasetRepository,
        *,
        llm_runner: ChatRunner | None = None,
    ):
        self.repository = repository
        self.llm_runner = llm_runner or VLLMRunner()

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
        request = prepare_request(
            dataset_name=self.repository.dataset_name,
            source_layer=source_layer,
            source_method=source_method,
            source_run_id=source_run_id,
            method=method,
            run_id=run_id,
            target_layer=target_layer,
            config=config,
            source_partitions=source_partitions,
            limit=limit,
        )
        store_class = BatchGenerationStore if request.perturbation_source == "LLM" else GenerationStore
        store = store_class(
            self.repository, request, overwrite=overwrite, retry_failed=retry_failed
        )
        with store if isinstance(store, BatchGenerationStore) else nullcontext():
            all_items = self.load_source_items(
                source_layer=request.source_layer,
                source_method=request.source_method,
                source_run_id=request.source_run_id,
                source_partitions=request.source_partitions or None,
                limit=request.limit,
            )
            items = store.select_items(all_items)
            if not items:
                if isinstance(store, BatchGenerationStore):
                    return store.finish()
                assert store.last_entry is not None
                return store.last_entry

            method_config = dict(request.method_config)
            if isinstance(store, BatchGenerationStore):
                method_config.update(store.planning_config())
            if retry_failed:
                # Retry sampling advances without changing the frozen assignment seed.
                method_config["sampling_seed"] = store.generation_seed
            adapter = request.spec.create(method_config)
            runtime = GenerationRuntime(chat_runner=self.llm_runner, attempts=getattr(store, "attempts", {}))
            batches = iter_generation_batches(
                items,
                adapter=adapter,
                runner=self.llm_runner,
                method_config=method_config,
                perturbation_source=request.perturbation_source,
                batch_size=request.batch_size,
                on_plan=store.save_plan if isinstance(store, BatchGenerationStore) else None,
                attempts=getattr(store, "attempts", {}),
            )
            if request.perturbation_source == "LLM" and hasattr(self.llm_runner, "generate_batches"):
                completed = self.llm_runner.generate_batches(
                    batches, adapter=adapter, runtime=runtime, batch_size=request.batch_size,
                )
            else:
                completed = (
                    (batch, list(adapter.generate(batch, runtime)),
                     getattr(self.llm_runner, "last_context_stats", None))
                    for batch in batches
                )
            with closing(completed):
                for batch, results, stats in completed:
                    store.record_context_stats(stats)
                    candidates, failures = collect_batch_results(
                        request, batch, results, store.candidate_counts
                    )
                    store.checkpoint(batch, candidates, failures)
            if isinstance(store, BatchGenerationStore):
                return store.finish()
            assert store.last_entry is not None
            return store.last_entry


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
