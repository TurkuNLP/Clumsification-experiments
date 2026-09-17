"""Resume selection and batch checkpoints in the existing canonical format.

This component owns mutable run state. Storage layout and checkpoint ordering
are deliberately unchanged; a future batch-file store can replace it without
changing inference or output validation.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from clumsification_code.data.io import read_json, write_json_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import CandidateRecord, LayerManifestEntry

from .generation_config import GenerationRequest, request_config
from .schemas import PerturbationInput


class GenerationStore:
    """Own the successful candidates, retry state, and checkpoint lifecycle."""

    def __init__(
        self,
        repository: DatasetRepository,
        request: GenerationRequest,
        *,
        overwrite: bool,
        retry_failed: bool,
    ):
        self.repository = repository
        self.request = request
        self.overwrite = overwrite
        self.retry_failed = retry_failed
        self.destination = repository.layer_path(
            request.method, request.run_id, request.target_layer
        )
        self.progress_path = self.destination.with_suffix(".progress.json")
        self.existing_entry = next(
            (
                entry for entry in repository.list_layers()
                if entry.identity == (request.method, request.run_id, request.target_layer)
            ),
            None,
        )
        if retry_failed and overwrite:
            raise ValueError("retry_failed and overwrite cannot be used together")
        self.progress = (
            read_json(self.progress_path)
            if self.progress_path.exists() and not overwrite else {}
        )
        if not isinstance(self.progress, dict):
            raise ValueError(f"Invalid generation progress file: {self.progress_path}")
        self._validate_destination()
        self.existing_candidates: list[CandidateRecord] = []
        self.candidates: list[CandidateRecord] = []
        self.attempted_parent_ids: set[str] = set()
        self.failed_parent_ids: set[str] = set()
        self.failure_records: dict[str, dict[str, Any]] = {}
        self.candidate_counts: dict[str, int] = defaultdict(int)
        self.context_bucket_counts: dict[str, int] = defaultdict(int)
        self.retry_round = 0
        self.generation_seed = request.generation_seed
        self.input_count = 0
        self.selected_input_count = 0
        self.last_entry = self.existing_entry

    def _validate_destination(self) -> None:
        request = self.request
        resumable = request.perturbation_source == "LLM" and not self.overwrite
        if not self.retry_failed:
            if (self.destination.exists() or self.existing_entry is not None) and not resumable:
                raise FileExistsError(
                    "Canonical generation destination already exists for "
                    f"method={request.method!r}, run_id={request.run_id!r}, "
                    f"target_layer={request.target_layer}"
                )
        elif self.existing_entry is None and not self.progress_path.exists():
            raise FileNotFoundError(
                "retry_failed requires an existing canonical generation layer for "
                f"method={request.method!r}, run_id={request.run_id!r}, "
                f"target_layer={request.target_layer}"
            )
        entry = self.existing_entry
        if entry is not None and (
            entry.source_layer != request.source_layer
            or entry.source_method != request.source_method
            or entry.source_run_id != request.source_run_id
            or request_config(entry.config) != request_config(request.persisted_config)
        ):
            mode = "retry_failed" if self.retry_failed else "Resume"
            raise ValueError(
                f"{mode} request does not match the existing layer's immutable "
                "source or generation configuration"
            )

    def select_items(self, all_items: list[PerturbationInput]) -> list[PerturbationInput]:
        """Restore progress and select either unattempted or explicitly failed inputs."""
        self.input_count = len(all_items)
        entry = self.existing_entry
        if entry is not None and not self.overwrite:
            if entry.input_count != self.input_count:
                raise ValueError(
                    "retry_failed source selection does not match the existing layer's "
                    "input count (check --source-partitions and --limit)"
                )
            self.existing_candidates = list(self.repository.read_candidates(entry))
            parent_ids = [record.parent_candidate_id for record in self.existing_candidates]
            if len(parent_ids) != len(set(parent_ids)):
                raise ValueError("Existing retry layer has duplicate parent candidates")
        existing_config = entry.config if entry else {}
        if self.retry_failed:
            self.retry_round = len(existing_config.get("retry_history", [])) + 1
            self.generation_seed += self.retry_round

        existing_parent_ids = {record.parent_candidate_id for record in self.existing_candidates}
        self.attempted_parent_ids = set(self.progress.get("attempted_parent_ids", []))
        self.failed_parent_ids = set(self.progress.get("failed_parent_ids", []))
        self.attempted_parent_ids.update(existing_config.get("attempted_parent_ids", []))
        self.failed_parent_ids.update(existing_config.get("failed_parent_ids", []))
        self.attempted_parent_ids.update(existing_parent_ids)
        # A canonical success wins when cancellation leaves the sidecar stale.
        self.failed_parent_ids.difference_update(existing_parent_ids)

        for record in self.existing_candidates:
            self.candidate_counts[record.parent_candidate_id] = max(
                self.candidate_counts[record.parent_candidate_id], record.candidate_index + 1
            )
        persisted_failures = [
            *existing_config.get("unresolved_failures", []),
            *self.progress.get("failures", []),
        ]
        self.failure_records = {
            str(failure["parent_candidate_id"]): dict(failure)
            for failure in persisted_failures
            if (
                isinstance(failure, dict)
                and "parent_candidate_id" in failure
                and str(failure["parent_candidate_id"]) in self.failed_parent_ids
            )
        }
        selected_parent_ids = (
            self.failed_parent_ids if self.retry_failed
            else {str(item.candidate_id) for item in all_items} - self.attempted_parent_ids
        )
        items = [item for item in all_items if str(item.candidate_id) in selected_parent_ids]
        self.selected_input_count = len(items)
        if not items and entry is None:
            raise ValueError("No inputs are eligible for this generation mode")
        parent_ids = [item.candidate_id for item in items]
        if None in parent_ids or len(parent_ids) != len(set(parent_ids)):
            raise ValueError("Generation inputs must have unique candidate identities")
        return items

    def record_context_stats(self, stats: Any) -> None:
        if isinstance(stats, dict):
            for bucket, count in stats.get("bucket_counts", {}).items():
                self.context_bucket_counts[str(bucket)] += int(count)

    def _checkpoint_config(self) -> dict[str, Any]:
        config = dict(self.request.persisted_config)
        config.update(
            attempted_parent_ids=sorted(self.attempted_parent_ids),
            failed_parent_ids=sorted(self.failed_parent_ids),
            completed_input_count=len(self.attempted_parent_ids),
            generation_complete=len(self.attempted_parent_ids) == self.input_count,
            unresolved_failure_count=len(self.failed_parent_ids),
        )
        if self.failure_records:
            config["unresolved_failures"] = self._ordered_failures()
        if self.context_bucket_counts:
            config["bucket_counts"] = dict(self.context_bucket_counts)
        if self.retry_failed:
            history = (self.existing_entry.config if self.existing_entry else {}).get(
                "retry_history", []
            )
            config["retry_history"] = [
                *history,
                {
                    "round": self.retry_round,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "effective_seed": self.generation_seed,
                    "attempted_input_count": self.selected_input_count,
                    "recovered_output_count": len(self.candidates),
                    "remaining_failure_count": len(self.failed_parent_ids),
                },
            ]
            config["retry_round"] = self.retry_round
        return config

    def _ordered_failures(self) -> list[dict[str, Any]]:
        return [self.failure_records[key] for key in sorted(self.failure_records)]

    def checkpoint(
        self,
        batch: list[PerturbationInput],
        candidates: list[CandidateRecord],
        failures: dict[str, dict[str, Any]],
    ) -> LayerManifestEntry:
        """Commit the canonical layer before its advisory progress sidecar."""
        parent_ids = {str(item.candidate_id) for item in batch}
        self.candidates.extend(candidates)
        self.attempted_parent_ids.update(parent_ids)
        self.failed_parent_ids.difference_update(parent_ids)
        self.failed_parent_ids.update(failures)
        for parent_id in parent_ids:
            self.failure_records.pop(parent_id, None)
        self.failure_records.update(failures)
        merged_candidates = [*self.existing_candidates, *self.candidates]
        checkpoint_config = self._checkpoint_config()
        progress = {
            "schema_version": 1,
            "attempted_parent_ids": sorted(self.attempted_parent_ids),
            "failed_parent_ids": sorted(self.failed_parent_ids),
            "failures": self._ordered_failures(),
            "input_count": self.input_count,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.last_entry = self.repository.write_candidate_layer(
            merged_candidates,
            **self.request.layer_kwargs,
            config=checkpoint_config,
            input_count=self.input_count,
            overwrite=True,
        )
        write_json_atomic(self.progress_path, progress, overwrite=True)
        print(
            f"Checkpoint: attempted={len(self.attempted_parent_ids)}/{self.input_count}, "
            f"successful={len(merged_candidates)}, failed={len(self.failed_parent_ids)}",
            flush=True,
        )
        return self.last_entry
