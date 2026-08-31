# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical repository for original texts and method-separated candidates."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .candidate_identity import make_original_candidate_id
from .io import (
    canonical_json_hash,
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .schemas import (
    CandidateRecord,
    LayerManifestEntry,
    OriginalRecord,
    PerturbationManifest,
    ScoreRecord,
)


def _path_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty path component")
    if "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain path separators")
    return value


class DatasetRepository:
    """Read and write the canonical on-disk representation of one dataset."""

    def __init__(self, dataset_dir: str | Path, *, dataset_name: str | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = _path_component(
            dataset_name or self.dataset_dir.name,
            "dataset_name",
        )

    @classmethod
    def from_root(
        cls,
        dataset_root: str | Path,
        dataset_name: str,
    ) -> "DatasetRepository":
        name = _path_component(dataset_name, "dataset_name")
        return cls(Path(dataset_root) / name, dataset_name=name)

    @property
    def original_path(self) -> Path:
        return self.dataset_dir / "original.jsonl"

    @property
    def perturbation_root(self) -> Path:
        return self.dataset_dir / "perturbations"

    @property
    def manifest_path(self) -> Path:
        return self.perturbation_root / "perturbation_manifest.json"

    @property
    def scores_root(self) -> Path:
        return self.dataset_dir / "scores"

    def score_method_root(self, scoring_method: str) -> Path:
        return self.scores_root / _path_component(scoring_method, "scoring_method")

    def score_path(self, scoring_method: str, scoring_run_id: str) -> Path:
        run = _path_component(scoring_run_id, "scoring_run_id")
        return self.score_method_root(scoring_method) / f"{run}.jsonl"

    def score_error_path(self, scoring_method: str, scoring_run_id: str) -> Path:
        return self.score_path(scoring_method, scoring_run_id).with_suffix(".errors.jsonl")

    def score_metadata_path(self, scoring_method: str, scoring_run_id: str) -> Path:
        return self.score_path(scoring_method, scoring_run_id).with_suffix(".metadata.json")

    def read_scores(
        self,
        *,
        scoring_methods: Iterable[str] | None = None,
        scoring_run_ids: Iterable[str] | None = None,
    ) -> tuple[ScoreRecord, ...]:
        method_filter = set(scoring_methods) if scoring_methods is not None else None
        run_filter = set(scoring_run_ids) if scoring_run_ids is not None else None
        records = []
        if not self.scores_root.is_dir():
            return ()
        for path in sorted(self.scores_root.glob("*/*.jsonl")):
            if path.name.endswith(".errors.jsonl"):
                continue
            method = path.parent.name
            run_id = path.stem
            if method_filter is not None and method not in method_filter:
                continue
            if run_filter is not None and run_id not in run_filter:
                continue
            for row in read_jsonl(path):
                record = ScoreRecord.from_row(row)
                if record.dataset_name != self.dataset_name:
                    raise ValueError(f"Score dataset_name does not match repository: {path}")
                if record.scoring_method != method or record.scoring_run_id != run_id:
                    raise ValueError(f"Score provenance does not match its path: {path}")
                records.append(record)
        identities = [
            (record.candidate_id, record.scoring_method, record.scoring_run_id)
            for record in records
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate candidate/method/run score records")
        return tuple(records)

    def write_scores(
        self,
        records: Iterable[ScoreRecord],
        *,
        scoring_method: str,
        scoring_run_id: str,
        errors: Iterable[dict[str, Any]] = (),
        metadata: dict[str, Any],
        overwrite: bool = False,
    ) -> tuple[Path, Path, Path]:
        values = list(records)
        error_values = [dict(error) for error in errors]
        expected = (self.dataset_name, scoring_method, scoring_run_id)
        for record in values:
            actual = (
                record.dataset_name,
                record.scoring_method,
                record.scoring_run_id,
            )
            if actual != expected:
                raise ValueError("Score provenance does not match requested score run")
        candidate_graph: dict[str, tuple[str, str, str, int]] = {
            make_original_candidate_id(
                dataset_name=self.dataset_name,
                base_text_id=original.base_text_id,
            ): (original.base_text_id, "original", "original", 0)
            for original in self.read_originals()
        }
        for entry in self.list_layers():
            for candidate in self.read_candidates(entry):
                candidate_graph[candidate.candidate_id] = (
                    candidate.base_text_id,
                    candidate.perturbation_method,
                    candidate.run_id,
                    candidate.target_layer,
                )
        for record in values:
            candidate = candidate_graph.get(record.candidate_id)
            if candidate is None:
                raise ValueError(f"Score references unknown candidate {record.candidate_id!r}")
            base_id, method, perturbation_run_id, target_layer = candidate
            if (
                record.base_text_id != base_id
                or record.perturbation_method != method
                or record.target_layer != target_layer
                or record.metadata.get("perturbation_run_id") != perturbation_run_id
            ):
                raise ValueError("Score provenance does not match its candidate")
            if record.reference_candidate_id is not None:
                reference = candidate_graph.get(record.reference_candidate_id)
                if reference is None:
                    raise ValueError("Score references an unknown comparison candidate")
                if reference[0] != base_id or reference[3] != record.source_layer:
                    raise ValueError("Score reference provenance does not match candidate")
        candidate_ids = [record.candidate_id for record in values]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("A score run may contain at most one score per candidate")
        score_path = self.score_path(scoring_method, scoring_run_id)
        error_path = self.score_error_path(scoring_method, scoring_run_id)
        metadata_path = self.score_metadata_path(scoring_method, scoring_run_id)
        destinations = (score_path, error_path, metadata_path)
        if not overwrite and any(path.exists() for path in destinations):
            existing = next(path for path in destinations if path.exists())
            raise FileExistsError(f"Score run output already exists: {existing}")
        canonical_json_hash(metadata)
        for error in error_values:
            canonical_json_hash(error)
        write_jsonl_atomic(score_path, [record.to_row() for record in values], overwrite=overwrite)
        write_jsonl_atomic(error_path, error_values, overwrite=overwrite)
        write_json_atomic(metadata_path, metadata, overwrite=overwrite)
        return score_path, error_path, metadata_path

    def method_root(self, method: str) -> Path:
        return self.perturbation_root / _path_component(method, "method")

    def run_root(self, method: str, run_id: str) -> Path:
        return self.method_root(method) / _path_component(run_id, "run_id")

    def layer_path(self, method: str, run_id: str, target_layer: int) -> Path:
        if isinstance(target_layer, bool) or not isinstance(target_layer, int) or target_layer < 1:
            raise ValueError("target_layer must be a positive integer")
        return self.run_root(method, run_id) / f"{target_layer}.jsonl"

    def read_originals(self) -> tuple[OriginalRecord, ...]:
        if not self.original_path.is_file():
            raise FileNotFoundError(f"Missing original dataset file: {self.original_path}")
        records = tuple(
            OriginalRecord.from_source_row(self.dataset_name, row)
            for row in read_jsonl(self.original_path)
        )
        identities = [record.base_text_id for record in records]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate source IDs in {self.original_path}")
        if not records:
            raise ValueError(f"Original dataset is empty: {self.original_path}")
        return records

    def write_originals(
        self,
        records: Iterable[OriginalRecord],
        *,
        overwrite: bool = False,
    ) -> Path:
        values = list(records)
        if not values:
            raise ValueError("Cannot write an empty original dataset")
        if any(record.dataset_name != self.dataset_name for record in values):
            raise ValueError("Original record dataset_name does not match repository")
        identities = [record.base_text_id for record in values]
        if len(identities) != len(set(identities)):
            raise ValueError("Original source IDs must be unique")
        return write_jsonl_atomic(
            self.original_path,
            [record.to_row() for record in values],
            overwrite=overwrite,
        )

    def read_manifest(self) -> PerturbationManifest:
        if not self.manifest_path.exists():
            return PerturbationManifest(dataset_name=self.dataset_name)
        value = read_json(self.manifest_path)
        if not isinstance(value, dict):
            raise ValueError(f"Manifest must be a JSON object: {self.manifest_path}")
        manifest = PerturbationManifest.from_dict(value)
        if manifest.dataset_name != self.dataset_name:
            raise ValueError(
                f"Manifest dataset_name {manifest.dataset_name!r} does not match "
                f"repository {self.dataset_name!r}"
            )
        return manifest

    def list_layers(
        self,
        *,
        methods: Iterable[str] | None = None,
        run_ids: Iterable[str] | None = None,
        target_layers: Iterable[int] | None = None,
    ) -> tuple[LayerManifestEntry, ...]:
        method_filter = set(methods) if methods is not None else None
        run_filter = set(run_ids) if run_ids is not None else None
        layer_filter = set(target_layers) if target_layers is not None else None
        entries = (
            entry
            for entry in self.read_manifest().layers
            if (method_filter is None or entry.method in method_filter)
            and (run_filter is None or entry.run_id in run_filter)
            and (layer_filter is None or entry.target_layer in layer_filter)
        )
        return tuple(sorted(entries, key=lambda item: item.identity))

    def get_layer(self, method: str, run_id: str, target_layer: int) -> LayerManifestEntry:
        identity = (method, run_id, target_layer)
        matches = [entry for entry in self.read_manifest().layers if entry.identity == identity]
        if not matches:
            raise FileNotFoundError(
                f"No manifest entry for method={method!r}, run_id={run_id!r}, "
                f"target_layer={target_layer}"
            )
        return matches[0]

    def _entry_path(self, entry: LayerManifestEntry) -> Path:
        relative = Path(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest layer path must be repository-relative: {entry.path}")
        path = self.dataset_dir / relative
        expected = self.layer_path(entry.method, entry.run_id, entry.target_layer)
        if path != expected:
            raise ValueError(
                f"Manifest path {entry.path!r} does not match its method/run/layer identity"
            )
        return path

    def read_candidates(
        self,
        entry: LayerManifestEntry,
        *,
        verify_hash: bool = True,
    ) -> tuple[CandidateRecord, ...]:
        path = self._entry_path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest references a missing layer: {path}")
        if verify_hash and sha256_file(path) != entry.content_hash:
            raise ValueError(f"Layer checksum does not match manifest: {path}")
        records = tuple(CandidateRecord.from_row(row) for row in read_jsonl(path))
        if len(records) != entry.output_count:
            raise ValueError(
                f"Layer row count does not match manifest for {path}: "
                f"{len(records)} != {entry.output_count}"
            )
        candidate_ids = [record.candidate_id for record in records]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Layer contains duplicate candidate IDs: {path}")
        for record in records:
            expected = (self.dataset_name, entry.method, entry.run_id, entry.target_layer)
            actual = (
                record.dataset_name,
                record.perturbation_method,
                record.run_id,
                record.target_layer,
            )
            if actual != expected:
                raise ValueError(f"Candidate provenance does not match manifest: {record.candidate_id}")
        return records

    def validate_lineage(self) -> None:
        """Validate all manifest layers as one parent-linked candidate graph."""
        originals = {
            make_original_candidate_id(
                dataset_name=self.dataset_name,
                base_text_id=record.base_text_id,
            ): record.base_text_id
            for record in self.read_originals()
        }
        entries = self.list_layers()
        records_by_layer: dict[tuple[str, str, int], tuple[CandidateRecord, ...]] = {}
        all_candidate_ids: set[str] = set(originals)
        for entry in entries:
            records = self.read_candidates(entry)
            records_by_layer[entry.identity] = records
            for record in records:
                if record.candidate_id in all_candidate_ids:
                    raise ValueError(
                        f"Duplicate candidate_id across repository: {record.candidate_id}"
                    )
                all_candidate_ids.add(record.candidate_id)

        for entry in entries:
            if entry.source_layer == 0:
                parents = originals
            else:
                source_identity = (
                    str(entry.source_method),
                    str(entry.source_run_id),
                    entry.source_layer,
                )
                if source_identity not in records_by_layer:
                    raise ValueError(
                        f"Layer {entry.identity} references missing source layer "
                        f"{source_identity}"
                    )
                parents = {
                    record.candidate_id: record.base_text_id
                    for record in records_by_layer[source_identity]
                }
            for record in records_by_layer[entry.identity]:
                parent_base_id = parents.get(record.parent_candidate_id)
                if parent_base_id is None:
                    raise ValueError(
                        f"Candidate {record.candidate_id} references unknown parent "
                        f"{record.parent_candidate_id}"
                    )
                if parent_base_id != record.base_text_id:
                    raise ValueError(
                        f"Candidate {record.candidate_id} and its parent have different "
                        "base_text_id values"
                    )

    @contextmanager
    def _manifest_lock(self) -> Iterator[None]:
        self.perturbation_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.perturbation_root / ".manifest.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def write_candidate_layer(
        self,
        records: Iterable[CandidateRecord],
        *,
        method: str,
        run_id: str,
        target_layer: int,
        source_layer: int,
        source_method: str | None,
        source_run_id: str | None,
        config: dict[str, Any],
        input_count: int,
        overwrite: bool = False,
    ) -> LayerManifestEntry:
        values = list(records)
        if not values:
            raise ValueError("Cannot write an empty candidate layer")
        if input_count < 0:
            raise ValueError("input_count must be non-negative")
        expected = (self.dataset_name, method, run_id, target_layer)
        for record in values:
            actual = (
                record.dataset_name,
                record.perturbation_method,
                record.run_id,
                record.target_layer,
            )
            if actual != expected:
                raise ValueError("Candidate provenance does not match requested layer")
            if record.source_layer != source_layer:
                raise ValueError("Candidate source_layer does not match requested layer")
            if record.source_method != source_method or record.source_run_id != source_run_id:
                raise ValueError("Candidate source method/run does not match requested layer")
        candidate_ids = [record.candidate_id for record in values]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate IDs must be unique within a layer")

        config_hash = canonical_json_hash(config)
        rows = [record.to_row() for record in values]
        for row in rows:
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)

        destination = self.layer_path(method, run_id, target_layer)
        relative_path = destination.relative_to(self.dataset_dir).as_posix()
        with self._manifest_lock():
            manifest = self.read_manifest()
            retained = [entry for entry in manifest.layers if entry.identity != (method, run_id, target_layer)]
            if len(retained) != len(manifest.layers) and not overwrite:
                raise FileExistsError(
                    f"Manifest entry already exists for method={method!r}, "
                    f"run_id={run_id!r}, target_layer={target_layer}"
                )
            write_jsonl_atomic(
                destination,
                rows,
                overwrite=overwrite,
            )
            entry = LayerManifestEntry(
                method=method,
                run_id=run_id,
                target_layer=target_layer,
                path=relative_path,
                source_layer=source_layer,
                source_method=source_method,
                source_run_id=source_run_id,
                config=dict(config),
                config_hash=config_hash,
                content_hash=sha256_file(destination),
                input_count=input_count,
                output_count=len(values),
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            updated = PerturbationManifest(
                dataset_name=self.dataset_name,
                layers=tuple(sorted([*retained, entry], key=lambda item: item.identity)),
            )
            write_json_atomic(self.manifest_path, updated.to_dict(), overwrite=True)
        return entry


__all__ = ["DatasetRepository"]
