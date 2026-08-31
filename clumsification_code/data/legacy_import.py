# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""One-way import of historical perturbation directories into the repository."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clumsification_code.perturbations.registry import get_method_spec

from .candidate_identity import make_candidate_id, make_original_candidate_id
from .io import read_json, read_jsonl
from .repository import DatasetRepository
from .schemas import CandidateRecord, LayerManifestEntry, MANIFEST_SCHEMA_VERSION


def _relative_source(dataset_dir: Path, value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("Legacy source directory must be a non-empty repository-relative path")
    source = dataset_dir / relative
    if not source.is_dir():
        raise FileNotFoundError(f"Legacy perturbation directory does not exist: {source}")
    return source


def _base_text_id(row: dict[str, Any], *, path: Path, row_number: int) -> str:
    value = row.get("base_text_id", row.get("head_id", row.get("custom_id")))
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError(f"{path}:{row_number} is missing a valid source identifier")
    return str(value)


def _edit_names(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("perturbation_edits must be an array when present")
    names = []
    for edit in value:
        if isinstance(edit, str) and edit:
            names.append(edit)
        elif isinstance(edit, dict):
            name = edit.get("transform_type", edit.get("name", edit.get("edit_id")))
            if not isinstance(name, str) or not name:
                name = json.dumps(edit, ensure_ascii=False, sort_keys=True)
            names.append(name)
        else:
            raise ValueError(f"Unsupported perturbation edit value: {edit!r}")
    return tuple(names)


def _dimensions(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("target_dimensions must be an array of non-empty strings")
    return tuple(dict.fromkeys(value))


def _seed(value: object, *, path: Path, row_number: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}:{row_number} seed must be an integer or null")
    return value


class LegacyLayoutImporter:
    """Stateful importer that retains old-to-canonical candidate ID mappings."""

    def __init__(self, repository: DatasetRepository):
        self.repository = repository
        self.originals = {
            record.base_text_id: make_original_candidate_id(
                dataset_name=repository.dataset_name,
                base_text_id=record.base_text_id,
            )
            for record in repository.read_originals()
        }
        self.candidate_id_map: dict[str, str] = {}

    def prepare_manifest(self) -> Path | None:
        """Move a pre-canonical manifest aside before the first canonical write."""
        path = self.repository.manifest_path
        if not path.exists():
            return None
        value = read_json(path)
        if isinstance(value, dict) and value.get("schema_version") == MANIFEST_SCHEMA_VERSION:
            self.repository.read_manifest()
            return None
        backup = path.with_name("perturbation_manifest.precanonical.json")
        if backup.exists():
            raise FileExistsError(
                f"Cannot preserve legacy manifest because backup exists: {backup}"
            )
        path.replace(backup)
        return backup

    def _source_candidates(
        self,
        *,
        source_layer: int,
        source_method: str | None,
        source_run_id: str | None,
    ) -> dict[str, list[tuple[str, frozenset[str]]]]:
        if source_layer == 0:
            return {
                base_text_id: [(candidate_id, frozenset({candidate_id}))]
                for base_text_id, candidate_id in self.originals.items()
            }
        if source_method is None or source_run_id is None:
            raise ValueError("Perturbed legacy inputs require source method and run")
        entry = self.repository.get_layer(source_method, source_run_id, source_layer)
        by_base: dict[str, list[tuple[str, frozenset[str]]]] = {}
        for candidate in self.repository.read_candidates(entry):
            aliases = {candidate.candidate_id}
            legacy_id = candidate.metadata.get("legacy_candidate_id")
            if legacy_id is not None:
                aliases.add(str(legacy_id))
            by_base.setdefault(candidate.base_text_id, []).append(
                (candidate.candidate_id, frozenset(aliases))
            )
        return by_base

    def _resolve_parent(
        self,
        row: dict[str, Any],
        *,
        base_text_id: str,
        available: dict[str, list[tuple[str, frozenset[str]]]],
        path: Path,
        row_number: int,
        candidate_id_map: dict[str, str],
    ) -> str:
        parents = available.get(base_text_id, [])
        explicit = row.get("parent_candidate_id")
        if explicit is not None:
            requested = str(explicit)
            mapped = candidate_id_map.get(requested, requested)
            matches = [
                candidate_id
                for candidate_id, aliases in parents
                if mapped == candidate_id or requested in aliases
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{path}:{row_number} references a parent outside its declared source layer"
                )
            return matches[0]
        if len(parents) == 1:
            return parents[0][0]
        if not parents:
            raise ValueError(f"{path}:{row_number} has no candidate in its source layer")
        raise ValueError(
            f"{path}:{row_number} has {len(parents)} possible parents; "
            "parent_candidate_id is required"
        )

    def import_directory(
        self,
        source_directory: str | Path,
        *,
        method: str,
        run_id: str = "legacy-import",
        overwrite: bool = False,
    ) -> tuple[LayerManifestEntry, ...]:
        """Import every numeric JSONL layer directly inside one legacy directory."""
        spec = get_method_spec(method)
        source = _relative_source(self.repository.dataset_dir, source_directory)
        destination_root = self.repository.run_root(method, run_id)
        if source == destination_root:
            raise ValueError("Legacy source and canonical destination must differ")
        paths = []
        for path in source.glob("*.jsonl"):
            try:
                layer = int(path.stem)
            except ValueError:
                continue
            if layer < 1:
                raise ValueError(f"Legacy perturbation layer must be positive: {path}")
            paths.append((layer, path))
        if not paths:
            raise FileNotFoundError(f"No numeric JSONL layers found in {source}")

        candidate_id_map = dict(self.candidate_id_map)
        planned_layers: dict[tuple[str, str, int], tuple[CandidateRecord, ...]] = {}
        prepared: list[
            tuple[int, Path, int, str | None, str | None, list[CandidateRecord], int]
        ] = []
        for target_layer, path in sorted(paths):
            rows = read_jsonl(path)
            if not rows:
                raise ValueError(f"Legacy perturbation layer is empty: {path}")
            declared_source_layers = {
                int(row.get("source_layer", target_layer - 1)) for row in rows
            }
            if len(declared_source_layers) != 1:
                raise ValueError(f"Legacy layer mixes multiple source_layer values: {path}")
            source_layer = declared_source_layers.pop()
            if source_layer < 0 or source_layer >= target_layer:
                raise ValueError(f"Invalid source/target layer relation in {path}")
            declared_source_methods = {
                row.get("source_method")
                for row in rows
                if row.get("source_method") is not None
            }
            declared_source_runs = {
                row.get("source_run_id", row.get("source_run"))
                for row in rows
                if row.get("source_run_id", row.get("source_run")) is not None
            }
            if len(declared_source_methods) > 1 or len(declared_source_runs) > 1:
                raise ValueError(f"Legacy layer mixes multiple source methods or runs: {path}")
            source_method = (
                None
                if source_layer == 0
                else str(next(iter(declared_source_methods), method))
            )
            source_run_id = (
                None
                if source_layer == 0
                else str(next(iter(declared_source_runs), run_id))
            )
            source_identity = (str(source_method), str(source_run_id), source_layer)
            if source_layer > 0 and source_identity in planned_layers:
                available: dict[str, list[tuple[str, frozenset[str]]]] = {}
                for candidate in planned_layers[source_identity]:
                    aliases = {candidate.candidate_id}
                    legacy_id = candidate.metadata.get("legacy_candidate_id")
                    if legacy_id is not None:
                        aliases.add(str(legacy_id))
                    available.setdefault(candidate.base_text_id, []).append(
                        (candidate.candidate_id, frozenset(aliases))
                    )
            else:
                available = self._source_candidates(
                    source_layer=source_layer,
                    source_method=source_method,
                    source_run_id=source_run_id,
                )
            candidate_counts: dict[str, int] = {}
            candidates = []
            pending_id_map: dict[str, str] = {}
            for row_number, row in enumerate(rows, start=1):
                base_text_id = _base_text_id(row, path=path, row_number=row_number)
                if base_text_id not in self.originals:
                    raise ValueError(
                        f"{path}:{row_number} source ID {base_text_id!r} is absent from original.jsonl"
                    )
                if row.get("dataset_name") not in {None, self.repository.dataset_name}:
                    raise ValueError(f"{path}:{row_number} has a different dataset_name")
                if row.get("perturbation_method") not in {None, method}:
                    raise ValueError(f"{path}:{row_number} has a different perturbation_method")
                if row.get("perturbation_source") not in {None, spec.perturbation_source}:
                    raise ValueError(f"{path}:{row_number} has a different perturbation_source")
                if row.get("target_layer") not in {None, target_layer}:
                    raise ValueError(f"{path}:{row_number} target_layer disagrees with filename")
                if not isinstance(row.get("text"), str):
                    raise ValueError(f"{path}:{row_number} is missing string text")
                parent_candidate_id = self._resolve_parent(
                    row,
                    base_text_id=base_text_id,
                    available=available,
                    path=path,
                    row_number=row_number,
                    candidate_id_map=candidate_id_map,
                )
                candidate_index = candidate_counts.get(parent_candidate_id, 0)
                candidate_counts[parent_candidate_id] = candidate_index + 1
                candidate_id = make_candidate_id(
                    dataset_name=self.repository.dataset_name,
                    perturbation_method=method,
                    run_id=run_id,
                    base_text_id=base_text_id,
                    target_layer=target_layer,
                    parent_candidate_id=parent_candidate_id,
                    candidate_index=candidate_index,
                )
                reserved = {
                    "schema_version", "dataset_name", "base_text_id", "custom_id",
                    "head_id", "candidate_id", "candidate_index", "text", "source_layer",
                    "source_method", "source_run_id", "source_run", "target_layer",
                    "parent_candidate_id", "perturbation_method", "perturbation_source",
                    "run_id", "perturbation_edits", "target_dimensions", "severity",
                    "generator", "model", "seed", "prompt_version", "prompt_hash",
                    "catalog_hash",
                }
                candidates.append(
                    CandidateRecord(
                        dataset_name=self.repository.dataset_name,
                        base_text_id=base_text_id,
                        candidate_id=candidate_id,
                        candidate_index=candidate_index,
                        text=row["text"],
                        perturbation_method=method,
                        perturbation_source=spec.perturbation_source,
                        run_id=run_id,
                        source_layer=source_layer,
                        source_method=source_method,
                        source_run_id=source_run_id,
                        target_layer=target_layer,
                        parent_candidate_id=parent_candidate_id,
                        perturbation_edits=_edit_names(row.get("perturbation_edits")),
                        target_dimensions=_dimensions(row.get("target_dimensions")),
                        severity=row.get("severity"),
                        generator=row.get("generator", row.get("model")),
                        seed=_seed(row.get("seed"), path=path, row_number=row_number),
                        prompt_version=row.get("prompt_version"),
                        prompt_hash=row.get("prompt_hash"),
                        catalog_hash=row.get("catalog_hash"),
                        metadata={
                            "legacy_source_path": path.relative_to(self.repository.dataset_dir).as_posix(),
                            **(
                                {"legacy_candidate_id": str(row["candidate_id"])}
                                if row.get("candidate_id") is not None
                                else {}
                            ),
                            "legacy_metadata": {
                                key: item for key, item in row.items() if key not in reserved
                            },
                        },
                    )
                )
                old_id = row.get("candidate_id")
                normalized_old_id = str(old_id) if old_id is not None else None
                if normalized_old_id is not None:
                    existing = candidate_id_map.get(
                        normalized_old_id,
                        pending_id_map.get(normalized_old_id),
                    )
                    if existing is not None and existing != candidate_id:
                        raise ValueError(
                            f"{path}:{row_number} duplicates legacy candidate_id "
                            f"{normalized_old_id!r}"
                        )
                    pending_id_map[normalized_old_id] = candidate_id
            candidate_id_map.update(pending_id_map)
            planned_layers[(method, run_id, target_layer)] = tuple(candidates)
            prepared.append(
                (
                    target_layer,
                    path,
                    source_layer,
                    source_method,
                    source_run_id,
                    candidates,
                    sum(len(values) for values in available.values()),
                )
            )

        if not overwrite:
            manifest_identities: set[tuple[str, str, int]] = set()
            if self.repository.manifest_path.exists():
                manifest_value = read_json(self.repository.manifest_path)
                if (
                    isinstance(manifest_value, dict)
                    and manifest_value.get("schema_version") == MANIFEST_SCHEMA_VERSION
                ):
                    manifest_identities = {
                        entry.identity for entry in self.repository.read_manifest().layers
                    }
            for target_layer, *_ in prepared:
                identity = (method, run_id, target_layer)
                destination = self.repository.layer_path(method, run_id, target_layer)
                if identity in manifest_identities or destination.exists():
                    raise FileExistsError(
                        "Canonical destination already exists for "
                        f"method={method!r}, run_id={run_id!r}, "
                        f"target_layer={target_layer}"
                    )

        self.prepare_manifest()
        imported: list[LayerManifestEntry] = []
        for (
            target_layer,
            path,
            source_layer,
            source_method,
            source_run_id,
            candidates,
            input_count,
        ) in prepared:
            entry = self.repository.write_candidate_layer(
                candidates,
                method=method,
                run_id=run_id,
                target_layer=target_layer,
                source_layer=source_layer,
                source_method=source_method,
                source_run_id=source_run_id,
                config={
                    "imported_from": path.relative_to(self.repository.dataset_dir).as_posix(),
                    "legacy_layout": True,
                },
                input_count=input_count,
                overwrite=overwrite,
            )
            imported.append(entry)

        self.repository.validate_lineage()
        self.candidate_id_map = candidate_id_map
        return tuple(imported)


__all__ = ["LegacyLayoutImporter"]
