# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical records and configuration contracts for dataset preparation.

These contracts deliberately do not contain filesystem behavior.  Readers,
writers, generation services, and HF builders consume the same records so a
candidate has one identity and one provenance representation throughout the
workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


ORIGINAL_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
SCORE_SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 1
WORKFLOW_SCHEMA_VERSION = 1

PERTURBATION_SOURCES = frozenset({"LLM", "trad"})
COMPOSITION_POLICIES = frozenset(
    {"all", "balanced", "weighted", "source_exclusive", "fixed_per_source"}
)
PAIR_POLICIES = frozenset(
    {"none", "parent_child", "original_only", "all_unequal_layers", "cross_source_unmatched"}
)


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: object, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field_name} must be a string or integer identifier")
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field_name)


def _integer(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _string_tuple(
    value: object,
    field_name: str,
    *,
    unique: bool = True,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array of strings")
    result = tuple(_nonempty(item, field_name) for item in value)
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {unknown}")


@dataclass(frozen=True)
class OriginalRecord:
    """One immutable source text in a named custom dataset."""

    dataset_name: str
    base_text_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", _nonempty(self.dataset_name, "dataset_name"))
        object.__setattr__(self, "base_text_id", _identifier(self.base_text_id, "base_text_id"))
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_source_row(
        cls,
        dataset_name: str,
        row: Mapping[str, Any],
    ) -> "OriginalRecord":
        """Normalize a source row; ``head_id`` is accepted only at this boundary."""
        if "custom_id" in row and "head_id" in row:
            custom_id = _identifier(row["custom_id"], "custom_id")
            head_id = _identifier(row["head_id"], "head_id")
            if custom_id != head_id:
                raise ValueError("custom_id and head_id identify different source texts")
            base_text_id = custom_id
        elif "custom_id" in row:
            base_text_id = _identifier(row["custom_id"], "custom_id")
        elif "head_id" in row:
            base_text_id = _identifier(row["head_id"], "head_id")
        else:
            raise ValueError("Source row requires custom_id (legacy head_id is accepted)")
        if "text" not in row:
            raise ValueError("Source row requires text")
        metadata = {
            key: item
            for key, item in row.items()
            if key not in {"schema_version", "custom_id", "head_id", "text"}
        }
        return cls(dataset_name, base_text_id, row["text"], metadata)

    def to_row(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "schema_version": ORIGINAL_SCHEMA_VERSION,
            "custom_id": self.base_text_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class CandidateRecord:
    """One generated candidate with complete method and parent provenance."""

    dataset_name: str
    base_text_id: str
    candidate_id: str
    text: str
    perturbation_method: str
    perturbation_source: str
    run_id: str
    candidate_index: int
    source_layer: int
    target_layer: int
    parent_candidate_id: str
    source_method: str | None = None
    source_run_id: str | None = None
    perturbation_edits: tuple[str, ...] = ()
    target_dimensions: tuple[str, ...] = ()
    severity: str | None = None
    generator: str | None = None
    seed: int | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    catalog_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", _nonempty(self.dataset_name, "dataset_name"))
        object.__setattr__(self, "base_text_id", _identifier(self.base_text_id, "base_text_id"))
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "perturbation_method", _nonempty(self.perturbation_method, "perturbation_method"))
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "parent_candidate_id", _nonempty(self.parent_candidate_id, "parent_candidate_id"))
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.perturbation_source not in PERTURBATION_SOURCES:
            raise ValueError(f"perturbation_source must be one of {sorted(PERTURBATION_SOURCES)}")
        _integer(self.candidate_index, "candidate_index", minimum=0)
        _integer(self.source_layer, "source_layer", minimum=0)
        _integer(self.target_layer, "target_layer", minimum=1)
        if self.target_layer <= self.source_layer:
            raise ValueError("target_layer must be greater than source_layer")
        if self.source_layer == 0 and (self.source_method is not None or self.source_run_id is not None):
            raise ValueError("source_method/source_run_id must be omitted for original inputs")
        if self.source_layer > 0 and (self.source_method is None or self.source_run_id is None):
            raise ValueError("source_method and source_run_id are required for perturbed inputs")
        object.__setattr__(self, "source_method", _optional_string(self.source_method, "source_method"))
        object.__setattr__(self, "source_run_id", _optional_string(self.source_run_id, "source_run_id"))
        object.__setattr__(
            self,
            "perturbation_edits",
            _string_tuple(self.perturbation_edits, "perturbation_edits", unique=False),
        )
        object.__setattr__(self, "target_dimensions", _string_tuple(self.target_dimensions, "target_dimensions"))
        for name in ("severity", "generator", "prompt_version", "prompt_hash", "catalog_hash"):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or null")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CandidateRecord":
        if row.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported candidate schema_version: {row.get('schema_version')!r}"
            )
        required = {
            "dataset_name", "base_text_id", "candidate_id", "candidate_index",
            "text", "perturbation_method", "perturbation_source", "run_id",
            "source_layer", "target_layer", "parent_candidate_id",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Candidate row is missing fields: {missing}")
        known = required | {
            "schema_version", "source_method", "source_run_id", "perturbation_edits",
            "target_dimensions", "severity", "generator", "seed", "prompt_version",
            "prompt_hash", "catalog_hash",
        }
        return cls(
            dataset_name=row["dataset_name"],
            base_text_id=row["base_text_id"],
            candidate_id=row["candidate_id"],
            candidate_index=row["candidate_index"],
            text=row["text"],
            perturbation_method=row["perturbation_method"],
            perturbation_source=row["perturbation_source"],
            run_id=row["run_id"],
            source_layer=row["source_layer"],
            source_method=row.get("source_method"),
            source_run_id=row.get("source_run_id"),
            target_layer=row["target_layer"],
            parent_candidate_id=row["parent_candidate_id"],
            perturbation_edits=_string_tuple(
                row.get("perturbation_edits"), "perturbation_edits", unique=False
            ),
            target_dimensions=_string_tuple(row.get("target_dimensions"), "target_dimensions"),
            severity=row.get("severity"),
            generator=row.get("generator"),
            seed=row.get("seed"),
            prompt_version=row.get("prompt_version"),
            prompt_hash=row.get("prompt_hash"),
            catalog_hash=row.get("catalog_hash"),
            metadata={key: item for key, item in row.items() if key not in known},
        )

    def to_row(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "base_text_id": self.base_text_id,
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "text": self.text,
            "source_layer": self.source_layer,
            "source_method": self.source_method,
            "source_run_id": self.source_run_id,
            "target_layer": self.target_layer,
            "parent_candidate_id": self.parent_candidate_id,
            "perturbation_method": self.perturbation_method,
            "perturbation_source": self.perturbation_source,
            "run_id": self.run_id,
            "perturbation_edits": list(self.perturbation_edits),
            "target_dimensions": list(self.target_dimensions),
            "severity": self.severity,
            "generator": self.generator,
            "seed": self.seed,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "catalog_hash": self.catalog_hash,
        }


@dataclass(frozen=True)
class ScoreRecord:
    """One scalar score attached to exactly one canonical candidate."""

    dataset_name: str
    base_text_id: str
    candidate_id: str
    perturbation_method: str
    scoring_method: str
    scoring_run_id: str
    score_value: float
    source_layer: int
    target_layer: int
    reference_candidate_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", _nonempty(self.dataset_name, "dataset_name"))
        object.__setattr__(self, "base_text_id", _identifier(self.base_text_id, "base_text_id"))
        for name in ("candidate_id", "perturbation_method", "scoring_method", "scoring_run_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if isinstance(self.score_value, bool) or not isinstance(self.score_value, (int, float)):
            raise ValueError("score_value must be numeric")
        if not math.isfinite(float(self.score_value)):
            raise ValueError("score_value must be finite")
        object.__setattr__(self, "score_value", float(self.score_value))
        _integer(self.source_layer, "source_layer", minimum=0)
        _integer(self.target_layer, "target_layer", minimum=0)
        object.__setattr__(self, "reference_candidate_id", _optional_string(self.reference_candidate_id, "reference_candidate_id"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ScoreRecord":
        if row.get("schema_version") != SCORE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported score schema_version: {row.get('schema_version')!r}"
            )
        required = {
            "dataset_name", "base_text_id", "candidate_id", "perturbation_method",
            "scoring_method", "scoring_run_id", "score_value", "source_layer",
            "target_layer",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Score row is missing fields: {missing}")
        known = required | {"schema_version", "reference_candidate_id"}
        return cls(
            dataset_name=row["dataset_name"],
            base_text_id=row["base_text_id"],
            candidate_id=row["candidate_id"],
            perturbation_method=row["perturbation_method"],
            scoring_method=row["scoring_method"],
            scoring_run_id=row["scoring_run_id"],
            score_value=row["score_value"],
            source_layer=row["source_layer"],
            target_layer=row["target_layer"],
            reference_candidate_id=row.get("reference_candidate_id"),
            metadata={key: item for key, item in row.items() if key not in known},
        )

    def to_row(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "schema_version": SCORE_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "base_text_id": self.base_text_id,
            "candidate_id": self.candidate_id,
            "perturbation_method": self.perturbation_method,
            "scoring_method": self.scoring_method,
            "scoring_run_id": self.scoring_run_id,
            "score_value": self.score_value,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "reference_candidate_id": self.reference_candidate_id,
        }


@dataclass(frozen=True)
class LayerManifestEntry:
    """Authoritative description of one completed method/run/layer file."""

    method: str
    run_id: str
    target_layer: int
    path: str
    source_layer: int
    source_method: str | None
    source_run_id: str | None
    config: dict[str, Any]
    config_hash: str
    content_hash: str
    input_count: int
    output_count: int
    created_at_utc: str

    def __post_init__(self) -> None:
        for name in ("method", "run_id", "path", "config_hash", "content_hash", "created_at_utc"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        _integer(self.source_layer, "source_layer", minimum=0)
        _integer(self.target_layer, "target_layer", minimum=1)
        if self.target_layer <= self.source_layer:
            raise ValueError("manifest layers must satisfy 0 <= source_layer < target_layer")
        if self.source_layer == 0 and (self.source_method is not None or self.source_run_id is not None):
            raise ValueError("Original manifest inputs cannot have source method/run")
        if self.source_layer > 0 and (self.source_method is None or self.source_run_id is None):
            raise ValueError("Perturbed manifest inputs require source method/run")
        _integer(self.input_count, "input_count", minimum=0)
        _integer(self.output_count, "output_count", minimum=0)
        object.__setattr__(self, "config", dict(self.config))

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.method, self.run_id, self.target_layer

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "run_id": self.run_id,
            "target_layer": self.target_layer,
            "path": self.path,
            "source_layer": self.source_layer,
            "source_method": self.source_method,
            "source_run_id": self.source_run_id,
            "config": dict(self.config),
            "config_hash": self.config_hash,
            "content_hash": self.content_hash,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerManifestEntry":
        allowed = {
            "method", "run_id", "target_layer", "path", "source_layer",
            "source_method", "source_run_id", "config", "config_hash",
            "content_hash", "input_count", "output_count", "created_at_utc",
        }
        _reject_unknown(value, allowed, "manifest layer")
        missing = sorted(allowed - {"source_method", "source_run_id"} - set(value))
        if missing:
            raise ValueError(f"Manifest layer is missing fields: {missing}")
        return cls(
            method=value["method"],
            run_id=value["run_id"],
            target_layer=value["target_layer"],
            path=value["path"],
            source_layer=value["source_layer"],
            source_method=value.get("source_method"),
            source_run_id=value.get("source_run_id"),
            config=_mapping(value["config"], "manifest layer config"),
            config_hash=value["config_hash"],
            content_hash=value["content_hash"],
            input_count=value["input_count"],
            output_count=value["output_count"],
            created_at_utc=value["created_at_utc"],
        )


@dataclass(frozen=True)
class PerturbationManifest:
    """Versioned index of canonical candidate layers for one dataset."""

    dataset_name: str
    layers: tuple[LayerManifestEntry, ...] = ()
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", _nonempty(self.dataset_name, "dataset_name"))
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema_version: {self.schema_version}")
        identities = [entry.identity for entry in self.layers]
        if len(identities) != len(set(identities)):
            raise ValueError("Manifest contains duplicate method/run/layer entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "layers": [entry.to_dict() for entry in self.layers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerturbationManifest":
        allowed = {"schema_version", "dataset_name", "layers"}
        _reject_unknown(value, allowed, "manifest")
        missing = sorted(allowed - set(value))
        if missing:
            raise ValueError(f"Manifest is missing fields: {missing}")
        layers = value["layers"]
        if not isinstance(layers, list) or any(not isinstance(item, Mapping) for item in layers):
            raise ValueError("manifest.layers must be an array of objects")
        return cls(
            schema_version=value["schema_version"],
            dataset_name=value["dataset_name"],
            layers=tuple(LayerManifestEntry.from_dict(item) for item in layers),
        )


@dataclass(frozen=True)
class GenerationSpec:
    """One generation stage within a workflow."""

    method: str
    run_id: str = "default"
    source_layer: int = 0
    source_method: str | None = None
    source_run_id: str | None = None
    target_layer: int | None = None
    limit: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationSpec":
        allowed = {
            "method", "run_id", "source_layer", "source_method", "source_run_id",
            "target_layer", "limit", "config",
        }
        _reject_unknown(value, allowed, "generation")
        if "method" not in value:
            raise ValueError("Generation requires method")
        result = cls(
            method=value["method"],
            run_id=value.get("run_id", "default"),
            source_layer=value.get("source_layer", 0),
            source_method=value.get("source_method"),
            source_run_id=value.get("source_run_id"),
            target_layer=value.get("target_layer"),
            limit=value.get("limit"),
            config=_mapping(value.get("config"), "generation.config"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        _nonempty(self.method, "generation.method")
        _nonempty(self.run_id, "generation.run_id")
        _integer(self.source_layer, "generation.source_layer", minimum=0)
        if self.target_layer is not None:
            _integer(self.target_layer, "generation.target_layer", minimum=1)
        target_layer = self.target_layer if self.target_layer is not None else self.source_layer + 1
        if target_layer <= self.source_layer:
            raise ValueError("generation.target_layer must be greater than source_layer")
        if self.source_layer == 0 and (self.source_method is not None or self.source_run_id is not None):
            raise ValueError("Original generation inputs cannot specify source method/run")
        if self.source_layer > 0 and (self.source_method is None or self.source_run_id is None):
            raise ValueError("Perturbed generation inputs require source_method and source_run_id")
        if self.limit is not None:
            _integer(self.limit, "generation.limit", minimum=1)


@dataclass(frozen=True)
class HFBuildSpec:
    """Selection, composition, and split policy for one HF dataset build."""

    output_name: str
    datasets: tuple[str, ...] = ()
    include_methods: tuple[str, ...] = ()
    include_runs: tuple[str, ...] = ()
    include_layers: tuple[int, ...] = ()
    composition: str = "all"
    method_weights: dict[str, float] = field(default_factory=dict)
    samples_per_source: int = 1
    pair_policy: str = "none"
    reuse_limit: int = 5
    downsample_size: int | None = None
    heldout_ratio: float = 0.3
    test_ratio_within_heldout: float = 0.5
    score_names: tuple[str, ...] = ()
    score_run_ids: tuple[str, ...] = ()
    seed: int = 42

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HFBuildSpec":
        allowed = {
            "output_name", "datasets", "include_methods", "include_runs",
            "include_layers", "composition", "method_weights", "samples_per_source",
            "pair_policy", "reuse_limit", "downsample_size", "heldout_ratio",
            "test_ratio_within_heldout", "score_names", "seed",
            "score_run_ids",
        }
        _reject_unknown(value, allowed, "hf")
        if "output_name" not in value:
            raise ValueError("HF configuration requires output_name")
        weights = _mapping(value.get("method_weights"), "hf.method_weights")
        raw_layers = value.get("include_layers", ())
        if not isinstance(raw_layers, (list, tuple)):
            raise ValueError("hf.include_layers must be an array of positive integers")
        result = cls(
            output_name=value["output_name"],
            datasets=_string_tuple(value.get("datasets"), "hf.datasets"),
            include_methods=_string_tuple(value.get("include_methods"), "hf.include_methods"),
            include_runs=_string_tuple(value.get("include_runs"), "hf.include_runs"),
            include_layers=tuple(raw_layers),
            composition=value.get("composition", "all"),
            method_weights={str(key): float(weight) for key, weight in weights.items()},
            samples_per_source=value.get("samples_per_source", 1),
            pair_policy=value.get("pair_policy", "none"),
            reuse_limit=value.get("reuse_limit", 5),
            downsample_size=value.get("downsample_size"),
            heldout_ratio=value.get("heldout_ratio", 0.3),
            test_ratio_within_heldout=value.get("test_ratio_within_heldout", 0.5),
            score_names=_string_tuple(value.get("score_names"), "hf.score_names"),
            score_run_ids=_string_tuple(value.get("score_run_ids"), "hf.score_run_ids"),
            seed=value.get("seed", 42),
        )
        result.validate()
        return result

    def validate(self) -> None:
        _nonempty(self.output_name, "hf.output_name")
        if self.composition not in COMPOSITION_POLICIES:
            raise ValueError(f"Unknown composition policy: {self.composition!r}")
        if self.pair_policy not in PAIR_POLICIES:
            raise ValueError(f"Unknown pair policy: {self.pair_policy!r}")
        for layer in self.include_layers:
            _integer(layer, "hf.include_layers", minimum=1)
        if len(self.include_layers) != len(set(self.include_layers)):
            raise ValueError("hf.include_layers must not contain duplicates")
        _integer(self.samples_per_source, "hf.samples_per_source", minimum=1)
        _integer(self.reuse_limit, "hf.reuse_limit", minimum=1)
        if self.downsample_size is not None:
            _integer(self.downsample_size, "hf.downsample_size", minimum=3)
        if (
            isinstance(self.heldout_ratio, bool)
            or not isinstance(self.heldout_ratio, (int, float))
            or isinstance(self.test_ratio_within_heldout, bool)
            or not isinstance(self.test_ratio_within_heldout, (int, float))
            or not 0 < self.heldout_ratio < 1
            or not 0 < self.test_ratio_within_heldout < 1
        ):
            raise ValueError("HF split ratios must be between 0 and 1")
        if any(weight < 0 or not math.isfinite(weight) for weight in self.method_weights.values()):
            raise ValueError("HF method weights must be finite and non-negative")
        if self.composition == "weighted" and not any(self.method_weights.values()):
            raise ValueError("Weighted composition requires at least one positive method weight")
        _integer(self.seed, "hf.seed", minimum=0)


@dataclass(frozen=True)
class WorkflowConfig:
    """Complete versioned configuration for generation and optional HF export."""

    dataset: str
    generations: tuple[GenerationSpec, ...] = ()
    hf: HFBuildSpec | None = None
    dataset_root: str = "data/custom_datasets"
    seed: int = 42
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowConfig":
        allowed = {"schema_version", "dataset", "dataset_root", "seed", "generations", "hf"}
        _reject_unknown(value, allowed, "workflow")
        if "dataset" not in value:
            raise ValueError("Workflow requires dataset")
        generation_values = value.get("generations", [])
        if not isinstance(generation_values, list):
            raise ValueError("workflow.generations must be an array")
        hf_value = value.get("hf")
        if hf_value is not None and not isinstance(hf_value, Mapping):
            raise ValueError("workflow.hf must be an object")
        if any(not isinstance(item, Mapping) for item in generation_values):
            raise ValueError("Every workflow generation must be an object")
        workflow_seed = value.get("seed", 42)
        resolved_hf = None
        if hf_value is not None:
            resolved_hf = dict(hf_value)
            resolved_hf.setdefault("datasets", [value["dataset"]])
            resolved_hf.setdefault("seed", workflow_seed)
        result = cls(
            dataset=value["dataset"],
            generations=tuple(GenerationSpec.from_dict(item) for item in generation_values),
            hf=HFBuildSpec.from_dict(resolved_hf) if resolved_hf is not None else None,
            dataset_root=value.get("dataset_root", "data/custom_datasets"),
            seed=workflow_seed,
            schema_version=value.get("schema_version", WORKFLOW_SCHEMA_VERSION),
        )
        result.validate()
        return result

    def validate(self) -> None:
        _nonempty(self.dataset, "workflow.dataset")
        _nonempty(self.dataset_root, "workflow.dataset_root")
        if self.schema_version != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"Unsupported workflow schema_version: {self.schema_version}")
        _integer(self.seed, "workflow.seed", minimum=0)


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "COMPOSITION_POLICIES",
    "CandidateRecord",
    "GenerationSpec",
    "HFBuildSpec",
    "LayerManifestEntry",
    "MANIFEST_SCHEMA_VERSION",
    "ORIGINAL_SCHEMA_VERSION",
    "OriginalRecord",
    "PAIR_POLICIES",
    "PERTURBATION_SOURCES",
    "PerturbationManifest",
    "SCORE_SCHEMA_VERSION",
    "ScoreRecord",
    "WORKFLOW_SCHEMA_VERSION",
    "WorkflowConfig",
]
