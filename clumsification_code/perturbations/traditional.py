# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical multilingual traditional fluency perturbations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

from .schemas import (
    GenerationRuntime,
    PerturbationInput,
    PerturbationResult,
    SkippedPerturbation,
)
from .rule_based_multilingual import (
    MultilingualRulePerturber,
    build_rule_templates,
    normalize_language,
)
from .morphology import load_morphology_backend, morphology_backend_name
from .sampling import sample_edit_count
from .unieval_fluency import UniEvalOperationUnavailable, apply_unieval_operation


_TRADITIONAL_WORKER: "TraditionalMethodAdapter | None" = None


class TraditionalNoChangeError(RuntimeError):
    """A valid traditional operation was inapplicable to one input."""

    def __init__(self, message: str, *, audit: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.audit = list(audit or ())


@dataclass(frozen=True)
class TraditionalEditOutcome:
    """One operation attempt and its audit information."""

    text: str
    operation: str
    details: dict[str, Any]
    unavailable_reason: str | None = None


def _init_traditional_worker(method_name: str, config: dict[str, Any]) -> None:
    global _TRADITIONAL_WORKER
    adapter_type = {
        "trad_single": TraditionalSingleMethod,
        "trad_sampled": TraditionalSampledMethod,
    }[method_name]
    _TRADITIONAL_WORKER = adapter_type(config)


def _generate_traditional_item(
    index: int, item: PerturbationInput
) -> tuple[int, PerturbationResult]:
    if _TRADITIONAL_WORKER is None:
        raise RuntimeError("Traditional worker was not initialized")
    return index, _TRADITIONAL_WORKER._generate_item(item, index=index)

@dataclass(frozen=True)
class TraditionalOperation:
    """Metadata and implementation binding for one traditional operation."""

    name: str
    dimensions: tuple[str, ...]
    backend: str
    description: str


TRADITIONAL_OPERATIONS: tuple[TraditionalOperation, ...] = (
    TraditionalOperation("repetition", ("Coherence", "Clarity"), "unieval", "Repeat a UniEval token span."),
    TraditionalOperation("deletion", ("Coherence", "Clarity"), "unieval", "Delete a UniEval token span."),
    TraditionalOperation("shuffle", ("Grammaticality", "Clarity"), "unieval", "Shuffle a UniEval token span."),
    TraditionalOperation("agreement_corruption", ("Grammaticality",), "morphology", "Corrupt finite-verb agreement features."),
    TraditionalOperation("random_inflection", ("Grammaticality",), "morphology", "Replace an inflectable word with a distinct same-lemma form."),
)

_OPERATION_BY_NAME = {operation.name: operation for operation in TRADITIONAL_OPERATIONS}


def list_traditional_operations() -> tuple[TraditionalOperation, ...]:
    """Return the stable traditional operation inventory."""
    return TRADITIONAL_OPERATIONS


def get_traditional_operation(name: str) -> TraditionalOperation:
    try:
        return _OPERATION_BY_NAME[name]
    except KeyError as exc:
        valid = ", ".join(_OPERATION_BY_NAME)
        raise ValueError(f"Unknown traditional operation {name!r}; choose one of: {valid}") from exc


def traditional_operations_for_language(language: str) -> tuple[TraditionalOperation, ...]:
    """Return operations supported for a language.

    Morphology-backed operations use Lemminflect for English and UniMorph for
    all other supported languages. UniEval operations apply to every language.
    """
    normalize_language(language)
    return TRADITIONAL_OPERATIONS


class TraditionalEditor:
    """Apply one named operation using its language-appropriate backend."""

    def __init__(self, language: str = "eng", store: Any | None = None, *, use_morphology: bool = True):
        self.language = normalize_language(language)
        self.morphology_backend = morphology_backend_name(self.language)
        self.store = store
        self.use_morphology = use_morphology
        self._morph_perturber: Any | None = None
        self._morph_templates: dict[str, Callable[[dict], str]] | None = None

    def _load_morphology(self) -> dict[str, Callable[[dict], str]]:
        if self._morph_templates is None:
            if not self.use_morphology:
                raise RuntimeError("Morphology backend is disabled")
            backend = load_morphology_backend(self.language, store=self.store)
            self.store = backend.store
            self._morph_perturber = MultilingualRulePerturber(self.language, self.store)
            self._morph_templates = {
                template.name: template.fn
                for template in build_rule_templates(self._morph_perturber)
            }
        return self._morph_templates

    def apply(
        self,
        text: str,
        operation: str,
        *,
        seed: int = 0,
        item_metadata: dict[str, Any] | None = None,
    ) -> TraditionalEditOutcome:
        """Apply one operation and retain its realization evidence."""
        spec = get_traditional_operation(operation)
        # The generated text is authoritative; caller metadata must not be able
        # to silently redirect an edit to a stale parent text.
        item = {**(item_metadata or {}), "text": text}
        if spec.backend == "unieval":
            try:
                output, details = apply_unieval_operation(
                    text,
                    operation=operation,
                    python_rng=random.Random(seed),
                    numpy_rng=np.random.default_rng(seed),
                )
            except UniEvalOperationUnavailable:
                return TraditionalEditOutcome(
                    text=text,
                    operation=operation,
                    details={},
                    unavailable_reason="no_substantive_unieval_edit",
                )
            return TraditionalEditOutcome(
                text=output, operation=operation, details=details
            )

        if spec.backend == "morphology":
            templates = self._load_morphology()
            fn = templates.get(operation)
            if fn is None:
                raise RuntimeError(f"Traditional backend does not implement {operation!r}")
            perturber = self._morph_perturber
            if perturber is not None:
                perturber.random_seed = seed
            output = str(fn(item))
            details = dict(getattr(perturber, "last_edit_metadata", {}) or {})
            if output.split() == text.split():
                return TraditionalEditOutcome(
                    text=text,
                    operation=operation,
                    details=details,
                    unavailable_reason="no_verified_morphology_edit",
                )
            if not _has_verified_morphology_change(details):
                return TraditionalEditOutcome(
                    text=text,
                    operation=operation,
                    details=details,
                    unavailable_reason="missing_morphology_verification",
                )
            return TraditionalEditOutcome(
                text=output, operation=operation, details=details
            )

        raise RuntimeError(f"Unsupported traditional backend {spec.backend!r}")


def _has_verified_morphology_change(details: dict[str, Any]) -> bool:
    """Require same-lemma, feature-changing evidence for a morphology edit."""
    source = str(details.get("source_form", ""))
    replacement = str(details.get("replacement_form", ""))
    if not source or not replacement or source == replacement:
        return False
    source_analyses = details.get("source_analyses")
    replacement_analyses = details.get("replacement_analyses")
    if not isinstance(source_analyses, list) or not isinstance(replacement_analyses, list):
        return False
    return any(
        isinstance(before, dict)
        and isinstance(after, dict)
        and before.get("lemma") == after.get("lemma")
        and before.get("features") != after.get("features")
        for before in source_analyses
        for after in replacement_analyses
    )


class TraditionalSingle:
    """Apply exactly one successful canonical traditional operation."""

    def __init__(self, editor: TraditionalEditor | None = None):
        self.editor = editor or TraditionalEditor()

    def apply(
        self,
        text: str,
        *,
        seed: int = 0,
        item_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        return _apply_canonical_edits(
            self.editor,
            text,
            seed=seed,
            n_edits=1,
            item_metadata=item_metadata,
        )


class TraditionalSampled:
    """Apply the shared length-scaled number of canonical traditional edits."""

    def __init__(self, editor: TraditionalEditor | None = None):
        self.editor = editor or TraditionalEditor()

    def apply(
        self,
        text: str,
        *,
        seed: int = 0,
        item_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        return _apply_canonical_edits(
            self.editor,
            text,
            seed=seed,
            n_edits=sample_edit_count(len(text), seed=seed),
            item_metadata=item_metadata,
        )


def _apply_canonical_edits(
    editor: TraditionalEditor,
    text: str,
    *,
    seed: int,
    n_edits: int,
    item_metadata: dict[str, Any] | None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Sample uniformly with replacement across successful edit positions."""
    if n_edits < 1:
        raise ValueError("n_edits must be at least 1")
    rng = random.Random(seed)
    operation_names = [operation.name for operation in traditional_operations_for_language(editor.language)]
    current = text
    applied: list[str] = []
    audit: list[dict[str, Any]] = []
    for edit_index in range(n_edits):
        remaining = list(operation_names)
        attempts: list[dict[str, str]] = []
        while remaining:
            operation_index = rng.randrange(len(remaining))
            operation = remaining.pop(operation_index)
            operation_seed = rng.randrange(2**63)
            outcome = editor.apply(
                current,
                operation,
                seed=operation_seed,
                item_metadata=item_metadata,
            )
            attempt = {"operation": operation}
            if outcome.unavailable_reason is not None:
                attempt["outcome"] = "unavailable"
                attempt["reason"] = outcome.unavailable_reason
                attempts.append(attempt)
                continue
            if outcome.text.split() != current.split():
                attempt["outcome"] = "applied"
                current = outcome.text
                applied.append(operation)
                edit_details = {
                    "operation": operation,
                    "requested_edit_index": edit_index,
                    "attempted_operations": [*attempts, attempt],
                    **outcome.details,
                }
                audit.append(edit_details)
                break
            attempt["outcome"] = "unavailable"
            attempt["reason"] = "no_substantive_token_change"
            attempts.append(attempt)
        else:
            raise TraditionalNoChangeError(
                f"No canonical operation could realize requested edit {edit_index + 1}/{n_edits}",
                audit=[
                    *audit,
                    {
                        "operation": None,
                        "attempted_operations": attempts,
                        "requested_edit_index": edit_index,
                        "outcome": "exhausted",
                    },
                ],
            )
    return current, applied, audit


def _input_seed(base_seed: int, item: PerturbationInput, index: int) -> int:
    identity = item.candidate_id or f"{item.dataset_name}:{item.base_text_id}:{index}"
    raw = f"{base_seed}:{identity}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


class TraditionalMethodAdapter:
    """Registry-compatible adapter for the traditional method family."""

    name = "trad_single"
    perturbation_source = "trad"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.editor = TraditionalEditor(
            language=str(self.config.get("language", "eng")),
            store=self.config.get("store"),
        )

    def _result(
        self,
        item,
        output: str,
        edits: list[str],
        audit: list[dict[str, Any]],
        seed: int,
    ) -> PerturbationResult:
        return PerturbationResult(
            dataset_name=item.dataset_name,
            base_text_id=item.base_text_id,
            text=output,
            source_layer=item.source_layer,
            source_method=item.source_method,
            source_run_id=item.source_run_id,
            parent_candidate_id=item.candidate_id,
            target_layer=int(self.config.get("target_layer", item.source_layer + 1)),
            perturbation_method=self.name,
            perturbation_source=self.perturbation_source,
            run_id=str(self.config.get("run_id", "default")),
            perturbation_edits=edits,
            edit_count=len(edits),
            generator=self.config.get("model"),
            seed=seed,
            method_config={key: value for key, value in self.config.items() if key != "store"},
            metadata={
                **item.metadata,
                "traditional_backend": self._backend_name(edits),
                "traditional_edit_audit": audit,
            },
        )

    def _backend_name(self, edits: list[str]) -> str:
        backends = []
        for edit in edits:
            operation = get_traditional_operation(edit)
            backend = (
                self.editor.morphology_backend
                if operation.backend == "morphology"
                else operation.backend
            )
            if backend not in backends:
                backends.append(backend)
        return "+".join(backends)

    def _generate_item(
        self, item: PerturbationInput, *, index: int
    ) -> PerturbationResult:
        # A resumed layer supplies ``sampling_seed`` so that only its missing
        # candidates explore a new deterministic trajectory.  The original
        # request seed remains in the layer manifest for reproducibility.
        generation_seed = int(self.config.get("sampling_seed", self.config.get("seed", 42)))
        base_seed = _input_seed(generation_seed, item, index)
        try:
            output, edits, audit = self._apply(item.text, seed=base_seed)
        except TraditionalNoChangeError as exc:
            return self._result(
                item,
                SkippedPerturbation(reason="no_change", attempts=1),
                [],
                exc.audit,
                base_seed,
            )
        return self._result(item, output, edits, audit, base_seed)

    def generate(
        self,
        items: list[PerturbationInput],
        runtime: GenerationRuntime | None = None,
    ) -> list[PerturbationResult]:
        target_layer = int(self.config.get("target_layer", 1))
        if not items:
            return []
        n_jobs = max(1, min(int(self.config.get("n_jobs", os.cpu_count() or 1)), len(items)))
        results: list[PerturbationResult | None] = [None] * len(items)
        if n_jobs == 1:
            for index, item in enumerate(items):
                results[index] = self._generate_item(item, index=index)
            return [result for result in results if result is not None]
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_traditional_worker,
            initargs=(self.name, self.config),
        ) as executor:
            futures = [
                executor.submit(_generate_traditional_item, index, item)
                for index, item in enumerate(items)
            ]
            with tqdm(
                total=len(items),
                desc=f"Generating {self.name} layer {target_layer}",
                unit="item",
            ) as progress:
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result
                    progress.update(1)
        return [result for result in results if result is not None]

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str], list[dict[str, Any]]]:
        raise NotImplementedError


class TraditionalSingleMethod(TraditionalMethodAdapter):
    name = "trad_single"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str], list[dict[str, Any]]]:
        return TraditionalSingle(self.editor).apply(text, seed=seed)


class TraditionalSampledMethod(TraditionalMethodAdapter):
    name = "trad_sampled"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str], list[dict[str, Any]]]:
        return TraditionalSampled(self.editor).apply(text, seed=seed)


__all__ = [
    "TraditionalOperation",
    "TRADITIONAL_OPERATIONS",
    "TraditionalEditor",
    "TraditionalSingle",
    "TraditionalSampled",
    "TraditionalMethodAdapter",
    "TraditionalSingleMethod",
    "TraditionalSampledMethod",
    "get_traditional_operation",
    "list_traditional_operations",
    "traditional_operations_for_language",
]
