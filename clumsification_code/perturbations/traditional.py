# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Unified multilingual traditional fluency perturbations.

The public traditional methods are ``trad_single``, ``trad_multi``, and
``unieval_trad``. English-only fluency rules are part of the same operation
inventory rather than a standalone method.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterator

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
from .english_fluency import EnglishFluencyRulePerturber
from .morphology import load_morphology_backend, morphology_backend_name
from .unieval_fluency import apply_unieval_disfluency


_TRADITIONAL_WORKER: "TraditionalMethodAdapter | None" = None


class TraditionalNoChangeError(RuntimeError):
    """A valid traditional operation was inapplicable to one input."""


def _init_traditional_worker(method_name: str, config: dict[str, Any]) -> None:
    global _TRADITIONAL_WORKER
    adapter_type = {
        "trad_single": TraditionalSingleMethod,
        "trad_multi": TraditionalMultiMethod,
        "unieval": UniEvalMethod,
        "unieval_trad": UniEvalTraditionalMethod,
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
    TraditionalOperation("jumble", ("Grammaticality", "Clarity"), "rules", "Swap word order."),
    TraditionalOperation("subject_verb_dis", ("Grammaticality",), "morphology", "Introduce subject–verb disagreement."),
    TraditionalOperation("random_inflection", ("Grammaticality",), "morphology", "Replace a word with another inflected form."),
    TraditionalOperation("typos", ("Grammaticality",), "rules", "Introduce a character-level typo."),
    TraditionalOperation("misplaced_punctuation", ("Grammaticality", "Clarity"), "english_fluency", "Remove or misplace punctuation (English only)."),
    TraditionalOperation("remove_punct", ("Grammaticality", "Clarity"), "english_fluency", "Remove punctuation (English only)."),
    TraditionalOperation("drop_stopwords", ("Grammaticality", "Clarity"), "english_fluency", "Delete function words (English only)."),
    TraditionalOperation("drop_adjectives", ("Clarity",), "english_fluency", "Delete selected adjectives (English only)."),
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
    all other supported languages. English-specific operations are
    intentionally excluded from non-English sampling.
    """
    if normalize_language(language) == "eng":
        return TRADITIONAL_OPERATIONS
    return tuple(operation for operation in TRADITIONAL_OPERATIONS if operation.backend != "english_fluency")


@contextmanager
def _seed_global_random(seed: int) -> Iterator[None]:
    """Isolate English fluency functions that use the module-level RNG."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class TraditionalEditor:
    """Apply one named operation using its language-appropriate backend."""

    def __init__(self, language: str = "eng", store: Any | None = None, *, use_morphology: bool = True):
        self.language = normalize_language(language)
        self.morphology_backend = morphology_backend_name(self.language)
        self.store = store
        self.use_morphology = use_morphology
        self._morph_perturber: Any | None = None
        self._morph_templates: dict[str, Callable[[dict], str]] | None = None
        self._rule_perturber: Any | None = None
        self._rule_templates: dict[str, Callable[[dict], str]] | None = None
        self._english_fluency_perturber: Any | None = None

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

    def _load_rules(self) -> dict[str, Callable[[dict], str]]:
        if self._rule_templates is None:
            self._rule_perturber = MultilingualRulePerturber(self.language, None)
            self._rule_templates = {
                template.name: template.fn
                for template in build_rule_templates(self._rule_perturber)
            }
        return self._rule_templates

    def _load_english_fluency(self) -> Any:
        if self._english_fluency_perturber is None:
            self._english_fluency_perturber = EnglishFluencyRulePerturber()
        return self._english_fluency_perturber

    def apply(
        self,
        text: str,
        operation: str,
        *,
        seed: int = 0,
        item_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Apply one operation and return the resulting text."""
        spec = get_traditional_operation(operation)
        item = {"text": text, **(item_metadata or {})}
        if spec.backend in {"morphology", "rules"}:
            templates = (
                self._load_morphology()
                if spec.backend == "morphology"
                else self._load_rules()
            )
            fn = templates.get(operation)
            if fn is None:
                raise RuntimeError(f"Traditional backend does not implement {operation!r}")
            perturber = (
                self._morph_perturber
                if spec.backend == "morphology"
                else self._rule_perturber
            )
            if perturber is not None:
                perturber.random_seed = seed
            return str(fn(item))

        if self.language != "eng":
            raise ValueError(f"Operation {operation!r} is English-only; language is {self.language!r}")
        fn = getattr(self._load_english_fluency(), operation)
        with _seed_global_random(seed):
            return str(fn(item))


class TraditionalSingle:
    """Apply one sampled or explicitly selected traditional operation."""

    def __init__(self, editor: TraditionalEditor | None = None):
        self.editor = editor or TraditionalEditor()

    def apply(
        self,
        text: str,
        *,
        seed: int = 0,
        operation: str | None = None,
        item_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        rng = random.Random(seed)
        available = traditional_operations_for_language(self.editor.language)
        if operation is not None:
            output = self.editor.apply(
                text, operation, seed=seed, item_metadata=item_metadata
            )
            if output.strip() == text.strip():
                raise TraditionalNoChangeError(
                    f"Traditional operation {operation!r} produced no change"
                )
            return output, [operation]

        # Some operations are inherently inapplicable to individual texts
        # (for example, drop_adjectives when no adjective is present). Try
        # each sampled operation at most once instead of failing the layer.
        names = [item.name for item in available]
        rng.shuffle(names)
        for name in names:
            output = self.editor.apply(
                text, name, seed=seed, item_metadata=item_metadata
            )
            if output.strip() != text.strip():
                return output, [name]
        raise TraditionalNoChangeError("No traditional operation produced a change")


class TraditionalMulti:
    """Apply several traditional operations sequentially."""

    def __init__(self, editor: TraditionalEditor | None = None):
        self.editor = editor or TraditionalEditor()

    def apply(
        self,
        text: str,
        *,
        seed: int = 0,
        n_edits: int = 2,
        operations: list[str] | None = None,
        item_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        if n_edits < 1:
            raise ValueError("n_edits must be at least 1")
        available = [item.name for item in traditional_operations_for_language(self.editor.language)]
        requested = list(operations) if operations is not None else []
        unknown = [name for name in requested if name not in _OPERATION_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown traditional operations: {unknown}")
        unsupported = [name for name in requested if name not in available]
        if unsupported:
            raise ValueError(
                f"Traditional operations are unsupported for {self.editor.language!r}: "
                f"{unsupported}"
            )
        rng = random.Random(seed)
        if requested:
            rng.shuffle(requested)
            selected = requested[:n_edits]
        else:
            selected = rng.sample(available, min(n_edits, len(available)))

        current = text
        applied: list[str] = []
        for index, name in enumerate(selected):
            output = self.editor.apply(
                current,
                name,
                seed=seed + index,
                item_metadata=item_metadata,
            )
            if output.strip() != current.strip():
                current = output
                applied.append(name)
        if not applied:
            raise TraditionalNoChangeError("Traditional operations produced no change")
        return current, applied


class UniEvalTraditional:
    """Apply UniEval noise followed by traditional operations."""

    def __init__(self, editor: TraditionalEditor | None = None):
        self.editor = editor or TraditionalEditor()

    def apply(
        self,
        text: str,
        *,
        seed: int = 0,
        n_noise: int = 1,
        n_edits: int = 1,
        operations: list[str] | None = None,
        item_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        noisy, unieval_edits = apply_unieval_disfluency(
            text,
            n_noise=n_noise,
            python_rng=random.Random(seed),
            numpy_rng=np.random.default_rng(seed),
        )
        traditional, applied = TraditionalMulti(self.editor).apply(
            noisy,
            seed=seed + 1,
            n_edits=n_edits,
            operations=operations,
            item_metadata=item_metadata,
        )
        return traditional, [*(edit["transform_type"] for edit in unieval_edits), *applied]


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

    def _result(self, item, output: str, edits: list[str], seed: int) -> PerturbationResult:
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
        max_attempts = int(self.config.get("max_attempts", 100))
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        base_seed = _input_seed(int(self.config.get("seed", 42)), item, index)
        for attempt in range(max_attempts):
            seed = base_seed + attempt
            try:
                output, edits = self._apply(item.text, seed=seed)
            except TraditionalNoChangeError:
                continue
            if output.strip() != item.text.strip():
                return self._result(item, output, edits, seed)
        return self._result(
            item,
            SkippedPerturbation(reason="no_change", attempts=max_attempts),
            [],
            base_seed + max_attempts - 1,
        )

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

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str]]:
        raise NotImplementedError


class TraditionalSingleMethod(TraditionalMethodAdapter):
    name = "trad_single"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str]]:
        return TraditionalSingle(self.editor).apply(text, seed=seed, operation=self.config.get("operation"))


class TraditionalMultiMethod(TraditionalMethodAdapter):
    name = "trad_multi"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str]]:
        return TraditionalMulti(self.editor).apply(
            text,
            seed=seed,
            n_edits=int(self.config.get("n_edits", 2)),
            operations=self.config.get("operations"),
        )


class UniEvalMethod(TraditionalMethodAdapter):
    name = "unieval"

    def _backend_name(self, edits: list[str]) -> str:
        return "unieval"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str]]:
        output, edits = apply_unieval_disfluency(
            text,
            n_noise=int(self.config.get("n_noise", 1)),
            python_rng=random.Random(seed),
            numpy_rng=np.random.default_rng(seed),
        )
        if output.strip() == text.strip():
            raise TraditionalNoChangeError("UniEval produced no change")
        return output, [str(edit["transform_type"]) for edit in edits]


class UniEvalTraditionalMethod(TraditionalMethodAdapter):
    name = "unieval_trad"

    def _backend_name(self, edits: list[str]) -> str:
        traditional_edits = [edit for edit in edits if edit in _OPERATION_BY_NAME]
        suffix = super()._backend_name(traditional_edits)
        return f"unieval+{suffix}" if suffix else "unieval"

    def _apply(self, text: str, *, seed: int) -> tuple[str, list[str]]:
        operations = self.config.get("operations")
        if operations is None and self.editor.language == "eng":
            # This method is UniEval noise plus the two intended English
            # Lemminflect morphology edits; unrelated traditional rules are
            # not part of its default semantics.
            operations = ["subject_verb_dis", "random_inflection"]
        return UniEvalTraditional(self.editor).apply(
            text,
            seed=seed,
            n_noise=int(self.config.get("n_noise", 1)),
            n_edits=int(self.config.get("n_edits", 1)),
            operations=operations,
        )


__all__ = [
    "TraditionalOperation",
    "TRADITIONAL_OPERATIONS",
    "TraditionalEditor",
    "TraditionalSingle",
    "TraditionalMulti",
    "UniEvalTraditional",
    "TraditionalMethodAdapter",
    "TraditionalSingleMethod",
    "TraditionalMultiMethod",
    "UniEvalMethod",
    "UniEvalTraditionalMethod",
    "get_traditional_operation",
    "list_traditional_operations",
    "traditional_operations_for_language",
]
