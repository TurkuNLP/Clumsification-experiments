# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical perturbation-method registry.

Method implementations are registered in later migration steps.  The
registry is introduced separately so CLI validation and dataset metadata can
use one stable vocabulary during the transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schemas import PerturbationMethod
from .llm_zero_shot import ZeroShotLLMMethod
from .llm_sampled import SampledLLMMethod
from .traditional import (
    TraditionalMultiMethod,
    TraditionalSingleMethod,
    UniEvalMethod,
    UniEvalTraditionalMethod,
)


PerturbationFactory = Callable[[dict[str, Any]], PerturbationMethod]


@dataclass(frozen=True)
class MethodSpec:
    """Metadata for one canonical perturbation method."""

    name: str
    perturbation_source: str
    description: str
    factory: PerturbationFactory | None = None

    @property
    def implemented(self) -> bool:
        return self.factory is not None

    def create(self, config: dict[str, Any] | None = None) -> PerturbationMethod:
        """Build and validate one implementation of the uniform method contract."""
        if self.factory is None:
            raise RuntimeError(f"Method is not implemented: {self.name}")
        method = self.factory(dict(config or {}))
        if not isinstance(method, PerturbationMethod):
            raise TypeError(f"Method {self.name!r} does not implement PerturbationMethod")
        if method.name != self.name or method.perturbation_source != self.perturbation_source:
            raise ValueError(
                f"Method implementation metadata disagrees with registry entry {self.name!r}"
            )
        return method


_METHODS: dict[str, MethodSpec] = {
    "llm_zero_shot": MethodSpec(
        "llm_zero_shot", "LLM", "Fixed-prompt LLM perturbation.",
        factory=lambda config: ZeroShotLLMMethod(config),
    ),
    "llm_sampled": MethodSpec(
        "llm_sampled", "LLM", "LLM perturbation with sampled edit operations.",
        factory=lambda config: SampledLLMMethod(config),
    ),
    "unieval": MethodSpec(
        "unieval", "trad", "UniEval insert/delete/shuffle perturbation.",
        factory=lambda config: UniEvalMethod(config),
    ),
    "trad_single": MethodSpec(
        "trad_single", "trad", "One sampled traditional fluency edit.",
        factory=lambda config: TraditionalSingleMethod(config),
    ),
    "trad_multi": MethodSpec(
        "trad_multi", "trad", "Multiple sampled traditional fluency edits.",
        factory=lambda config: TraditionalMultiMethod(config),
    ),
    "unieval_trad": MethodSpec(
        "unieval_trad",
        "trad",
        "UniEval perturbation followed by traditional, complementary fluency edits like breaking S-V agreement.",
        factory=lambda config: UniEvalTraditionalMethod(config),
    ),
}


def list_method_specs(*, implemented_only: bool = False) -> tuple[MethodSpec, ...]:
    """Return canonical methods in stable registration order."""
    specs = tuple(_METHODS.values())
    if implemented_only:
        specs = tuple(spec for spec in specs if spec.implemented)
    return specs


def get_method_spec(name: str) -> MethodSpec:
    """Resolve a canonical method name, without implicit aliases."""
    try:
        return _METHODS[name]
    except KeyError as exc:
        valid = ", ".join(_METHODS)
        raise ValueError(f"Unknown perturbation method {name!r}; choose one of: {valid}") from exc


def register_method(
    name: str,
    factory: PerturbationFactory,
    *,
    perturbation_source: str | None = None,
    description: str | None = None,
) -> MethodSpec:
    """Attach an implementation to an existing canonical method.

    New names are rejected intentionally: the canonical vocabulary should be
    reviewed before it becomes part of dataset provenance.
    """
    current = get_method_spec(name)
    updated = MethodSpec(
        name=current.name,
        perturbation_source=perturbation_source or current.perturbation_source,
        description=description or current.description,
        factory=factory,
    )
    _METHODS[name] = updated
    return updated


__all__ = [
    "MethodSpec",
    "get_method_spec",
    "list_method_specs",
    "register_method",
]
