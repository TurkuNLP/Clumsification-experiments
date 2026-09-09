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
from .llm_sampled import SampledLLMMethod, SingleLLMMethod
from .traditional import (
    TraditionalSampledMethod,
    TraditionalSingleMethod,
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
    "llm_single": MethodSpec(
        "llm_single", "LLM", "LLM perturbation with one sampled edit operation.",
        factory=lambda config: SingleLLMMethod(config),
    ),
    "llm_sampled": MethodSpec(
        "llm_sampled", "LLM", "LLM perturbation with sampled edit operations.",
        factory=lambda config: SampledLLMMethod(config),
    ),
    "trad_single": MethodSpec(
        "trad_single", "trad", "One sampled traditional fluency edit.",
        factory=lambda config: TraditionalSingleMethod(config),
    ),
    "trad_sampled": MethodSpec(
        "trad_sampled", "trad", "Length-scaled sampled traditional fluency edits.",
        factory=lambda config: TraditionalSampledMethod(config),
    ),
}

def list_method_specs(*, implemented_only: bool = False) -> tuple[MethodSpec, ...]:
    """Return canonical methods in stable registration order."""
    specs = tuple(_METHODS.values())
    if implemented_only:
        specs = tuple(spec for spec in specs if spec.implemented)
    return specs


def get_method_spec(name: str) -> MethodSpec:
    """Resolve one of the four supported perturbation methods."""
    try:
        return _METHODS[name]
    except KeyError as exc:
        valid = ", ".join(_METHODS)
        raise ValueError(f"Unknown perturbation method {name!r}; choose one of: {valid}") from exc


__all__ = [
    "MethodSpec",
    "get_method_spec",
    "list_method_specs",
]
