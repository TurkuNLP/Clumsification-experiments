# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical language-aware morphology backend selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .english_lemminflect import LemminflectStore
from .rule_based_multilingual import load_unimorph_store, normalize_language


@dataclass(frozen=True)
class MorphologyBackend:
    language: str
    name: str
    store: Any


def morphology_backend_name(language: str) -> str:
    """Return the backend mandated for a supported language."""
    return "lemminflect" if normalize_language(language) == "eng" else "unimorph"


def load_morphology_backend(
    language: str,
    *,
    store: Any | None = None,
) -> MorphologyBackend:
    """Load Lemminflect for English and UniMorph for every other language."""
    normalized = normalize_language(language)
    name = morphology_backend_name(normalized)
    if store is None:
        store = LemminflectStore() if normalized == "eng" else load_unimorph_store(normalized)
    if hasattr(store, "has_language") and not store.has_language(normalized):
        raise ValueError(
            f"Supplied {name} store does not contain language {normalized!r}"
        )
    return MorphologyBackend(language=normalized, name=name, store=store)


__all__ = [
    "MorphologyBackend",
    "load_morphology_backend",
    "morphology_backend_name",
]
