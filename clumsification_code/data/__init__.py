# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical dataset preparation contracts and helpers."""

from .schemas import (
    CandidateRecord,
    GenerationSpec,
    HFBuildSpec,
    LayerManifestEntry,
    OriginalRecord,
    PerturbationManifest,
    ScoreRecord,
)
from .repository import DatasetRepository


def build_hf_dataset(*args, **kwargs):
    """Load the optional Arrow-backed builder only when it is invoked."""
    from .hf_dataset import build_hf_dataset as _build_hf_dataset

    return _build_hf_dataset(*args, **kwargs)

__all__ = [
    "CandidateRecord",
    "DatasetRepository",
    "GenerationSpec",
    "HFBuildSpec",
    "LayerManifestEntry",
    "OriginalRecord",
    "PerturbationManifest",
    "ScoreRecord",
    "build_hf_dataset",
]
