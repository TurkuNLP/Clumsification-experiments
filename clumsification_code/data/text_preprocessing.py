"""Shared text preprocessing for model-ready FE datasets."""
from __future__ import annotations


TEXT_PREPROCESSING_NAME = "collapse_whitespace_v1"


def normalize_model_text(text: str) -> str:
    """Collapse every Unicode whitespace run to one ASCII space."""
    if not isinstance(text, str):
        raise TypeError("Model text must be a string")
    return " ".join(text.split())


__all__ = ["TEXT_PREPROCESSING_NAME", "normalize_model_text"]
