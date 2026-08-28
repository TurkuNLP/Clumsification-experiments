# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Backbone-specific representation policies for FE models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneProfile:
    canonical_name: str
    pooling: str
    trust_remote_code: bool = True


_PROFILES = {
    "jinaai/jina-embeddings-v5-text-small": BackboneProfile("jinaai/jina-embeddings-v5-text-small", "last_token"),
    "qwen/qwen3-embedding-0.6b": BackboneProfile("Qwen/Qwen3-Embedding-0.6B", "last_token"),
    "intfloat/multilingual-e5-large": BackboneProfile("intfloat/multilingual-e5-large", "mean", False),
    "microsoft/harrier-oss-v1-0.6b": BackboneProfile("microsoft/harrier-oss-v1-0.6b", "last_token"),
}


def resolve_backbone_profile(model_name: str) -> BackboneProfile:
    try:
        return _PROFILES[model_name.strip().lower()]
    except KeyError as exc:
        supported = ", ".join(p.canonical_name for p in _PROFILES.values())
        raise ValueError(
            f"No FE backbone profile for {model_name!r}. Supported models: {supported}. "
            "Pass an explicit --pooling override for an experimental model."
        ) from exc


def resolve_pooling(model_name: str, pooling: str = "auto") -> str:
    if pooling == "auto":
        return resolve_backbone_profile(model_name).pooling
    if pooling not in {"mean", "last_token"}:
        raise ValueError(f"Unknown pooling strategy: {pooling!r}")
    return pooling
