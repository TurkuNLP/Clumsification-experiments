# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical identity helpers for perturbation candidates."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _identifier(value: object, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field_name} must be a string or integer identifier")
    result = str(value)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _component(value: str, *, maximum: int = 40) -> str:
    """Return a readable, path-neutral prefix; the identity hash remains exact."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (normalized or "value")[:maximum]


def _identity_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=10).hexdigest()


def make_candidate_id(
    *,
    dataset_name: str,
    perturbation_method: str,
    run_id: str,
    base_text_id: str | int,
    target_layer: int,
    parent_candidate_id: str,
    candidate_index: int,
) -> str:
    """Create a deterministic ID unique to one method run and parent candidate."""
    dataset = _identifier(dataset_name, "dataset_name")
    method = _identifier(perturbation_method, "perturbation_method")
    run = _identifier(run_id, "run_id")
    base_id = _identifier(base_text_id, "base_text_id")
    parent_id = _identifier(parent_candidate_id, "parent_candidate_id")
    if isinstance(target_layer, bool) or not isinstance(target_layer, int) or target_layer < 1:
        raise ValueError("target_layer must be a positive integer")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise ValueError("candidate_index must be a non-negative integer")
    payload = {
        "dataset_name": dataset,
        "perturbation_method": method,
        "run_id": run,
        "base_text_id": base_id,
        "target_layer": target_layer,
        "parent_candidate_id": parent_id,
        "candidate_index": candidate_index,
    }
    return (
        f"candidate__{_component(dataset)}__{_component(method)}__"
        f"{_component(run)}__base_{_component(base_id)}__layer_{target_layer}__"
        f"candidate_{candidate_index:06d}__{_identity_hash(payload)}"
    )


def make_original_candidate_id(*, dataset_name: str, base_text_id: str | int) -> str:
    """Create the stable candidate ID for an unperturbed original text."""
    return f"original__{dataset_name}__base_{base_text_id}"


__all__ = [
    "make_candidate_id",
    "make_original_candidate_id",
]
