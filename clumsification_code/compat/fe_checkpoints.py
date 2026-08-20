# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Compatibility routing for checkpoints created before FE terminology."""

from pathlib import Path
from typing import Any


LEGACY_HEAD_FILENAME = "ltr_head.pt"


def find_legacy_head(model_dir: str) -> str:
    legacy_path = Path(model_dir) / LEGACY_HEAD_FILENAME
    if legacy_path.exists():
        return str(legacy_path)
    raise FileNotFoundError(
        f"Could not find fe_head.pt or {LEGACY_HEAD_FILENAME} in {model_dir}. "
        "Pass the final directory produced by the trainer."
    )


def normalize_legacy_head_state(head_state: dict[str, Any]) -> dict[str, Any]:
    """Translate the old checkpoint schema to the canonical FE schema."""
    if "evaluation_head" in head_state:
        return head_state
    legacy_state = head_state["scorer"]
    return {
        **head_state,
        "evaluation_head": {
            key.removeprefix("scorer."): value
            for key, value in legacy_state.items()
        },
    }
