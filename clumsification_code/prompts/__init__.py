# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Shared loading and rendering helpers for versioned prompt files."""

from clumsification_code.prompts.loader import (
    DEFAULT_PROMPT_ROOT,
    PROMPT_SCHEMA_VERSION,
    PromptMessage,
    PromptSpec,
    PromptSpecError,
    load_prompt_data,
    load_prompt_spec,
    resolve_prompt_path,
)

__all__ = [
    "DEFAULT_PROMPT_ROOT",
    "PROMPT_SCHEMA_VERSION",
    "PromptMessage",
    "PromptSpec",
    "PromptSpecError",
    "load_prompt_data",
    "load_prompt_spec",
    "resolve_prompt_path",
]
