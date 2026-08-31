# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Fixed-prompt LLM perturbation method.

This module owns prompt construction but not model inference; the shared chat
runner consumes its returned messages.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from clumsification_code.data.io import canonical_json_hash, sha256_file

from .schemas import GenerationRuntime, PerturbationInput, PerturbationResult


ZERO_SHOT_METHOD = "llm_zero_shot"
_PROMPT_KEYS = ("base_prompt", "system_prompt", "ex_user", "ex_assistant", "context_prompt_user")


@dataclass(frozen=True)
class ZeroShotPromptSpec:
    base_prompt: str
    system_prompt: str
    ex_user: str
    ex_assistant: str
    context_prompt_user: str
    path: str


def load_zero_shot_prompt(
    language: str,
    *,
    prompt_root: str | Path = "data/perturbation_prompts",
) -> ZeroShotPromptSpec:
    """Load the canonical prompt, falling back to the legacy filename."""
    root = Path(prompt_root) / language
    candidates = (root / "llm_zero_shot.json", root / "clumsification.json")
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            missing = [key for key in _PROMPT_KEYS if key not in data]
            if missing:
                raise ValueError(f"Prompt {path} is missing fields: {missing}")
            return ZeroShotPromptSpec(
                **{key: str(data[key]) for key in _PROMPT_KEYS},
                path=str(path),
            )
    paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No zero-shot prompt found; checked {paths}")


def build_zero_shot_messages(
    item: dict[str, Any],
    language: str,
    *,
    prompt_root: str | Path = "data/perturbation_prompts",
) -> list[dict[str, str]]:
    """Render one legacy-compatible fixed-prompt chat request."""
    prompt = load_zero_shot_prompt(language, prompt_root=prompt_root)
    text = str(item.get("text", "")).replace("\n", " ")
    max_length = item.get("max_length")
    if max_length is None:
        max_length = min(int(len(text) * 1.1), len(text) + 500)
    context = (
        prompt.context_prompt_user
        + "\n The absolute maximum length of the edited text in characters is "
        + str(max_length)
        + ". You are never allowed to edit a text to be longer than this."
        + " Now, edit this text: \n"
        + text
    )
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.base_prompt + prompt.ex_user},
        {"role": "assistant", "content": prompt.ex_assistant},
        {"role": "user", "content": context},
    ]


class ZeroShotLLMMethod:
    """Prompt-only adapter used by the canonical ``llm_zero_shot`` method."""

    name = ZERO_SHOT_METHOD
    perturbation_source = "LLM"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.language = str(self.config.get("language", "english"))
        self.prompt_root = self.config.get("prompt_root", "data/perturbation_prompts")

    def build_prompts(self, items: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        return [
            build_zero_shot_messages(
                item,
                self.language,
                prompt_root=self.prompt_root,
            )
            for item in items
        ]

    def generate(
        self,
        items: list[PerturbationInput],
        runtime: GenerationRuntime,
    ) -> list[PerturbationResult]:
        prompts = self.build_prompts(
            [item.metadata | {"text": item.text} for item in items]
        )
        prompt_spec = load_zero_shot_prompt(
            self.language, prompt_root=self.prompt_root
        )
        prompt_file_hash = sha256_file(prompt_spec.path)
        model, outputs = runtime.run_chat(self.config, prompts)
        return [
            PerturbationResult(
                dataset_name=item.dataset_name,
                base_text_id=item.base_text_id,
                text=output,
                source_layer=item.source_layer,
                source_method=item.source_method,
                source_run_id=item.source_run_id,
                parent_candidate_id=item.candidate_id,
                target_layer=int(self.config["target_layer"]),
                perturbation_method=self.name,
                perturbation_source=self.perturbation_source,
                run_id=str(self.config["run_id"]),
                generator=model,
                seed=int(self.config.get("seed", 42)),
                prompt_version="llm-zero-shot-v1",
                prompt_hash=canonical_json_hash(prompts[index]),
                method_config=dict(self.config),
                metadata={
                    "prompt_file_hash": prompt_file_hash,
                    "max_output_chars": int(
                        item.metadata.get("max_length")
                        or min(int(len(item.text.replace("\n", " ")) * 1.1), len(item.text.replace("\n", " ")) + 500)
                    ),
                },
            )
            for index, (item, output) in enumerate(zip(items, outputs))
        ]


__all__ = [
    "ZERO_SHOT_METHOD",
    "ZeroShotPromptSpec",
    "ZeroShotLLMMethod",
    "build_zero_shot_messages",
    "load_zero_shot_prompt",
]
