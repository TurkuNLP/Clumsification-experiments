# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Load, validate, and render repository prompt specifications.

Prompt text belongs in JSON files under ``data/prompts``. This module contains
only the transport-independent mechanics needed to use those files: stable path
resolution, schema validation, template-variable validation, rendering, and
reproducibility metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from string import Formatter
from typing import Any, Mapping, Sequence


PROMPT_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_ROOT = REPOSITORY_ROOT / "data" / "prompts"

_ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "version",
    "description",
    "required_variables",
    "messages",
    "metadata",
}
_ALLOWED_MESSAGE_KEYS = {"role", "content"}
_ALLOWED_ROLES = {"system", "user", "assistant"}


class PromptSpecError(ValueError):
    """Raised when a prompt file or its rendering inputs are invalid."""


@dataclass(frozen=True)
class PromptMessage:
    """One validated, unrendered chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class PromptSpec:
    """A validated and versioned prompt specification.

    The prompt ID, version, and source path are retained for evaluation
    metadata.
    """

    schema_version: int
    prompt_id: str
    version: str
    description: str
    required_variables: tuple[str, ...]
    messages: tuple[PromptMessage, ...]
    metadata: Mapping[str, Any]
    source_path: Path

    def render_messages(self, values: Mapping[str, object]) -> list[dict[str, str]]:
        """Render the prompt after enforcing its exact variable contract."""

        provided = set(values)
        required = set(self.required_variables)
        missing = sorted(required - provided)
        unexpected = sorted(provided - required)

        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing variables: {missing}")
            if unexpected:
                details.append(f"unexpected variables: {unexpected}")
            raise PromptSpecError(
                f"Cannot render prompt {self.prompt_id!r}: " + "; ".join(details)
            )

        rendered = []
        for message in self.messages:
            try:
                content = message.content.format_map(dict(values))
            except (KeyError, ValueError) as exc:
                raise PromptSpecError(
                    f"Failed to render prompt {self.prompt_id!r}: {exc}"
                ) from exc
            rendered.append({"role": message.role, "content": content})
        return rendered

    def reproducibility_metadata(self) -> dict[str, object]:
        """Return stable prompt identity fields suitable for run metadata."""

        try:
            source = str(self.source_path.relative_to(REPOSITORY_ROOT))
        except ValueError:
            source = str(self.source_path)

        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.version,
            "prompt_schema_version": self.schema_version,
            "prompt_path": source,
        }


def resolve_prompt_path(
    path: str | Path,
    *,
    prompt_root: str | Path | None = None,
) -> Path:
    """Resolve a prompt path without depending on the process working directory.

    Short relative paths are interpreted below ``data/prompts``. Paths beginning
    with ``data/`` are interpreted from the repository root, which also supports
    existing repository-style CLI paths during future migrations.
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if candidate.parts and candidate.parts[0] == "data":
        return (REPOSITORY_ROOT / candidate).resolve()

    root = Path(prompt_root).expanduser() if prompt_root is not None else DEFAULT_PROMPT_ROOT
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    return (root / candidate).resolve()


def load_prompt_spec(
    path: str | Path,
    *,
    prompt_root: str | Path | None = None,
) -> PromptSpec:
    """Load and fully validate one JSON prompt specification."""

    resolved_path = resolve_prompt_path(path, prompt_root=prompt_root)
    try:
        raw_bytes = resolved_path.read_bytes()
    except OSError as exc:
        raise PromptSpecError(f"Could not read prompt file {resolved_path}: {exc}") from exc

    try:
        document = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except UnicodeDecodeError as exc:
        raise PromptSpecError(f"Prompt file {resolved_path} is not valid UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise PromptSpecError(f"Prompt file {resolved_path} is not valid JSON: {exc}") from exc

    return _validate_document(
        document,
        source_path=resolved_path,
    )


def load_prompt_data(
    path: str | Path,
    *,
    prompt_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a JSON prompt-data object that is not a chat prompt specification.

    Rubric collections and task/aspect prompt tables have method-specific
    shapes. They share path handling and basic JSON validation without being
    forced into the chat-message schema used by :func:`load_prompt_spec`.
    """

    resolved_path = resolve_prompt_path(path, prompt_root=prompt_root)
    try:
        document = json.loads(
            resolved_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except OSError as exc:
        raise PromptSpecError(f"Could not read prompt file {resolved_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PromptSpecError(f"Prompt file {resolved_path} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise PromptSpecError(f"Prompt file {resolved_path} must contain a JSON object.")
    return document


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromptSpecError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _validate_document(
    document: object,
    *,
    source_path: Path,
) -> PromptSpec:
    if not isinstance(document, dict):
        raise PromptSpecError(f"Prompt file {source_path} must contain a JSON object.")

    unknown_keys = sorted(set(document) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise PromptSpecError(f"Prompt file {source_path} has unknown keys: {unknown_keys}.")

    schema_version = document.get("schema_version")
    if schema_version != PROMPT_SCHEMA_VERSION:
        raise PromptSpecError(
            f"Prompt file {source_path} uses schema_version {schema_version!r}; "
            f"expected {PROMPT_SCHEMA_VERSION}."
        )

    prompt_id = _required_nonempty_string(document, "id", source_path)
    version = _required_nonempty_string(document, "version", source_path)
    description = document.get("description", "")
    if not isinstance(description, str):
        raise PromptSpecError(f"Prompt file {source_path}: 'description' must be a string.")

    required_variables = _validate_required_variables(
        document.get("required_variables"), source_path
    )
    messages = _validate_messages(document.get("messages"), source_path)

    discovered_variables: set[str] = set()
    for message in messages:
        discovered_variables.update(_template_fields(message.content, source_path))
    if discovered_variables != set(required_variables):
        raise PromptSpecError(
            f"Prompt file {source_path}: required_variables must exactly match the "
            f"message templates; declared={sorted(required_variables)}, "
            f"discovered={sorted(discovered_variables)}."
        )

    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PromptSpecError(f"Prompt file {source_path}: 'metadata' must be an object.")

    return PromptSpec(
        schema_version=schema_version,
        prompt_id=prompt_id,
        version=version,
        description=description,
        required_variables=required_variables,
        messages=messages,
        metadata=metadata,
        source_path=source_path,
    )


def _required_nonempty_string(document: Mapping[str, Any], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromptSpecError(f"Prompt file {path}: {key!r} must be a non-empty string.")
    return value


def _validate_required_variables(value: object, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PromptSpecError(f"Prompt file {path}: 'required_variables' must be a list.")
    if not all(isinstance(item, str) and item for item in value):
        raise PromptSpecError(
            f"Prompt file {path}: every required variable must be a non-empty string."
        )
    if len(value) != len(set(value)):
        raise PromptSpecError(f"Prompt file {path}: required variables must be unique.")
    return tuple(value)


def _validate_messages(value: object, path: Path) -> tuple[PromptMessage, ...]:
    if not isinstance(value, list) or not value:
        raise PromptSpecError(f"Prompt file {path}: 'messages' must be a non-empty list.")

    messages = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise PromptSpecError(f"Prompt file {path}: message {index} must be an object.")
        unknown_keys = sorted(set(item) - _ALLOWED_MESSAGE_KEYS)
        if unknown_keys:
            raise PromptSpecError(
                f"Prompt file {path}: message {index} has unknown keys: {unknown_keys}."
            )
        role = item.get("role")
        content = item.get("content")
        if role not in _ALLOWED_ROLES:
            raise PromptSpecError(
                f"Prompt file {path}: message {index} role must be one of "
                f"{sorted(_ALLOWED_ROLES)}, got {role!r}."
            )
        if not isinstance(content, str):
            raise PromptSpecError(
                f"Prompt file {path}: message {index} content must be a string."
            )
        messages.append(PromptMessage(role=role, content=content))
    return tuple(messages)


def _template_fields(template: str, path: Path) -> set[str]:
    fields: set[str] = set()
    try:
        parsed = Formatter().parse(template)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name:
                raise PromptSpecError(f"Prompt file {path}: positional fields are not supported.")
            if "." in field_name or "[" in field_name or "]" in field_name:
                raise PromptSpecError(
                    f"Prompt file {path}: template field {field_name!r} must be a simple name."
                )
            if format_spec or conversion:
                raise PromptSpecError(
                    f"Prompt file {path}: formatting and conversions are not supported "
                    f"for field {field_name!r}."
                )
            fields.add(field_name)
    except ValueError as exc:
        raise PromptSpecError(f"Prompt file {path} has an invalid template: {exc}") from exc
    return fields
