# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from OpenAI_lib import get_google_client


client = get_google_client()


# ---------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------

LANGUAGE_NAMES = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "es": "Spanish",
    "cs": "Czech",
    "gl": "Galician",
    "is": "Icelandic",
}


class GeneratedText(BaseModel):
    prompt_sentence: str = Field(
        description="The sentence given as a reference."
    )
    text: str = Field(
        description="The full generated text in the requested target language."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate UD documents using Google GenAI/Gemma/Gemini batch processing."
    )

    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Benchmark folder containing language-specific UD data folders.",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Target language code, e.g. en, fi, sv, es.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Google model to use, e.g. gemini-3.5-flash, gemini-3.5-pro, "
            "or a Gemma model id if available in your Google account/API."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["submit", "retrieve"],
        help="Use 'submit' to create a batch, then 'retrieve' once it has completed.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to a batch manifest JSON file. "
            "If omitted in retrieve mode, the latest matching manifest is used."
        ),
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help="In retrieve mode, wait until the batch is completed.",
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval when --wait is used.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Optional generation temperature. If omitted, provider/model default is used. "
            "For scientific comparison, set this explicitly and identically across providers."
        ),
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top_p. If omitted, provider/model default is used.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top_k. If omitted, provider/model default is used.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional max output tokens.",
    )

    parser.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="Optional Google batch display name.",
    )

    parser.add_argument(
        "--json-key-style",
        type=str,
        default="snake",
        choices=["snake", "camel"],
        help=(
            "How to write generation config keys inside the JSONL request. "
            "Google Python examples commonly use snake_case; REST examples often use camelCase."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Paths / IO
# ---------------------------------------------------------------------

CACHE_DIR_NAME = "cache"


def safe_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-")


def input_path(args: argparse.Namespace) -> Path:
    return args.base_folder / args.language / "ud_data.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_dir_for_data_dir(data_dir: Path, *, create: bool = True) -> Path:
    cache_dir = data_dir / CACHE_DIR_NAME

    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def cache_path_for_output(output_path: Path, suffix: str) -> Path:
    cache_dir = cache_dir_for_data_dir(output_path.parent)
    return cache_dir / f"{output_path.stem}{suffix}"


def next_output_path(data_dir: Path, model: str) -> Path:
    model_part = safe_filename_part(model)
    prefix = f"{model_part}_google_regens_"

    cache_dir = cache_dir_for_data_dir(data_dir)

    existing_numbers: list[int] = []

    for path in data_dir.glob(f"{prefix}*.jsonl"):
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.jsonl", path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    run_number = len(existing_numbers)

    while True:
        candidate = data_dir / f"{prefix}{run_number}.jsonl"

        manifest = cache_dir / f"{candidate.stem}.manifest.json"
        batch_input = cache_dir / f"{candidate.stem}.batch_input.jsonl"
        batch_output = cache_dir / f"{candidate.stem}.batch_output.jsonl"
        errors = cache_dir / f"{candidate.stem}.errors.jsonl"

        if (
            not candidate.exists()
            and not manifest.exists()
            and not batch_input.exists()
            and not batch_output.exists()
            and not errors.exists()
        ):
            return candidate

        run_number += 1


def manifest_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".manifest.json")


def batch_input_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".batch_input.jsonl")


def batch_output_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".batch_output.jsonl")


def error_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".errors.jsonl")


def find_latest_manifest(data_dir: Path, model: str) -> Path:
    model_part = safe_filename_part(model)
    prefix = f"{model_part}_google_regens_"

    cache_dir = cache_dir_for_data_dir(data_dir, create=False)

    manifests: list[Path] = []

    if cache_dir.exists():
        manifests.extend(cache_dir.glob(f"{prefix}*.manifest.json"))

    # Backwards-compatibility if a manifest was written outside cache.
    manifests.extend(data_dir.glob(f"{prefix}*.manifest.json"))

    manifests = sorted(
        manifests,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        raise FileNotFoundError(
            f"No manifest found in {cache_dir} or {data_dir} for model={model}."
        )

    return manifests[0]


# ---------------------------------------------------------------------
# Prompting / schema
# ---------------------------------------------------------------------

def get_language_name(language_code: str) -> str:
    return LANGUAGE_NAMES.get(language_code, language_code)


def generated_text_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prompt_sentence": {
                "type": "string",
                "description": "The sentence given as a reference.",
            },
            "text": {
                "type": "string",
                "description": "The full generated text in the requested target language.",
            },
        },
        "required": ["prompt_sentence", "text"],
    }


def make_prompt(document: dict[str, Any], language_code: str) -> str:
    language_name = get_language_name(language_code)

    return f"""You are tasked with writing a natural sounding, fluent text given a set of conditions.

Target language: {language_name} ({language_code})

Write the generated document in the target language only.

Task:
Based on the given sentence, write one complete {document["register"]} text.
Do not include the given text in your writing, only use it as a reference.
The generated text must be between {document["text_sent_amount"]} sentences long.
The text should be a full fluent text that flows naturally instead of a collection or listing of sentences.

Given text:
{document["prompt_sentence"]}
"""


def make_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Google batch JSONL accepts a GenerateContentRequest-like object.

    Google examples vary between SDK-style snake_case and REST-style camelCase.
    This function supports either via --json-key-style.
    """
    schema = generated_text_json_schema()

    if args.json_key_style == "camel":
        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }

        if args.temperature is not None:
            config["temperature"] = args.temperature
        if args.top_p is not None:
            config["topP"] = args.top_p
        if args.top_k is not None:
            config["topK"] = args.top_k
        if args.max_output_tokens is not None:
            config["maxOutputTokens"] = args.max_output_tokens

    else:
        config = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        if args.temperature is not None:
            config["temperature"] = args.temperature
        if args.top_p is not None:
            config["top_p"] = args.top_p
        if args.top_k is not None:
            config["top_k"] = args.top_k
        if args.max_output_tokens is not None:
            config["max_output_tokens"] = args.max_output_tokens

    return config


def make_batch_request(
    document: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Google Batch JSONL line.

    Shape follows Google's file-batch pattern:
        {"key": "doc-0", "request": {GenerateContentRequest...}}

    The key is critical for rejoining outputs to inputs.
    """
    prompt = make_prompt(document, args.language)
    generation_config = make_generation_config(args)

    if args.json_key_style == "camel":
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
    else:
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generation_config": generation_config,
        }

    return {
        "key": f"doc-{index}",
        "request": request,
    }


# ---------------------------------------------------------------------
# Google object helpers
# ---------------------------------------------------------------------

def to_plain_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [to_plain_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_plain_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): to_plain_jsonable(v) for k, v in value.items()}

    if hasattr(value, "model_dump"):
        return to_plain_jsonable(value.model_dump())

    if hasattr(value, "dict"):
        return to_plain_jsonable(value.dict())

    return str(value)


def get_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)

    if state is None and isinstance(batch_job, dict):
        state = batch_job.get("state")

    if hasattr(state, "name"):
        return state.name

    if isinstance(state, str):
        return state

    return str(state)


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


FAILED_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------

def submit_batch(args: argparse.Namespace) -> None:
    source_path = input_path(args)

    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    documents = read_jsonl(source_path)

    output_path = next_output_path(source_path.parent, args.model)
    batch_input_path = batch_input_path_for_output(output_path)
    manifest_path = manifest_path_for_output(output_path)

    batch_input_path.parent.mkdir(parents=True, exist_ok=True)

    with batch_input_path.open("w", encoding="utf-8") as writer:
        for index, document in enumerate(documents):
            request = make_batch_request(
                document=document,
                index=index,
                args=args,
            )
            writer.write(json.dumps(request, ensure_ascii=False) + "\n")

    display_name = args.display_name or output_path.stem

    uploaded_file = client.files.upload(
        file=str(batch_input_path),
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )

    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={
            "display_name": display_name,
        },
    )

    manifest = {
        "provider": "google",
        "batch_name": batch_job.name,
        "batch_state": get_state_name(batch_job),
        "model": args.model,
        "language": args.language,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "batch_input_path": str(batch_input_path),
        "uploaded_file_name": uploaded_file.name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "json_key_style": args.json_key_style,
        "schema": generated_text_json_schema(),
        "num_inputs": len(documents),
        "created_at_unix": time.time(),
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Submitted Google batch: {batch_job.name}")
    print(f"State: {get_state_name(batch_job)}")
    print(f"Manifest: {manifest_path}")
    print(f"Planned output: {output_path}")


# ---------------------------------------------------------------------
# Retrieve / parse
# ---------------------------------------------------------------------

def parse_custom_id(custom_id: str) -> int:
    match = re.fullmatch(r"doc-(\d+)", custom_id)

    if not match:
        raise ValueError(f"Unexpected custom_id/key: {custom_id}")

    return int(match.group(1))


def extract_text_from_generate_content_response(response: dict[str, Any]) -> str:
    """
    Robustly extract generated text from a Google GenerateContentResponse-like dict.

    Handles common shapes:
      - {"text": "..."}
      - {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
      - objects converted through model_dump()
    """
    if "text" in response and isinstance(response["text"], str):
        return response["text"]

    candidates = response.get("candidates")
    if isinstance(candidates, list):
        text_parts: list[str] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            content = candidate.get("content")
            if not isinstance(content, dict):
                continue

            parts = content.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

        if text_parts:
            return "".join(text_parts)

    # Some SDK objects include nested response-like fields.
    for key in ("output_text", "content", "message"):
        value = response.get(key)
        if isinstance(value, str):
            return value

    raise ValueError("Could not find generated text in Google response.")


def normalize_google_batch_row(row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, Any]:
    """
    Returns:
        key, response, error

    Google batch output shapes can vary by SDK/API version. This attempts to
    handle the common ones while preserving raw errors if parsing fails.
    """
    key = (
        row.get("key")
        or row.get("custom_id")
        or row.get("metadata", {}).get("key")
        or row.get("request", {}).get("key")
    )

    response = row.get("response")
    error = row.get("error") or row.get("status")

    # Some file outputs may be directly GenerateContentResponse-like with key attached.
    if response is None and "candidates" in row:
        response = row

    # Some wrappers may use "inlineResponse".
    if response is None and isinstance(row.get("inlineResponse"), dict):
        inline = row["inlineResponse"]
        response = inline.get("response")
        error = error or inline.get("error")

    return key, response, error


def parse_generated_text(raw_text: str) -> GeneratedText:
    data = json.loads(raw_text)
    return GeneratedText(**data)


def download_result_file(batch_job: Any) -> str:
    """
    For file-based batch jobs, Google stores results in batch_job.dest.file_name.
    """
    dest = getattr(batch_job, "dest", None)

    file_name = None

    if dest is not None:
        file_name = getattr(dest, "file_name", None)

    if file_name is None and isinstance(batch_job, dict):
        file_name = (
            batch_job.get("dest", {}).get("file_name")
            or batch_job.get("dest", {}).get("fileName")
        )

    if not file_name:
        raise ValueError("Batch job does not contain dest.file_name/fileName.")

    content = client.files.download(file=file_name)

    if isinstance(content, bytes):
        return content.decode("utf-8")

    if hasattr(content, "decode"):
        return content.decode("utf-8")

    return str(content)


def retrieve_batch(args: argparse.Namespace) -> None:
    source_path = input_path(args)

    if args.manifest is not None:
        manifest_path = args.manifest
    else:
        manifest_path = find_latest_manifest(source_path.parent, args.model)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    batch_name = manifest["batch_name"]
    output_path = Path(manifest["output_path"])
    raw_batch_output_path = batch_output_path_for_output(output_path)

    while True:
        batch_job = client.batches.get(name=batch_name)
        state_name = get_state_name(batch_job)

        print(f"Batch {batch_name} state: {state_name}")

        if state_name in COMPLETED_STATES:
            break

        if not args.wait:
            print("Batch is not complete yet. Re-run with --mode retrieve later, or use --wait.")
            return

        time.sleep(args.poll_seconds)

    if state_name in FAILED_STATES:
        manifest["batch_state"] = state_name
        manifest["error"] = to_plain_jsonable(getattr(batch_job, "error", None))
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Google batch ended with state: {state_name}")

    documents = read_jsonl(source_path)

    raw_output = download_result_file(batch_job)
    raw_batch_output_path.write_text(raw_output, encoding="utf-8")

    final_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    seen_indices: set[int] = set()

    for line_number, line in enumerate(raw_output.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            batch_row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "line_number": line_number,
                    "error": f"Invalid JSON in Google result file: {repr(exc)}",
                    "raw_line": line,
                }
            )
            continue

        key, response, error = normalize_google_batch_row(batch_row)

        if key is None:
            errors.append(
                {
                    "line_number": line_number,
                    "error": "Could not find key/custom_id in Google batch row.",
                    "raw_row": batch_row,
                }
            )
            continue

        try:
            document_index = parse_custom_id(key)
        except Exception as exc:
            errors.append(
                {
                    "line_number": line_number,
                    "key": key,
                    "error": repr(exc),
                    "raw_row": batch_row,
                }
            )
            continue

        if document_index < 0 or document_index >= len(documents):
            errors.append(
                {
                    "line_number": line_number,
                    "key": key,
                    "error": f"Document index out of range: {document_index}",
                    "raw_row": batch_row,
                }
            )
            continue

        original = documents[document_index]
        seen_indices.add(document_index)

        if error is not None and response is None:
            errors.append(
                {
                    "key": key,
                    "id": original.get("id"),
                    "error": error,
                    "raw_row": batch_row,
                }
            )
            continue

        if response is None:
            errors.append(
                {
                    "key": key,
                    "id": original.get("id"),
                    "error": "No response found in Google batch row.",
                    "raw_row": batch_row,
                }
            )
            continue

        try:
            response_text = extract_text_from_generate_content_response(response)
            generated = parse_generated_text(response_text)

            final_rows.append(
                {
                    "id": original["id"],
                    "provider": "google",
                    "model": manifest["model"],
                    "language": manifest["language"],
                    "register": original["register"],
                    "prompt_sentence": generated.prompt_sentence,
                    "text": generated.text,
                    "text_sent_amount": original["text_sent_amount"],
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "key": key,
                    "id": original.get("id"),
                    "error": repr(exc),
                    "raw_response": response,
                    "raw_row": batch_row,
                }
            )

    missing_indices = sorted(set(range(len(documents))) - seen_indices)

    for missing_index in missing_indices:
        original = documents[missing_index]
        errors.append(
            {
                "key": f"doc-{missing_index}",
                "id": original.get("id"),
                "error": "No output row found for this input document.",
            }
        )

    write_jsonl(output_path, final_rows)

    manifest["batch_state"] = state_name
    manifest["raw_batch_output_path"] = str(raw_batch_output_path)
    manifest["completed_output_path"] = str(output_path)
    manifest["num_outputs"] = len(final_rows)
    manifest["num_errors"] = len(errors)
    manifest["num_missing_outputs"] = len(missing_indices)
    manifest["retrieved_at_unix"] = time.time()
    manifest["batch_job"] = to_plain_jsonable(batch_job)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if errors:
        error_path = error_path_for_output(output_path)
        write_jsonl(error_path, errors)
        print(f"Wrote {len(errors)} errors to: {error_path}")

    print(f"Wrote {len(final_rows)} regenerated documents to: {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.mode == "submit":
        submit_batch(args)
    elif args.mode == "retrieve":
        retrieve_batch(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
