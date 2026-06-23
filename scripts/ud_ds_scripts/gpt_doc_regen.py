#Co-created with GPT5.5

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import OpenAI_lib as ol


client = ol.get_client_local()


LANGUAGE_NAMES = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "es": "Spanish",
    "cs": "Czech",
    "gl": "Galician",
    "is": "Islandic"
}


class GeneratedText(BaseModel):
    prompt_sentence: str = Field(
        description="The sentence given as a reference."
    )
    text: str = Field(
        description="The full generated text in the requested target language."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate UD documents using OpenAI batch processing."
    )

    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Benchmark folder containing language-specific UD data folders.",
    )

    parser.add_argument(
        "--lan",
        type=str,
        required=True,
        help="Target language code, e.g. en, fi, sv, de.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenAI model to use.",
    )

    parser.add_argument(
        "--effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort. Use 'none' to omit the reasoning field.",
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

    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-")


def input_path(args: argparse.Namespace) -> Path:
    return args.base_folder / args.lan / "ud_data.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_language_name(language_code: str) -> str:
    return LANGUAGE_NAMES.get(language_code, language_code)


def generated_text_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "generated_text",
        "strict": True,
        "schema": {
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
            "additionalProperties": False,
        },
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


def make_batch_request(
    document: dict[str, Any],
    index: int,
    model: str,
    effort: str,
    language_code: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": make_prompt(document, language_code),
        "text": {
            "format": generated_text_schema(),
        },
    }

    if effort != "none":
        body["reasoning"] = {"effort": effort}

    return {
        "custom_id": f"doc-{index}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


CACHE_DIR_NAME = "cache"


def cache_dir_for_data_dir(data_dir: Path, *, create: bool = True) -> Path:
    cache_dir = data_dir / CACHE_DIR_NAME

    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def cache_path_for_output(output_path: Path, suffix: str) -> Path:
    cache_dir = cache_dir_for_data_dir(output_path.parent)
    return cache_dir / f"{output_path.stem}{suffix}"


def next_output_path(data_dir: Path, model: str, effort: str) -> Path:
    model_part = safe_filename_part(model)
    effort_part = safe_filename_part(effort)

    prefix = f"{model_part}_{effort_part}_regens_"

    cache_dir = cache_dir_for_data_dir(data_dir)

    existing_numbers = []

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


def find_latest_manifest(data_dir: Path, model: str, effort: str) -> Path:
    model_part = safe_filename_part(model)
    effort_part = safe_filename_part(effort)

    prefix = f"{model_part}_{effort_part}_regens_"

    cache_dir = cache_dir_for_data_dir(data_dir, create=False)

    manifests = []

    if cache_dir.exists():
        manifests.extend(cache_dir.glob(f"{prefix}*.manifest.json"))

    # Optional backwards compatibility:
    # allows retrieval of older batches whose manifests were written
    # directly into the language folder before this change.
    manifests.extend(data_dir.glob(f"{prefix}*.manifest.json"))

    manifests = sorted(
        manifests,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        raise FileNotFoundError(
            f"No manifest found in {cache_dir} or {data_dir} "
            f"for model={model}, effort={effort}."
        )

    return manifests[0]


def to_plain_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def submit_batch(args: argparse.Namespace) -> None:
    source_path = input_path(args)

    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    documents = read_jsonl(source_path)

    output_path = next_output_path(source_path.parent, args.model, args.effort)
    batch_input_path = batch_input_path_for_output(output_path)
    manifest_path = manifest_path_for_output(output_path)

    with batch_input_path.open("w", encoding="utf-8") as writer:
        for index, document in enumerate(documents):
            request = make_batch_request(
                document=document,
                index=index,
                model=args.model,
                effort=args.effort,
                language_code=args.lan,
            )
            writer.write(json.dumps(request, ensure_ascii=False) + "\n")

    with batch_input_path.open("rb") as file_handle:
        uploaded_file = client.files.create(
            file=file_handle,
            purpose="batch",
        )

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    manifest = {
        "batch_id": batch.id,
        "batch_status": batch.status,
        "model": args.model,
        "effort": args.effort,
        "language": args.lan,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "batch_input_path": str(batch_input_path),
        "uploaded_file_id": uploaded_file.id,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Submitted batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Manifest: {manifest_path}")
    print(f"Planned output: {output_path}")


def read_openai_file(file_id: str) -> str:
    response = client.files.content(file_id)

    if hasattr(response, "text"):
        return response.text

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)

    if hasattr(response, "read"):
        content = response.read()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)

    raise TypeError("Could not read OpenAI file response.")


def extract_response_text(response_body: dict[str, Any]) -> str:
    if "output_text" in response_body:
        return response_body["output_text"]

    for output_item in response_body.get("output", []):
        for content_item in output_item.get("content", []):
            if content_item.get("type") in {"output_text", "text"}:
                return content_item["text"]

    raise ValueError("Could not find output text in response body.")


def parse_generated_text(raw_text: str) -> GeneratedText:
    data = json.loads(raw_text)
    return GeneratedText(**data)


def parse_custom_id(custom_id: str) -> int:
    match = re.fullmatch(r"doc-(\d+)", custom_id)

    if not match:
        raise ValueError(f"Unexpected custom_id: {custom_id}")

    return int(match.group(1))


def retrieve_batch(args: argparse.Namespace) -> None:
    source_path = input_path(args)

    if args.manifest is not None:
        manifest_path = args.manifest
    else:
        manifest_path = find_latest_manifest(source_path.parent, args.model, args.effort)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    batch_id = manifest["batch_id"]
    output_path = Path(manifest["output_path"])
    raw_batch_output_path = batch_output_path_for_output(output_path)

    while True:
        batch = client.batches.retrieve(batch_id)

        print(f"Batch {batch.id} status: {batch.status}")

        if batch.status == "completed":
            break

        if batch.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Batch ended with status: {batch.status}")

        if not args.wait:
            print("Batch is not complete yet. Re-run with --mode retrieve later.")
            return

        time.sleep(args.poll_seconds)

    documents = read_jsonl(source_path)

    raw_output = read_openai_file(batch.output_file_id)
    raw_batch_output_path.write_text(raw_output, encoding="utf-8")

    final_rows = []
    errors = []

    for line in raw_output.splitlines():
        if not line.strip():
            continue

        batch_row = json.loads(line)
        custom_id = batch_row["custom_id"]
        document_index = parse_custom_id(custom_id)
        original = documents[document_index]

        response = batch_row.get("response")
        error = batch_row.get("error")

        if error is not None:
            errors.append(
                {
                    "custom_id": custom_id,
                    "id": original.get("id"),
                    "error": error,
                }
            )
            continue

        status_code = response.get("status_code")

        if status_code != 200:
            errors.append(
                {
                    "custom_id": custom_id,
                    "id": original.get("id"),
                    "status_code": status_code,
                    "response": response,
                }
            )
            continue

        try:
            response_text = extract_response_text(response["body"])
            generated = parse_generated_text(response_text)

            final_rows.append(
                {
                    "id": original["id"],
                    "model": manifest["model"],
                    "effort": manifest["effort"],
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
                    "custom_id": custom_id,
                    "id": original.get("id"),
                    "error": repr(exc),
                    "raw_response": response,
                }
            )

    write_jsonl(output_path, final_rows)

    manifest["batch_status"] = batch.status
    manifest["raw_batch_output_path"] = str(raw_batch_output_path)
    manifest["completed_output_path"] = str(output_path)
    manifest["num_outputs"] = len(final_rows)
    manifest["num_errors"] = len(errors)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if errors:
        error_path = error_path_for_output(output_path)
        write_jsonl(error_path, errors)
        print(f"Wrote {len(errors)} errors to: {error_path}")

    print(f"Wrote {len(final_rows)} regenerated documents to: {output_path}")


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