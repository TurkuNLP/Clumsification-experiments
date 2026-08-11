# Co-created with GPT5.5
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams


LANGUAGE_NAMES = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "es": "Spanish",
    "cs": "Czech",
    "gl": "Galician",
    "is": "Icelandic",
}


CACHE_DIR_NAME = "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate UD documents locally with Qwen3 models using vLLM."
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
        help="Target language code, e.g. en, fi, sv, es.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Qwen3 model checkpoint or local path, e.g. "
            "Qwen/Qwen3-32B or /path/to/Qwen3-checkpoint."
        ),
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "Number of GPUs for tensor parallelism. "
            "Default: torch.cuda.device_count(), or 1 if CUDA is unavailable."
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM dtype, e.g. auto, float16, bfloat16.",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization passed to vLLM.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length for vLLM.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of generated tokens per document.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to vLLM sampling.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing; only process the first N documents.",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )

    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable Qwen3 thinking mode in the chat template. "
            "By default this script disables thinking and asks for JSON only."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional manual batch size. If omitted, all prompts are sent to "
            "vLLM in one call."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional explicit output JSONL path. If omitted, the script creates "
            "the next available <model>_vllm_regens_<N>.jsonl file in the "
            "language folder."
        ),
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Retry only documents listed in the previous errors file instead of "
            "generating all documents from scratch. If --output is omitted, the "
            "latest matching output file for this model/language with an errors "
            "file is used."
        ),
    )

    parser.add_argument(
        "--num-retries",
        type=int,
        default=0,
        help=(
            "Number of additional retry attempts for documents whose model output "
            "cannot be parsed/validated. 0 means no additional retries."
        ),
    )

    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-")


def input_path(args: argparse.Namespace) -> Path:
    return args.base_folder / args.lan / "ud_data.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_language_name(language_code: str) -> str:
    return LANGUAGE_NAMES.get(language_code, language_code)


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
    prefix = f"{model_part}_vllm_regens_"

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
        raw_output = cache_dir / f"{candidate.stem}.raw_outputs.jsonl"
        errors = cache_dir / f"{candidate.stem}.errors.jsonl"

        if (
            not candidate.exists()
            and not manifest.exists()
            and not raw_output.exists()
            and not errors.exists()
        ):
            return candidate

        run_number += 1


def manifest_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".manifest.json")


def raw_output_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".raw_outputs.jsonl")


def error_path_for_output(output_path: Path) -> Path:
    return cache_path_for_output(output_path, ".errors.jsonl")

def latest_retry_output_path(data_dir: Path, model: str) -> Path:
    """
    Find the latest existing output file for this model that has a matching
    errors file in the cache directory.
    """
    model_part = safe_filename_part(model)
    prefix = f"{model_part}_vllm_regens_"

    candidates: list[tuple[int, Path]] = []

    for path in data_dir.glob(f"{prefix}*.jsonl"):
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.jsonl", path.name)
        if not match:
            continue

        error_path = error_path_for_output(path)
        if error_path.exists():
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            f"No previous output file with an errors file found for model "
            f"{model!r} in {data_dir}"
        )

    return max(candidates, key=lambda item: item[0])[1]


def load_retry_items(
    error_path: Path,
    documents: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """
    Load the documents that failed in a previous run.

    Returns:
        List of (original_document_index, document).
    """
    if not error_path.exists():
        raise FileNotFoundError(f"Cannot retry errors; errors file does not exist: {error_path}")

    error_rows = read_jsonl(error_path)

    if not error_rows:
        raise RuntimeError(f"Errors file is empty; nothing to retry: {error_path}")

    retry_indices: list[int] = []
    seen_indices: set[int] = set()

    for error_row in error_rows:
        index: int | None = None

        custom_id = error_row.get("custom_id")
        if isinstance(custom_id, str):
            match = re.fullmatch(r"doc-(\d+)", custom_id)
            if match:
                parsed_index = int(match.group(1))
                if 0 <= parsed_index < len(documents):
                    index = parsed_index

        if index is None:
            error_id = error_row.get("id")
            matching_indices = [
                i for i, document in enumerate(documents)
                if document.get("id") == error_id
            ]

            if len(matching_indices) == 1:
                index = matching_indices[0]
            else:
                raise ValueError(
                    f"Could not uniquely map error row back to source document: {error_row}"
                )

        if index not in seen_indices:
            retry_indices.append(index)
            seen_indices.add(index)

    return [(index, documents[index]) for index in retry_indices]


def get_output_text(output: Any) -> str:
    if not getattr(output, "outputs", None):
        raise ValueError("vLLM output has no candidates.")

    if len(output.outputs) == 0:
        raise ValueError("vLLM output.outputs is empty.")

    text = getattr(output.outputs[0], "text", None)

    if not isinstance(text, str):
        raise TypeError("vLLM output text is not a string.")

    return text


def make_final_row(
    original: dict[str, Any],
    generated: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "id": original["id"],
        "model": args.model,
        "effort": "local-vllm",
        "language": args.lan,
        "register": original["register"],
        "prompt_sentence": generated["prompt_sentence"],
        "text": generated["text"],
        "text_sent_amount": original["text_sent_amount"],
    }


def generate_with_retries(
    llm: LLM,
    items: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate documents and retry only failed parses.

    Args:
        items:
            List of (source_index, original_document).

    Returns:
        final_rows:
            Successfully parsed/generated rows.
        raw_rows:
            Raw model outputs for every attempt that produced text.
        errors:
            Final remaining errors after all retry attempts.
    """
    final_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    pending = items
    final_errors: list[dict[str, Any]] = []

    max_attempts = args.num_retries + 1

    for attempt_index in range(max_attempts):
        if not pending:
            break

        if attempt_index == 0:
            print(f"Generating {len(pending)} documents...")
        else:
            print(
                f"Retry attempt {attempt_index}/{args.num_retries}: "
                f"retrying {len(pending)} failed documents..."
            )

        pending_documents = [document for _, document in pending]
        outputs = run_generation(llm, pending_documents, args)

        if len(outputs) != len(pending):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(pending)} documents."
            )

        next_pending: list[tuple[int, dict[str, Any]]] = []
        final_errors = []

        for (source_index, original), output in zip(pending, outputs):
            raw_text: str | None = None

            try:
                raw_text = get_output_text(output)

                raw_rows.append(
                    {
                        "custom_id": f"doc-{source_index}",
                        "id": original.get("id"),
                        "attempt": attempt_index,
                        "raw_text": raw_text,
                    }
                )

                generated = parse_generated_text(raw_text)

                final_rows.append(make_final_row(original, generated, args))

            except Exception as exc:
                error_row = {
                    "custom_id": f"doc-{source_index}",
                    "id": original.get("id"),
                    "attempt": attempt_index,
                    "error": repr(exc),
                    "raw_output": raw_text,
                }

                final_errors.append(error_row)
                next_pending.append((source_index, original))

        pending = next_pending

    return final_rows, raw_rows, final_errors


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

Return only valid JSON with exactly these two fields:
{{
  "prompt_sentence": "the given reference sentence",
  "text": "the generated text in the target language"
}}

Do not include Markdown.
Do not include explanations.
Do not include any text outside the JSON object.
"""


def make_chat_messages(document: dict[str, Any], language_code: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful multilingual text generation system. "
                "You always follow the requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": make_prompt(document, language_code),
        },
    ]


def strip_thinking_text(text: str) -> str:
    """
    Qwen3 may emit hidden/visible thinking text when thinking mode is enabled.
    This removes common <think>...</think> regions if present.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def strip_markdown_code_fence(text: str) -> str:
    text = text.strip()

    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fence_match:
        return fence_match.group(1).strip()

    return text

def decode_basic_json_escapes(value: str) -> str:
    """
    Decode common JSON-style escapes without requiring the whole string to be
    valid JSON. This is intentionally conservative and only handles simple
    escape sequences.
    """
    def replace_escape(match: re.Match[str]) -> str:
        escape = match.group(1)

        mapping = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }

        if escape in mapping:
            return mapping[escape]

        if escape.startswith("u") and len(escape) == 5:
            try:
                return chr(int(escape[1:], 16))
            except ValueError:
                return "\\" + escape

        return "\\" + escape

    return re.sub(r"\\(u[0-9a-fA-F]{4}|[\"\\/bfnrt])", replace_escape, value)


def clean_jsonish_string_value(value: str) -> str:
    """
    Clean a value that is supposed to be a JSON string but may contain
    unescaped quotes or other minor model mistakes.

    Examples this accepts:

        "hello"
        "hello", 
        "Architecture won't help ..., " he says.
        "If somebody can't handle ..., " Obama mocked.",

    This does not try to be a full JSON parser. It is just a schema-specific
    repair helper.
    """
    value = value.strip()

    # Remove a trailing comma left before the next field.
    if value.endswith(","):
        value = value[:-1].rstrip()

    # Remove one leading quote if present.
    if value.startswith('"'):
        value = value[1:]

    # Remove one trailing quote if present.
    if value.endswith('"'):
        value = value[:-1]

    value = decode_basic_json_escapes(value)

    return value.strip()


def parse_mechanically_fixable_two_field_json(text: str) -> dict[str, Any]:
    """
    Very rudimentary fallback parser for the exact expected schema:

        {
          "prompt_sentence": "...",
          "text": "..."
        }

    It tolerates:
      - unescaped double quotes inside string values
      - a missing comma between prompt_sentence and text
      - extra text before/after the JSON-looking object
      - a missing final closing brace
    """
    text = strip_thinking_text(text)
    text = strip_markdown_code_fence(text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        raise ValueError("No JSON-like object found in model output.")

    # Normal case: {...}
    if end != -1 and end > start:
        body = text[start + 1 : end]
    else:
        # Tolerate truncated JSON where the opening brace exists but the final
        # closing brace is missing.
        body = text[start + 1 :]

    key_pattern = re.compile(
        r'"(?P<key>prompt_sentence|text)"\s*:',
        flags=re.DOTALL,
    )
    matches = list(key_pattern.finditer(body))

    if not matches:
        raise ValueError("No expected fields found in JSON-like object.")

    found_keys = {match.group("key") for match in matches}
    missing = {"prompt_sentence", "text"} - found_keys

    if missing:
        raise ValueError(
            f"JSON-like object is missing required fields: {sorted(missing)}"
        )

    parsed: dict[str, Any] = {}

    for i, match in enumerate(matches):
        key = match.group("key")
        value_start = match.end()

        if i + 1 < len(matches):
            value_end = matches[i + 1].start()
        else:
            value_end = len(body)

        raw_value = body[value_start:value_end]
        parsed[key] = clean_jsonish_string_value(raw_value)

    return {
        "prompt_sentence": parsed["prompt_sentence"],
        "text": parsed["text"],
    }

def extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Tries to parse a JSON object from model output.

    Strategy:
      1. Strict JSON parse of the whole cleaned output.
      2. Strict JSON parse of the first {...} block.
      3. If the object appears truncated with a missing final }, try appending }.
      4. Rudimentary schema-specific repair parser for mechanically fixable
         cases such as unescaped quotes, missing comma, or missing final brace.
    """
    text = strip_thinking_text(text)
    text = strip_markdown_code_fence(text)

    # First try the whole output as valid JSON.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        raise ValueError("No JSON object found in model output.")

    if end != -1 and end > start:
        candidate = text[start : end + 1]

        # Try the extracted object as valid JSON.
        try:
            data = json.loads(candidate)

            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not an object.")

            return data
        except json.JSONDecodeError:
            pass

        # Try the repair parser on the extracted object.
        return parse_mechanically_fixable_two_field_json(candidate)

    # If we get here, there was an opening { but no usable closing }.
    # This happens when the model emits a valid-looking object but truncates the
    # final brace.
    candidate = text[start:].strip()

    # Try the simple repair: append one missing closing brace.
    try:
        data = json.loads(candidate + "}")

        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object.")

        return data
    except json.JSONDecodeError:
        pass

    # Fall back to schema-specific parsing, which can also handle the missing }.
    return parse_mechanically_fixable_two_field_json(candidate)


def parse_generated_text(raw_text: str) -> dict[str, str]:
    data = extract_first_json_object(raw_text)

    missing = {"prompt_sentence", "text"} - set(data)
    if missing:
        raise ValueError(f"Generated JSON is missing required fields: {sorted(missing)}")

    prompt_sentence = data["prompt_sentence"]
    generated_text = data["text"]

    if not isinstance(prompt_sentence, str):
        raise TypeError("Field 'prompt_sentence' must be a string.")

    if not isinstance(generated_text, str):
        raise TypeError("Field 'text' must be a string.")

    prompt_sentence = prompt_sentence.strip()
    generated_text = generated_text.strip()

    if not prompt_sentence:
        raise ValueError("Field 'prompt_sentence' is empty.")

    if not generated_text:
        raise ValueError("Field 'text' is empty.")

    return {
        "prompt_sentence": prompt_sentence,
        "text": generated_text,
    }


def batched(items: list[Any], batch_size: int | None) -> list[list[Any]]:
    if batch_size is None or batch_size <= 0:
        return [items]

    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_llm(args: argparse.Namespace) -> LLM:
    if args.tensor_parallel_size is not None:
        tensor_parallel_size = args.tensor_parallel_size
    else:
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    print(f"Loading model with vLLM: {args.model}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    print(f"max_model_len={args.max_model_len}")
    print(f"dtype={args.dtype}")

    return LLM(
        model=args.model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )


def run_generation(
    llm: LLM,
    documents: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Any]:
    all_outputs: list[Any] = []

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    message_batches = batched(
        [make_chat_messages(document, args.lan) for document in documents],
        args.batch_size,
    )

    total = len(documents)
    done = 0

    for batch_index, messages in enumerate(message_batches):
        print(
            f"Generating batch {batch_index + 1}/{len(message_batches)} "
            f"({len(messages)} documents)..."
        )

        try:
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                chat_template_kwargs={"enable_thinking": args.enable_thinking},
            )
        except TypeError:
            # Older vLLM versions may not accept chat_template_kwargs.
            # In that case, run without it.
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
            )

        all_outputs.extend(outputs)

        done += len(messages)
        print(f"Finished {done}/{total} documents.")

    return all_outputs


def main() -> None:
    args = parse_args()

    if args.num_retries < 0:
        raise ValueError("--num-retries must be >= 0")

    source_path = input_path(args)

    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    documents = read_jsonl(source_path)

    if args.limit is not None:
        documents = documents[: args.limit]

    if not documents:
        print("No input documents found.", file=sys.stderr)
        sys.exit(1)

    data_dir = source_path.parent

    if args.retry_errors:
        if args.output is not None:
            output_path = args.output
        else:
            output_path = latest_retry_output_path(data_dir, args.model)

        if not output_path.exists():
            raise FileNotFoundError(
                f"Cannot retry errors; successful output file does not exist: {output_path}"
            )

        print(f"Retrying errors for existing output: {output_path}")

    else:
        if args.output is not None:
            output_path = args.output
        else:
            output_path = next_output_path(data_dir, args.model)

    raw_output_path = raw_output_path_for_output(output_path)
    error_path = error_path_for_output(output_path)
    manifest_path = manifest_path_for_output(output_path)

    if args.retry_errors:
        existing_final_rows = read_jsonl(output_path) if output_path.exists() else []
        existing_raw_rows = read_jsonl(raw_output_path) if raw_output_path.exists() else []

        retry_items = load_retry_items(error_path, documents)

        existing_success_ids = {row.get("id") for row in existing_final_rows}

        retry_items = [
            (index, document)
            for index, document in retry_items
            if document.get("id") not in existing_success_ids
        ]

        if not retry_items:
            print("All documents from the errors file already appear in the output file.")
            if error_path.exists():
                error_path.unlink()

            manifest = {
                "model": args.model,
                "language": args.lan,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "raw_output_path": str(raw_output_path),
                "error_path": str(error_path),
                "retry_errors": args.retry_errors,
                "num_retries": args.num_retries,
                "num_input_documents": len(documents),
                "num_documents_attempted_this_run": 0,
                "num_new_outputs": 0,
                "num_outputs": len(existing_final_rows),
                "num_errors": 0,
            }

            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"Output remains unchanged: {output_path}")
            print(f"Removed stale errors file: {error_path}")
            print(f"Manifest: {manifest_path}")
            return

        generation_items = retry_items

    else:
        existing_final_rows = []
        existing_raw_rows = []
        generation_items = list(enumerate(documents))

    manifest = {
        "model": args.model,
        "language": args.lan,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "raw_output_path": str(raw_output_path),
        "error_path": str(error_path),
        "retry_errors": args.retry_errors,
        "num_retries": args.num_retries,
        "num_input_documents": len(documents),
        "num_documents_attempted_this_run": len(generation_items),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "batch_size": args.batch_size,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    llm = build_llm(args)

    try:
        new_final_rows, new_raw_rows, errors = generate_with_retries(
            llm=llm,
            items=generation_items,
            args=args,
        )
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_final_rows = existing_final_rows + new_final_rows
    combined_raw_rows = existing_raw_rows + new_raw_rows

    write_jsonl(raw_output_path, combined_raw_rows)
    write_jsonl(output_path, combined_final_rows)

    if errors:
        write_jsonl(error_path, errors)
        print(f"Wrote {len(errors)} remaining errors to: {error_path}")
    else:
        if error_path.exists():
            error_path.unlink()
        print("No remaining errors.")

    manifest["num_new_outputs"] = len(new_final_rows)
    manifest["num_outputs"] = len(combined_final_rows)
    manifest["num_errors"] = len(errors)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote raw model outputs to: {raw_output_path}")
    print(f"Wrote {len(combined_final_rows)} regenerated documents to: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()