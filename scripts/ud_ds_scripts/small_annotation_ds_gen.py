"""Generate the fixed 50-document annotation sample with one selected model."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LANGUAGE_NAMES = {"fi": "Finnish", "cs": "Czech"}
SAMPLE_IDS = ('w01085004', 'w01106073', 'w01046056', 'w02011012', 'w04006023', 'w01050068', 'w01138045', 'w03001037', 'w01033067', 'n01076006', 'n01029014', 'w04004059', 'n01135002', 'n01134005', 'n02066010', 'n02027007', 'n01043027', 'n01147085', 'n01144041', 'n02040023', 'w01080128', 'n01150051', 'n01011004', 'w01111089', 'n01061041', 'w01027035', 'n01128025', 'w01047094', 'w02019077', 'w01058013', 'n04008016', 'w04010031', 'w01075038', 'n01004009', 'n01116018', 'w01016034', 'n01027041', 'w03007039', 'n01129021', 'n01053008', 'w01075039', 'n01054017', 'w01084102', 'w01028004', 'w01095092', 'w01033022', 'w02004065', 'w02001069', 'w01105053', 'w01075037')
MODELS = ("gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17", "gemini-3.5-flash", "gemini-3.5-flash-lite", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")


def make_prompt(document: dict[str, Any], language_code: str) -> str:
    language_name = LANGUAGE_NAMES.get(language_code, language_code)
    return f"""You are tasked with writing a natural sounding, fluent text given a set of conditions.

Target language: {language_name} ({language_code})

Write the generated text in the target language only.

Task:
Based on the given sentence, write a short continuation of it.
You must write exactly two more sentences that connect to the one you are given.
One of the sentences must include exactly two sub clauses.
The text should be fluent text that flows naturally instead of a collection or listing of sentences.

Given text:
{document["prompt_sentence"]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGE_NAMES, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--base-folder", type=Path, default="data/benchmarks/ud_regens")
    parser.add_argument("--annotation-folder", type=Path, default="annotation_demo/data")
    return parser.parse_args()


def read_documents(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.base_folder / args.language / "ud_data.jsonl"
    documents = {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
    missing = [doc_id for doc_id in SAMPLE_IDS if doc_id not in documents]
    if missing:
        raise ValueError(f"Missing fixed PUD ids in {path}: {missing}")
    return [documents[doc_id] for doc_id in SAMPLE_IDS]


def clean_generated_text(text: str) -> str:
    """Remove reasoning/template artefacts if a backend emits them anyway."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*(?:assistant\s*:\s*)", "", text, flags=re.IGNORECASE)
    return text.strip()


def generate_openai(model: str, documents: list[dict[str, Any]], language: str) -> list[str]:
    import OpenAI_lib as ol
    api = ol.get_client_local()
    results = []
    for document in documents:
        events = api.responses.create(model=model, input=make_prompt(document, language), stream=True, reasoning={"effort": "none"})
        results.append(clean_generated_text("".join(event.delta for event in events if getattr(event, "type", None) == "response.output_text.delta")))
    return results


def generate_gemini(model: str, documents: list[dict[str, Any]], language: str) -> list[str]:
    from OpenAI_lib import get_google_client
    from google.genai import types
    client = get_google_client()
    # Gemini model families differ in which thinking control they accept.
    # Try the explicit disabled setting first, then the Gemini-3 equivalent.
    configs = []
    try:
        configs.append(types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))
    except (TypeError, ValueError):
        pass
    try:
        configs.append(types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="minimal")))
    except (TypeError, ValueError):
        pass
    configs.append(None)
    results = []
    for document in documents:
        prompt = make_prompt(document, language)
        for config_index, config in enumerate(configs):
            try:
                kwargs = {"model": model, "contents": prompt}
                if config is not None:
                    kwargs["config"] = config
                chunks = client.models.generate_content_stream(**kwargs)
                results.append(clean_generated_text("".join(getattr(chunk, "text", None) or "" for chunk in chunks)))
                break
            except Exception as exc:
                # Only retry configuration incompatibilities. Provider errors
                # for the final fallback should remain visible to the caller.
                if config_index == len(configs) - 1 or getattr(exc, "code", None) != 400:
                    raise
    return results


def generate_qwen(model: str, documents: list[dict[str, Any]], language: str) -> list[str]:
    from vllm import LLM, SamplingParams
    llm = LLM(model=model)
    sampling_params = SamplingParams(max_tokens=256, temperature=0.7)
    messages = [[{"role": "user", "content": make_prompt(document, language)}] for document in documents]
    try:
        outputs = llm.chat(messages, sampling_params=sampling_params,
                           chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        # Older vLLM releases lack chat_template_kwargs. The cleanup below
        # still prevents any emitted reasoning block from entering the data.
        outputs = llm.chat(messages, sampling_params=sampling_params)
    texts = []
    for output in outputs:
        if not output.outputs or not isinstance(output.outputs[0].text, str):
            raise ValueError("vLLM returned no text for one of the documents")
        texts.append(clean_generated_text(output.outputs[0].text))
    return texts


def main() -> None:
    args = parse_args()
    documents = read_documents(args)
    if args.model.startswith("gpt-"):
        texts = generate_openai(args.model, documents, args.language)
    elif args.model.startswith("gemini-"):
        texts = generate_gemini(args.model, documents, args.language)
    else:
        texts = generate_qwen(args.model, documents, args.language)
    rows = [{"id": document["id"], "model": args.model, "language": args.language, "register": document.get("register"), "prompt_sentence": document["prompt_sentence"], "text": text} for document, text in zip(documents, texts)]
    out = args.annotation_folder / args.language / f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.model).strip('-')}_small.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
