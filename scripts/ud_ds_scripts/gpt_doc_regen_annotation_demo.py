"""Generate small paired PUD samples for the annotation demo.

This is intentionally separate from ``gpt_doc_regen.py``: it samples a fixed
number of documents and writes only demo-sized outputs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

client = None
LANGUAGE_NAMES = {"fi": "Finnish", "cs": "Czech"}
MODELS = ("gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17")
SAMPLE_SIZE = 50
SEED = 20260910


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


def schema() -> dict[str, Any]:
    return {"type": "json_schema", "name": "generated_text", "strict": True,
            "schema": {"type": "object", "properties": {
                "prompt_sentence": {"type": "string"}, "text": {"type": "string"}},
                "required": ["prompt_sentence", "text"], "additionalProperties": False}}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def get_client() -> Any:
    """Load the OpenAI dependency only when submitting or retrieving."""
    global client
    if client is None:
        import OpenAI_lib as ol
        client = ol.get_client_local()
    return client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-folder", type=Path, required=True,
                   help="Folder containing fi/ud_data.jsonl and cs/ud_data.jsonl")
    p.add_argument("--annotation-folder", type=Path, default=Path("annotation_demo/data"))
    p.add_argument("--language", choices=sorted(LANGUAGE_NAMES), action="append")
    p.add_argument("--model", choices=MODELS, action="append")
    p.add_argument("--mode", choices=("generate", "submit", "retrieve"), default="generate")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def choose_documents(path: Path, language: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) < SAMPLE_SIZE:
        raise ValueError(f"{path} contains {len(rows)} rows; need at least {SAMPLE_SIZE}")
    rng = random.Random(f"{SEED}:{language}")
    return rng.sample(rows, SAMPLE_SIZE)


def output_path(args: argparse.Namespace, language: str, model: str) -> Path:
    return args.annotation_folder / language / f"{safe(model)}_demo.jsonl"


def submit(args: argparse.Namespace) -> None:
    api = get_client()
    languages, models = args.language or list(LANGUAGE_NAMES), args.model or list(MODELS)
    for language in languages:
        docs = choose_documents(args.base_folder / language / "ud_data.jsonl", language)
        for model in models:
            out = output_path(args, language, model)
            batch_input = out.with_suffix(".batch_input.jsonl")
            manifest = out.with_suffix(".manifest.json")
            requests = [{"custom_id": f"doc-{i}", "method": "POST", "url": "/v1/responses",
                         "body": {"model": model, "input": make_prompt(doc, language),
                                  "text": {"format": schema()},
                                  "reasoning": {"effort": "none"}}} for i, doc in enumerate(docs)]
            write_jsonl(batch_input, requests)
            with batch_input.open("rb") as fh:
                uploaded = api.files.create(file=fh, purpose="batch")
            batch = api.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h")
            manifest.write_text(json.dumps({"batch_id": batch.id, "batch_status": batch.status,
                "model": model, "language": language, "documents": docs, "output_path": str(out)}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Submitted {language} {model}: {batch.id}\nManifest: {manifest}")


def response_text(body: dict[str, Any]) -> str:
    if "output_text" in body:
        return body["output_text"]
    return next(item["text"] for output in body.get("output", []) for item in output.get("content", []) if item.get("type") in {"output_text", "text"})


def stream_text(api: Any, *, model: str, prompt: str) -> str:
    """Generate one response with streaming and return its accumulated text."""
    events = api.responses.create(model=model, input=prompt, text={"format": schema()},
                                  reasoning={"effort": "none"}, stream=True)
    chunks = []
    for event in events:
        if getattr(event, "type", None) == "response.output_text.delta":
            chunks.append(event.delta)
    return "".join(chunks)


def generate(args: argparse.Namespace) -> None:
    api = get_client()
    languages, models = args.language or list(LANGUAGE_NAMES), args.model or list(MODELS)
    for language in languages:
        docs = choose_documents(args.base_folder / language / "ud_data.jsonl", language)
        for model in models:
            out = output_path(args, language, model)
            rows = []
            for index, original in enumerate(docs, 1):
                generated = json.loads(stream_text(api, model=model, prompt=make_prompt(original, language)))
                rows.append({"id": original["id"], "model": model, "effort": "none",
                             "language": language, "register": original.get("register"),
                             "prompt_sentence": generated["prompt_sentence"], "text": generated["text"],
                             "text_sent_amount": original.get("text_sent_amount")})
                write_jsonl(out, rows)
                print(f"{language} {model}: {index}/{len(docs)}")
            print(f"Wrote {len(rows)} records to {out}")


def retrieve_one(args: argparse.Namespace, manifest_path: Path) -> None:
    api = get_client()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    while True:
        batch = api.batches.retrieve(manifest["batch_id"])
        if batch.status == "completed": break
        if batch.status in {"failed", "expired", "cancelled"}: raise RuntimeError(batch.status)
        if not args.wait: print(f"{manifest_path}: {batch.status} (re-run with --wait)"); return
        time.sleep(args.poll_seconds)
    raw = api.files.content(batch.output_file_id)
    raw_text = raw.text if hasattr(raw, "text") else raw.content.decode("utf-8")
    rows = []
    for line in raw_text.splitlines():
        item = json.loads(line)
        if item.get("error") or item["response"].get("status_code") != 200: continue
        idx = int(item["custom_id"].split("-")[-1]); original = manifest["documents"][idx]
        generated = json.loads(response_text(item["response"]["body"]))
        rows.append({"id": original["id"], "model": manifest["model"], "effort": "none",
                     "language": manifest["language"], "register": original.get("register"),
                     "prompt_sentence": generated["prompt_sentence"], "text": generated["text"],
                     "text_sent_amount": original.get("text_sent_amount")})
    out = Path(manifest["output_path"]); write_jsonl(out, rows)
    manifest["batch_status"], manifest["num_outputs"] = "completed", len(rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} records to {out}")


def main() -> None:
    args = parse_args()
    if args.mode == "generate": generate(args); return
    if args.mode == "submit": submit(args); return
    manifests = [args.manifest] if args.manifest else list(args.annotation_folder.glob("*/gpt-5.4-*_demo.manifest.json"))
    if not manifests: raise FileNotFoundError("No demo batch manifests found")
    for manifest in manifests: retrieve_one(args, manifest)


if __name__ == "__main__": main()
