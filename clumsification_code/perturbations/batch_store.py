"""Durable LLM batch journal; manifests contain summaries, not failure histories."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import time
import uuid

from clumsification_code.data.io import read_json, write_json_atomic
from clumsification_code.data.schemas import CandidateRecord
from .generation_config import request_config


def fingerprint(request, items):
    settings = request_config(request.persisted_config)
    files = {}
    for key, default in (("assignment_file", None), ("edit_catalog", "data/perturbation_prompts/english/edit_types.jsonl")):
        path = settings.pop(key, default)
        if path:
            files[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    tokenizer_path = Path(settings.get("tokenizer") or "__remote_tokenizer__")
    if tokenizer_path.is_dir():
        for path in sorted(tokenizer_path.iterdir()):
            if path.is_file() and path.suffix in (".json", ".jinja", ".txt"):
                files["tokenizer/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    files["prompt_code"] = hashlib.sha256(Path(__file__).with_name("llm_sampled.py").read_bytes()).hexdigest()
    sources = hashlib.sha256()
    for item in sorted(items, key=lambda item: str(item.candidate_id)):
        sources.update(json.dumps([item.candidate_id, item.text], ensure_ascii=False).encode())
    payload = {"settings": settings, "files": files, "source": sources.hexdigest(),
               "layer": request.layer_kwargs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


class BatchGenerationStore(AbstractContextManager):
    """One writer, immutable batch commits, and atomic final snapshot publication."""

    def __init__(self, repository, request, *, overwrite=False, retry_failed=False):
        if overwrite and retry_failed:
            raise ValueError("retry_failed and overwrite cannot be used together")
        self.repository, self.request = repository, request
        self.overwrite, self.retry_failed = overwrite, retry_failed
        self.root = repository.run_root(request.method, request.run_id) / f"{request.target_layer}.batches"
        self.lock_path = self.root.with_suffix(".lock")
        self.candidate_counts = defaultdict(int)
        self.successes, self.failures, self.attempts = {}, {}, {}
        self.last_entry = None
        self.sequence = 0
        self.context_bucket_counts = Counter()
        self.elapsed_seconds = 0.
        self.generation_seed = request.generation_seed

    def __enter__(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.lock_path.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError("Another process owns this generation run") from exc
        return self

    def __exit__(self, *args):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def select_items(self, items):
        self.input_count = len(items)
        ids = [str(item.candidate_id) for item in items]
        if any(item.candidate_id is None for item in items) or len(set(ids)) != len(ids):
            raise ValueError("Generation inputs must have unique candidate identities")
        self.fingerprint, provenance = fingerprint(self.request, items)
        header = self.root / "request.json"
        if self.overwrite and self.root.exists():
            self.root.rename(self.root.with_name(self.root.name + ".archived-" + uuid.uuid4().hex))
        entries = self.repository.list_layers()
        self.last_entry = next((e for e in entries if e.identity == (self.request.method, self.request.run_id, self.request.target_layer)), None)
        if header.exists():
            if read_json(header)["fingerprint"] != self.fingerprint:
                raise ValueError("Resume request changed immutable source, prompt, catalog, model or generation settings; use a new run ID")
        else:
            if self.retry_failed:
                raise FileNotFoundError("retry_failed requires an existing batch journal; use a new run ID for old-format layers")
            if self.last_entry and not self.overwrite:
                raise ValueError("Existing layer has no batch journal; use a new run ID for the revised prompt")
            versions = {"python": platform.python_version()}
            for package in ("transformers", "torch", "vllm"):
                try: versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError: versions[package] = None
            write_json_atomic(header, {"fingerprint": self.fingerprint, "provenance": provenance, "environment": versions})
        for path in sorted(self.root.glob("batch-*.json")):
            if path.name != f"batch-{self.sequence:08d}.json":
                raise ValueError(f"Missing committed batch before {path.name}; restore the journal before resuming")
            batch = read_json(path)
            if batch["fingerprint"] != self.fingerprint:
                raise ValueError(f"Batch fingerprint mismatch: {path}")
            self.sequence += 1
            self._apply(batch)
        for parent_id, candidate in self.successes.items():
            self.candidate_counts[parent_id] = candidate.candidate_index + 1
        if self.retry_failed:
            self.generation_seed += max(self.attempts.values(), default=0)
        selected = self.failures.keys() if self.retry_failed else set(ids) - self.attempts.keys()
        return [item for item in items if str(item.candidate_id) in selected]

    def _apply(self, batch):
        for row in batch["successes"]:
            candidate = CandidateRecord.from_row(row)
            parent = candidate.parent_candidate_id
            self.successes[parent] = candidate
            self.failures.pop(parent, None)
        for failure in batch["failures"]:
            parent = failure["parent_candidate_id"]
            if parent not in self.successes:
                self.failures[parent] = failure
        self.attempts.update(batch["next_attempts"])
        self.context_bucket_counts.update(batch.get("bucket_counts", {}))
        self.elapsed_seconds += batch.get("elapsed_seconds", 0.)

    def planning_config(self):
        path = self.root / "planning.json"
        if path.exists():
            plan = read_json(path)
            return {k: plan[k] for k in ("source_buckets", "extra_character_tokens")}
        return {}

    def save_plan(self, profile):
        path = self.root / "planning.json"
        if not path.exists():
            write_json_atomic(path, profile)

    def record_context_stats(self, stats):
        self.pending_stats = stats if isinstance(stats, dict) else {}

    def checkpoint(self, items, candidates, failures):
        stats = getattr(self, "pending_stats", {})
        batch = {
            "fingerprint": self.fingerprint,
            "successes": [c.to_row() for c in candidates],
            "failures": list(failures.values()),
            "next_attempts": {str(item.candidate_id): self.attempts.get(str(item.candidate_id), 0)+1 for item in items},
            "bucket_counts": stats.get("bucket_counts", {}),
            "elapsed_seconds": stats.get("elapsed_seconds", 0.),
        }
        # Publication is the commit point; a stale summary cannot undo this batch.
        write_json_atomic(self.root / f"batch-{self.sequence:08d}.json", batch)
        self.sequence += 1
        self._apply(batch)
        self._summary()
        print(f"Checkpoint: attempted={len(self.attempts)}/{self.input_count}, successful={len(self.successes)}, failed={len(self.failures)}", flush=True)

    def _summary(self):
        summary = {
            "input_count": self.input_count, "completed_input_count": len(self.attempts),
            "output_count": len(self.successes), "unresolved_failure_count": len(self.failures),
            "failure_counts": dict(Counter(f["reason"] for f in self.failures.values())),
            "generation_complete": len(self.attempts) == self.input_count,
            "all_outputs_ready": len(self.successes) == self.input_count,
            "committed_batches": self.sequence, "generation_fingerprint": self.fingerprint,
            "bucket_counts": dict(self.context_bucket_counts), "elapsed_seconds": self.elapsed_seconds,
        }
        write_json_atomic(self.root / "summary.json", summary, overwrite=True)
        return summary

    def finish(self):
        summary = self._summary()
        # One compact retry index, rewritten once per submission, outside the manifest.
        write_json_atomic(self.root / "failed.json", list(self.failures.values()), overwrite=True)
        if self.last_entry and not self.overwrite and all(self.last_entry.config.get(k) == v for k,v in summary.items()):
            return self.last_entry
        config = dict(self.request.persisted_config) | summary | {
            "journal_path": str(self.root.relative_to(self.repository.dataset_dir)), "storage": "batch-journal-v1",
        }
        self.last_entry = self.repository.write_candidate_layer(
            list(self.successes.values()), **self.request.layer_kwargs,
            config=config, input_count=self.input_count, overwrite=True, snapshot=True,
        )
        return self.last_entry
