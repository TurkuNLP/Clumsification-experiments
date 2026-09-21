# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Resumable OpenAI Batch execution for benchmark G-Eval scoring."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, Optional

import numpy as np

from clumsification_code.data.io import (
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from clumsification_code.evals.geval.parser import parse_score_response
from clumsification_code.evals.geval.prompts import (
    GEVAL_QE_PROMPT_VERSION,
    build_messages,
    build_response_format_json_schema,
)


ENDPOINT = "/v1/chat/completions"
MAX_REQUESTS_PER_FILE = 50_000
MAX_FILE_BYTES = 200_000_000
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BenchmarkBatchConfig:
    model_name: str
    temperature: float = 0.0
    n_samples: int = 1
    max_output_tokens: int = 256
    max_input_chars: int = 12_000
    response_format: str = "json_schema"
    score_min: float = 1.0
    score_max: float = 5.0

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.score_min >= self.score_max:
            raise ValueError("score_min must be smaller than score_max")
        if self.response_format not in {"json_schema", "json_object", "none"}:
            raise ValueError("Unsupported G-Eval response format")


class BatchGEvalScorer:
    """Collect benchmark requests first, then replay collected Batch scores."""

    protocol = "geval_json.json"
    rubric = "geval_no_reference.json"

    def __init__(
        self,
        config: BenchmarkBatchConfig,
        *,
        task: Optional[str] = None,
        aspect: Optional[str] = None,
    ) -> None:
        self.config = config
        self.task = task
        self.aspect = aspect
        self.requests: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.collected_scores: Optional[Dict[str, float]] = None

    @classmethod
    def from_args(cls, args: Any) -> "BatchGEvalScorer":
        return cls(
            BenchmarkBatchConfig(
                model_name=str(args.geval_model),
                temperature=float(args.temperature),
                n_samples=int(args.n_samples),
                max_output_tokens=int(args.max_output_tokens),
                max_input_chars=int(args.max_input_chars),
                response_format=str(args.geval_response_format),
                score_min=float(args.geval_score_min),
                score_max=float(args.geval_score_max),
            ),
            task=getattr(args, "geval_task", None),
            aspect=getattr(args, "geval_aspect", None),
        )

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task = task_name
        self.aspect = aspect

    def _messages(self, text: str) -> list[dict[str, str]]:
        return build_messages(
            text,
            max_input_chars=self.config.max_input_chars,
            task=self.task,
            aspect=self.aspect,
        )

    def score_cache_context(self) -> str:
        return repr(self._messages(""))

    def _logical_key(self, text: str) -> str:
        payload = {
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "config": asdict(self.config),
            "messages": self._messages(text),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _register(self, text: str) -> str:
        key = self._logical_key(text)
        if key not in self.requests:
            self.requests[key] = {
                "logical_key": key,
                "task": self.task,
                "aspect": self.aspect,
                "messages": self._messages(text),
            }
        return key

    def score_texts(
        self,
        texts: Iterable[str],
        device: Any = None,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> np.ndarray:
        del device, batch_size, max_length
        keys = [self._register("" if text is None else str(text)) for text in texts]
        if self.collected_scores is None:
            # The suite runner needs numeric placeholders while it traverses all
            # dimensions. These values are discarded before any result is written.
            return np.full(len(keys), 3.0, dtype=np.float64)
        missing = [key for key in keys if key not in self.collected_scores]
        if missing:
            raise RuntimeError(
                f"Batch output is missing {len(set(missing))} logical G-Eval requests"
            )
        return np.asarray([self.collected_scores[key] for key in keys], dtype=np.float64)

    def set_collected_scores(self, scores: Dict[str, float]) -> None:
        self.collected_scores = {str(key): float(value) for key, value in scores.items()}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "prompt_version": GEVAL_QE_PROMPT_VERSION,
                    "config": asdict(self.config),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for key in self.requests:
            digest.update(key.encode("ascii"))
        return digest.hexdigest()

    def _response_format(self) -> Optional[dict[str, Any]]:
        if self.config.response_format == "json_schema":
            return build_response_format_json_schema()
        if self.config.response_format == "json_object":
            return {"type": "json_object"}
        return None

    def request_rows(self) -> list[dict[str, Any]]:
        rows = []
        response_format = self._response_format()
        for logical_key, request in self.requests.items():
            for sample_index in range(self.config.n_samples):
                request_hash = hashlib.sha256(
                    f"{logical_key}:{sample_index}".encode("ascii")
                ).hexdigest()
                custom_id = f"geval-{request_hash[:58]}"
                body: dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": request["messages"],
                    "temperature": self.config.temperature,
                    "reasoning_effort": "none",
                    "max_completion_tokens": self.config.max_output_tokens,
                }
                if response_format is not None:
                    body["response_format"] = response_format
                rows.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": ENDPOINT,
                        "body": body,
                        "_logical_key": logical_key,
                        "_sample_index": sample_index,
                    }
                )
        return rows


def _public_request(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _state_path(state_dir: Path) -> Path:
    return state_dir / "state.json"


def _validate_run_id(run_id: str) -> str:
    run_id = str(run_id).strip()
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("G-Eval Batch run ID must be one non-empty path component")
    return run_id


def _write_request_file(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_public_request(row), ensure_ascii=False) + "\n")


def _chunk_record(path: Path, index: int, count: int, attempt: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"Batch input {path} is {size:,} bytes, above the 200 MB limit; "
            "use a smaller --geval-batch-size and a new run ID."
        )
    return {
        "index": index,
        "attempt": attempt,
        "request_file": path.name,
        "request_count": count,
        "request_bytes": size,
        "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_file_id": None,
        "batch_id": None,
        "status": "prepared",
        "output_file_id": None,
        "error_file_id": None,
    }


def prepare_benchmark_batch(
    *,
    scorer: BatchGEvalScorer,
    state_root: Path,
    run_id: str,
    batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    if not 1 <= batch_size <= MAX_REQUESTS_PER_FILE:
        raise ValueError("G-Eval Batch size must be between 1 and 50,000")
    run_id = _validate_run_id(run_id)
    state_root = Path(state_root)
    state_dir = state_root / run_id
    fingerprint = scorer.fingerprint()

    if _state_path(state_dir).exists():
        state = read_json(_state_path(state_dir))
        if state.get("fingerprint") != fingerprint:
            raise ValueError(
                "Existing G-Eval Batch state does not match this suite or configuration; "
                "use the original command or a new --geval-batch-run-id."
            )
        return state_dir, state
    if state_dir.exists() and any(state_dir.iterdir()):
        raise FileExistsError(f"Incomplete Batch state directory exists: {state_dir}")

    rows = scorer.request_rows()
    state_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geval_batch_", dir=state_root) as name:
        temporary_dir = Path(name)
        chunks = []
        for index, offset in enumerate(range(0, len(rows), batch_size)):
            chunk_rows = rows[offset : offset + batch_size]
            request_path = temporary_dir / f"requests_{index:05d}.jsonl"
            _write_request_file(request_path, chunk_rows)
            chunks.append(_chunk_record(request_path, index, len(chunk_rows), 0))
        request_map = {
            row["custom_id"]: {
                "logical_key": row["_logical_key"],
                "sample_index": row["_sample_index"],
            }
            for row in rows
        }
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "fingerprint": fingerprint,
            "model": scorer.config.model_name,
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "config": asdict(scorer.config),
            "batch_size": batch_size,
            "logical_request_count": len(scorer.requests),
            "physical_request_count": len(rows),
            "request_map": request_map,
            "chunks": chunks,
            "retry_attempts": 0,
            "results_written": False,
            "results_path": None,
        }
        write_json_atomic(temporary_dir / "state.json", state)
        temporary_dir.rename(state_dir)
    return state_dir, state


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(_state_path(state_dir), state, overwrite=True)


def _client() -> Any:
    """Use the repository's local OpenAI credential helper."""
    from scripts.ud_ds_scripts import OpenAI_lib as ol

    return ol.get_client_local()


def _submit(state_dir: Path, state: dict[str, Any], client: Any) -> bool:
    """Submit every chunk that fits; return whether the active queue filled."""
    for chunk in state["chunks"]:
        if chunk["batch_id"]:
            continue
        if chunk["status"] == "submission_unconfirmed":
            raise RuntimeError(
                f"Submission of chunk {chunk['index']} was interrupted. Check the "
                "OpenAI Batch dashboard for its input file before retrying, to avoid "
                "submitting the same requests twice."
            )
        request_path = state_dir / chunk["request_file"]
        if hashlib.sha256(request_path.read_bytes()).hexdigest() != chunk["request_sha256"]:
            raise ValueError(f"Batch request file changed: {request_path}")
        if not chunk["input_file_id"]:
            with request_path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="batch")
            chunk["input_file_id"] = uploaded.id
            _save_state(state_dir, state)
        chunk["status"] = "submission_unconfirmed"
        _save_state(state_dir, state)
        try:
            batch = client.batches.create(
                input_file_id=chunk["input_file_id"],
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata={
                    "geval_run": state["run_id"],
                    "chunk": str(chunk["index"]),
                    "attempt": str(chunk["attempt"]),
                    "fingerprint": state["fingerprint"][:32],
                },
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                chunk["status"] = "prepared"
                _save_state(state_dir, state)
                return True
            raise
        chunk["batch_id"] = batch.id
        chunk["status"] = batch.status
        _save_state(state_dir, state)
    return False


def _collect(state_dir: Path, state: dict[str, Any], client: Any) -> bool:
    all_terminal = True
    for chunk in state["chunks"]:
        if not chunk["batch_id"]:
            all_terminal = False
            continue
        batch = client.batches.retrieve(chunk["batch_id"])
        chunk["status"] = batch.status
        chunk["output_file_id"] = batch.output_file_id
        chunk["error_file_id"] = batch.error_file_id
        if batch.status not in TERMINAL_STATUSES:
            all_terminal = False
        for kind, file_id in (
            ("output", batch.output_file_id),
            ("error", batch.error_file_id),
        ):
            path = state_dir / f"{kind}_{chunk['index']:05d}.jsonl"
            if file_id and not path.exists():
                content = client.files.content(file_id).text
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(path)
        _save_state(state_dir, state)
    return all_terminal


def _parse_batch_results(
    state_dir: Path,
    state: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], set[str]]:
    successes: dict[str, float] = {}
    errors: dict[str, dict[str, Any]] = {}
    for chunk in state["chunks"]:
        for kind in ("output", "error"):
            path = state_dir / f"{kind}_{chunk['index']:05d}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                custom_id = str(row.get("custom_id", ""))
                if custom_id not in state["request_map"]:
                    raise ValueError(f"Unknown Batch custom_id: {custom_id!r}")
                try:
                    if row.get("error"):
                        raise ValueError(str(row["error"]))
                    response = row["response"]
                    if response["status_code"] != 200:
                        raise ValueError(
                            f"HTTP {response['status_code']}: {response.get('body')}"
                        )
                    content = response["body"]["choices"][0]["message"]["content"]
                    parsed = parse_score_response(
                        content,
                        score_min=float(state["config"]["score_min"]),
                        score_max=float(state["config"]["score_max"]),
                        clamp=True,
                    )
                    successes[custom_id] = parsed.score
                    errors.pop(custom_id, None)
                except Exception as exc:
                    if custom_id not in successes:
                        errors[custom_id] = {
                            "custom_id": custom_id,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }

    samples: dict[str, dict[int, float]] = {}
    failures: list[dict[str, Any]] = []
    failed_custom_ids: set[str] = set()
    for custom_id, mapping in state["request_map"].items():
        logical_key = str(mapping["logical_key"])
        sample_index = int(mapping["sample_index"])
        if custom_id in successes:
            samples.setdefault(logical_key, {})[sample_index] = successes[custom_id]
        else:
            failed_custom_ids.add(custom_id)
            failure = dict(errors.get(custom_id, {}))
            failure.setdefault("custom_id", custom_id)
            failure.setdefault("error_type", "MissingBatchResult")
            failure.setdefault("error_message", "No successful result was downloaded")
            failure.update(logical_key=logical_key, sample_index=sample_index)
            failures.append(failure)

    expected_samples = int(state["config"]["n_samples"])
    logical_scores = {
        key: float(np.mean([values[index] for index in range(expected_samples)]))
        for key, values in samples.items()
        if all(index in values for index in range(expected_samples))
    }
    if any(not math.isfinite(value) for value in logical_scores.values()):
        raise ValueError("Non-finite score found in parsed Batch output")
    write_json_atomic(
        state_dir / "parsed_scores.json", logical_scores, overwrite=True
    )
    write_jsonl_atomic(state_dir / "failures.jsonl", failures, overwrite=True)
    return logical_scores, failures, failed_custom_ids


def _request_rows_by_id(state_dir: Path, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for chunk in state["chunks"]:
        for row in read_jsonl(state_dir / chunk["request_file"]):
            rows.setdefault(str(row["custom_id"]), row)
    return rows


def _append_retry_chunks(
    state_dir: Path,
    state: dict[str, Any],
    failed_custom_ids: set[str],
) -> None:
    if not failed_custom_ids:
        return
    by_id = _request_rows_by_id(state_dir, state)
    retry_rows = [by_id[custom_id] for custom_id in sorted(failed_custom_ids)]
    attempt = int(state.get("retry_attempts", 0)) + 1
    next_index = max((int(chunk["index"]) for chunk in state["chunks"]), default=-1) + 1
    batch_size = int(state["batch_size"])
    for offset in range(0, len(retry_rows), batch_size):
        chunk_rows = retry_rows[offset : offset + batch_size]
        index = next_index + offset // batch_size
        request_path = state_dir / f"retry_{attempt:03d}_{index:05d}.jsonl"
        _write_request_file(request_path, chunk_rows)
        state["chunks"].append(
            _chunk_record(request_path, index, len(chunk_rows), attempt)
        )
    state["retry_attempts"] = attempt
    _save_state(state_dir, state)


def run_benchmark_batch_action(
    *,
    scorer: BatchGEvalScorer,
    state_root: Path,
    run_id: str,
    batch_size: int,
    action: str,
    client: Any = None,
) -> dict[str, Any]:
    """Prepare, submit, collect, or retry a benchmark Batch run."""
    if action not in {"prepare", "submit", "collect", "retry"}:
        raise ValueError("Batch action must be prepare, submit, collect, or retry")
    state_dir, state = prepare_benchmark_batch(
        scorer=scorer,
        state_root=state_root,
        run_id=run_id,
        batch_size=batch_size,
    )
    if action == "prepare":
        return {
            "status": "prepared",
            "state_path": str(_state_path(state_dir)),
            "num_requests": state["physical_request_count"],
            "num_logical_requests": state["logical_request_count"],
            "num_batches": len(state["chunks"]),
        }

    api = client or _client()
    if action in {"submit", "retry"} and any(
        not chunk["batch_id"] for chunk in state["chunks"]
    ):
        queue_limited = _submit(state_dir, state, api)
    else:
        queue_limited = False
    all_terminal = _collect(state_dir, state, api)
    if not all_terminal:
        return {
            "status": "pending",
            "state_path": str(_state_path(state_dir)),
            "batch_statuses": [chunk["status"] for chunk in state["chunks"]],
            "queue_limited": queue_limited,
        }

    scores, failures, failed_custom_ids = _parse_batch_results(state_dir, state)
    if action == "retry" and failures:
        _append_retry_chunks(state_dir, state, failed_custom_ids)
        queue_limited = _submit(state_dir, state, api)
        all_terminal = _collect(state_dir, state, api)
        if not all_terminal:
            return {
                "status": "pending",
                "state_path": str(_state_path(state_dir)),
                "batch_statuses": [chunk["status"] for chunk in state["chunks"]],
                "queue_limited": queue_limited,
            }
        scores, failures, failed_custom_ids = _parse_batch_results(state_dir, state)

    if failures:
        return {
            "status": "completed_with_errors",
            "state_path": str(_state_path(state_dir)),
            "failure_path": str(state_dir / "failures.jsonl"),
            "num_scores": len(scores),
            "num_failures": len(failures),
            "batch_ids": [chunk["batch_id"] for chunk in state["chunks"]],
        }
    return {
        "status": "completed",
        "state_path": str(_state_path(state_dir)),
        "scores": scores,
        "num_scores": len(scores),
        "batch_ids": [chunk["batch_id"] for chunk in state["chunks"]],
        "results_written": bool(state.get("results_written")),
        "results_path": state.get("results_path"),
    }


def mark_benchmark_batch_results_written(
    state_path: Path,
    results_path: Path,
) -> None:
    state_path = Path(state_path)
    state = read_json(state_path)
    state["results_written"] = True
    state["results_path"] = str(results_path)
    write_json_atomic(state_path, state, overwrite=True)


def remove_unsubmitted_benchmark_batch(state_root: Path, run_id: str) -> None:
    """Remove prepared state only when nothing has been uploaded or submitted."""
    state_dir = Path(state_root) / _validate_run_id(run_id)
    state = read_json(_state_path(state_dir))
    if any(chunk["input_file_id"] or chunk["batch_id"] for chunk in state["chunks"]):
        raise ValueError("Cannot remove Batch state after upload or submission")
    shutil.rmtree(state_dir)
