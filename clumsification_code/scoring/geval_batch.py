# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Resumable OpenAI Batch scoring for canonical custom-dataset G-Eval runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from clumsification_code.data.io import read_json, read_jsonl, write_json_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import ScoreRecord
from clumsification_code.evals.geval.parser import parse_score_response
from clumsification_code.evals.geval.prompts import (
    GEVAL_QE_PROMPT_VERSION,
    build_messages,
    build_response_format_json_schema,
)


SCORING_METHOD = "geval_gpt54mini_fluency"
MODEL = "gpt-5.4-mini-2026-03-17"
ENDPOINT = "/v1/chat/completions"
MAX_REQUESTS = 50_000
MAX_FILE_BYTES = 200_000_000
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _fingerprint(tasks: Iterable[Any], *, batch_size: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{MODEL}|{GEVAL_QE_PROMPT_VERSION}|full_text|"
        f"max_completion_tokens=256|reasoning_effort=none|{batch_size}".encode()
    )
    for task in tasks:
        digest.update(task.candidate_id.encode())
        digest.update(b"\0")
        digest.update(task.target_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _request(task: Any) -> dict[str, Any]:
    return {
        "custom_id": task.candidate_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": MODEL,
            "messages": build_messages(
                task.target_text,
                max_input_chars=0,
                task="custom_dataset",
                aspect="fluency",
            ),
            "temperature": 0.0,
            "reasoning_effort": "none",
            "max_completion_tokens": 256,
            "response_format": build_response_format_json_schema(),
        },
    }


def _prepare(
    state_dir: Path, tasks: list[Any], *,
    batch_size: int, fingerprint: str, scoring_run_id: str,
) -> dict[str, Any]:
    state_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".batch_prepare_", dir=state_dir.parent) as name:
        temporary_dir = Path(name)
        chunks = []
        for index, offset in enumerate(range(0, len(tasks), batch_size)):
            chunk = tasks[offset : offset + batch_size]
            path = temporary_dir / f"requests_{index:05d}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for task in chunk:
                    handle.write(json.dumps(_request(task), ensure_ascii=False) + "\n")
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ValueError(
                    f"Batch input {index} is {size:,} bytes, above the 200 MB limit; "
                    "lower --geval-batch-size and use a new scoring run ID."
                )
            chunks.append({
                "index": index,
                "request_file": path.name,
                "request_count": len(chunk),
                "request_bytes": size,
                "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_file_id": None,
                "batch_id": None,
                "status": "prepared",
                "output_file_id": None,
                "error_file_id": None,
            })
        state = {
            "schema_version": 1,
            "scoring_run_id": scoring_run_id,
            "fingerprint": fingerprint,
            "model": MODEL,
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "max_input_chars": None,
            "batch_size": batch_size,
            "request_count": len(tasks),
            "chunks": chunks,
        }
        write_json_atomic(temporary_dir / "state.json", state)
        temporary_dir.rename(state_dir)
    return state


def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI Batch scoring requires the openai package") from exc
    return OpenAI()


def _submit(state_dir: Path, state: dict[str, Any], client: Any) -> None:
    for chunk in state["chunks"]:
        if chunk["batch_id"]:
            continue
        if chunk["status"] == "submission_unconfirmed":
            raise RuntimeError(
                f"Submission of chunk {chunk['index']} was interrupted. "
                "Check the OpenAI Batch dashboard for its input file ID before retrying "
                "to avoid submitting the same requests twice."
            )
        if not chunk["input_file_id"]:
            request_path = state_dir / chunk["request_file"]
            if hashlib.sha256(request_path.read_bytes()).hexdigest() != chunk["request_sha256"]:
                raise ValueError(f"Batch request file changed: {request_path}")
            with request_path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="batch")
            chunk["input_file_id"] = uploaded.id
            write_json_atomic(state_dir / "state.json", state, overwrite=True)
        chunk["status"] = "submission_unconfirmed"
        write_json_atomic(state_dir / "state.json", state, overwrite=True)
        try:
            batch = client.batches.create(
                input_file_id=chunk["input_file_id"],
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata={
                    "scoring_run": state_dir.name,
                    "chunk": str(chunk["index"]),
                    "fingerprint": state["fingerprint"][:32],
                },
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                chunk["status"] = "prepared"
                write_json_atomic(state_dir / "state.json", state, overwrite=True)
            raise
        chunk["batch_id"] = batch.id
        chunk["status"] = batch.status
        write_json_atomic(state_dir / "state.json", state, overwrite=True)


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
        write_json_atomic(state_dir / "state.json", state, overwrite=True)
    return all_terminal


def _parse_results(
    state_dir: Path, state: dict[str, Any], tasks: list[Any]
) -> tuple[list[ScoreRecord], list[dict[str, Any]]]:
    by_id = {task.candidate_id: task for task in tasks}
    results: dict[str, tuple[float | None, str | None, str | None]] = {}
    for chunk in state["chunks"]:
        for kind in ("output", "error"):
            path = state_dir / f"{kind}_{chunk['index']:05d}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                candidate_id = row.get("custom_id")
                if candidate_id not in by_id:
                    raise ValueError(f"Unknown Batch custom_id: {candidate_id!r}")
                if candidate_id in results:
                    raise ValueError(f"Duplicate Batch result: {candidate_id!r}")
                try:
                    if row.get("error"):
                        raise ValueError(str(row["error"]))
                    response = row["response"]
                    if response["status_code"] != 200:
                        raise ValueError(
                            f"HTTP {response['status_code']}: {response.get('body')}"
                        )
                    choice = response["body"]["choices"][0]
                    content = choice["message"]["content"]
                    score = parse_score_response(content).score
                    results[candidate_id] = (score, None, None)
                except Exception as exc:
                    results[candidate_id] = (None, type(exc).__name__, str(exc))
    records: list[ScoreRecord] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        score, error_type, error_message = results.get(
            task.candidate_id,
            (None, "MissingBatchResult", "No result in terminal Batch output or error file"),
        )
        if score is not None:
            records.append(ScoreRecord(
                dataset_name=task.dataset_name,
                base_text_id=task.base_text_id,
                candidate_id=task.candidate_id,
                perturbation_method=task.perturbation_method,
                scoring_method=SCORING_METHOD,
                scoring_run_id=state["scoring_run_id"],
                score_value=score,
                source_layer=task.source_layer,
                target_layer=task.target_layer,
                reference_candidate_id=task.reference_candidate_id,
                metadata={"perturbation_run_id": task.perturbation_run_id},
            ))
        else:
            errors.append({
                "schema_version": 1,
                "dataset_name": task.dataset_name,
                "base_text_id": task.base_text_id,
                "candidate_id": task.candidate_id,
                "perturbation_method": task.perturbation_method,
                "perturbation_run_id": task.perturbation_run_id,
                "source_layer": task.source_layer,
                "target_layer": task.target_layer,
                "reference_candidate_id": task.reference_candidate_id,
                "scoring_method": SCORING_METHOD,
                "scoring_run_id": state["scoring_run_id"],
                "error_type": error_type,
                "error_message": error_message,
            })
    return records, errors


def score_geval_batch(
    *,
    repository: DatasetRepository,
    scoring_run_id: str,
    methods: Iterable[str] | None,
    perturbation_run_ids: Iterable[str] | None,
    target_layers: Iterable[int] | None,
    sample_limit: int | None,
    seed: int,
    include_originals: bool,
    reference_policy: str,
    source_partitions: Iterable[str] | None,
    batch_size: int,
    action: str,
    retry_failed: bool = False,
    overwrite_prepared: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Prepare, submit, or collect a reproducible Batch score run."""
    from clumsification_code.scoring.custom_dataset import (
        _load_failed_run,
        _task_fingerprint,
        load_score_tasks,
    )

    if action not in {"prepare", "submit", "collect"}:
        raise ValueError("Batch action must be prepare, submit, or collect")
    if batch_size < 1 or batch_size > MAX_REQUESTS:
        raise ValueError("Batch size must be between 1 and 50,000")
    tasks, selected_ids = load_score_tasks(
        repository=repository,
        methods=methods,
        run_ids=perturbation_run_ids,
        target_layers=target_layers,
        sample_limit=sample_limit,
        seed=seed,
        include_originals=include_originals,
        reference_policy=reference_policy,
        source_partitions=source_partitions,
    )
    full_tasks = tasks
    prior_records: list[ScoreRecord] = []
    prior_metadata: dict[str, Any] = {}
    score_path = repository.score_path(SCORING_METHOD, scoring_run_id)
    error_path = repository.score_error_path(SCORING_METHOD, scoring_run_id)
    metadata_path = repository.score_metadata_path(SCORING_METHOD, scoring_run_id)
    destinations = (score_path, error_path, metadata_path)
    if overwrite_prepared and any(path.exists() for path in destinations):
        raise FileExistsError("Cannot overwrite a completed Batch score run")
    if retry_failed:
        if not all(path.exists() for path in destinations):
            raise FileNotFoundError("--retry-failed requires a completed score run")
        prior_records, tasks, prior_metadata = _load_failed_run(
            score_path=score_path,
            error_path=error_path,
            metadata_path=metadata_path,
            tasks=full_tasks,
        )
        if not tasks:
            return {
                "status": "completed", "score_path": str(score_path),
                "error_path": str(error_path), "metadata_path": str(metadata_path),
                "num_successful_scores": len(prior_records), "num_failures": 0,
            }
        attempt = prior_metadata.get("retry_attempts", 0) + 1
        state_name = f"{scoring_run_id}.retry-{attempt}"
    else:
        state_name = scoring_run_id
        if any(path.exists() for path in destinations):
            raise FileExistsError("Score run already exists; use --retry-failed or a new run ID")
    if not tasks:
        raise ValueError("No candidates selected for Batch scoring")
    state_dir = repository.score_method_root(SCORING_METHOD) / ".batch_runs" / state_name
    state_path = state_dir / "state.json"
    fingerprint = _fingerprint(tasks, batch_size=batch_size)
    if overwrite_prepared and state_path.exists():
        old_state = read_json(state_path)
        if any(c["input_file_id"] or c["batch_id"] for c in old_state["chunks"]):
            raise ValueError("Cannot overwrite Batch files after upload or submission")
        shutil.rmtree(state_dir)
    if state_path.exists():
        state = read_json(state_path)
        if state["fingerprint"] != fingerprint or state["scoring_run_id"] != scoring_run_id:
            raise ValueError("Existing Batch state does not match the selected texts and settings")
    else:
        if state_dir.exists() and any(state_dir.iterdir()):
            raise FileExistsError(f"Incomplete Batch preparation exists: {state_dir}")
        state = _prepare(
            state_dir, tasks, batch_size=batch_size,
            fingerprint=fingerprint, scoring_run_id=scoring_run_id,
        )
    if action != "prepare":
        api = _client(client)
        if action == "submit":
            _submit(state_dir, state, api)
        all_terminal = _collect(state_dir, state, api)
        if all_terminal:
            records, errors = _parse_results(state_dir, state, tasks)
            records = prior_records + records
            metadata = {
                "schema_version": 3,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": repository.dataset_name,
                "scoring_method": SCORING_METHOD,
                "scoring_run_id": scoring_run_id,
                "selected_methods": sorted(methods) if methods is not None else None,
                "selected_perturbation_run_ids": (
                    sorted(perturbation_run_ids) if perturbation_run_ids is not None else None
                ),
                "selected_target_layers": (
                    sorted(target_layers) if target_layers is not None else None
                ),
                "reference_policy": reference_policy,
                "include_originals": include_originals,
                "sample_limit": sample_limit,
                "seed": seed,
                "selected_original_ids": selected_ids,
                "num_selected_originals": len(selected_ids),
                "num_candidate_tasks": len(full_tasks),
                "num_original_tasks": sum(t.target_layer == 0 for t in full_tasks),
                "num_perturbation_tasks": sum(t.target_layer > 0 for t in full_tasks),
                "num_successful_scores": len(records),
                "num_failures": len(errors),
                "score_direction": "higher_is_better",
                "score_transform": "G-Eval 1-5 fluency score; higher is better.",
                "task_fingerprint": _task_fingerprint(full_tasks),
                "teacher_input_mode": "candidate_only",
                "uses_reference": False,
                "scorer_config": {
                    "model_name": MODEL,
                    "protocol": "geval_json.json",
                    "protocol_id": "geval.no_reference",
                    "prompt_version": GEVAL_QE_PROMPT_VERSION,
                    "parser": "json_score",
                    "temperature": 0.0,
                    "n_samples": 1,
                    "max_input_chars": None,
                    "max_output_tokens": 256,
                    "processing": "openai_batch",
                    "batch_size": batch_size,
                    "batch_ids": [c["batch_id"] for c in state["chunks"]],
                },
            }
            if retry_failed:
                metadata["retry_attempts"] = attempt
                metadata["retried_failed_tasks"] = len(tasks)
            repository.write_scores(
                records,
                scoring_method=SCORING_METHOD,
                scoring_run_id=scoring_run_id,
                errors=errors,
                metadata=metadata,
                overwrite=retry_failed,
            )
            return {
                "status": "completed", "score_path": str(score_path),
                "error_path": str(error_path), "metadata_path": str(metadata_path),
                "num_successful_scores": len(records), "num_failures": len(errors),
            }
    return {
        "status": "prepared" if action == "prepare" else "pending",
        "state_path": str(state_path),
        "num_requests": len(tasks),
        "num_batches": len(state["chunks"]),
        "batch_statuses": [c["status"] for c in state["chunks"]],
    }
