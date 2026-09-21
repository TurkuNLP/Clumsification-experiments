# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Score canonical custom-dataset candidates with explicit provenance.

The public entry point is :func:`score_custom_dataset`; command-line parsing
lives in ``scripts/score_custom_dataset.py``. Score records deliberately only
contain successful scores. Failures live in a separate error JSONL file.
"""

from __future__ import annotations

import math
import random
import sys
import time
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from clumsification_code.data.candidate_identity import make_original_candidate_id
from clumsification_code.data.io import (
    append_jsonl_durable,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import ScoreRecord

DEFAULT_PPL_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_BLEURT_CHECKPOINT = "BLEURT-20"
DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
DEFAULT_THEMIS_MODEL = "PKU-ONELab/Themis"
DEFAULT_GEVAL_MODEL = "gpt-5.4-mini-2026-03-17"
SUPPORTED_SCORING_TYPES = frozenset(
    {
        "token_normalized_perplexity",
        "bertscore_f1",
        "bleurt",
        "metricx24_source_qe",
        "gptscore_source_fluency",
        "geval_gpt54mini_fluency",
        "menlo_themis_fluency",
    }
)


@dataclass(frozen=True)
class ScoreTask:
    """One original/candidate comparison to be scored."""

    dataset_name: str
    base_text_id: str
    candidate_id: str
    perturbation_method: str
    perturbation_run_id: str
    source_layer: int
    target_layer: int
    reference_candidate_id: str
    source_text: str
    target_text: str


@dataclass(frozen=True)
class ScoreFailure:
    """A task that could not be evaluated, kept outside the score JSONL."""

    task: ScoreTask
    error_type: str
    error_message: str


def _task_fingerprint(tasks: Sequence[ScoreTask]) -> str:
    """Return a stable identity for an ordered score run."""
    identities = [
        (task.candidate_id, task.reference_candidate_id, task.source_layer, task.target_layer)
        for task in tasks
    ]
    payload = json.dumps(identities, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_progress(
    *,
    score_path: Path,
    error_path: Path,
    metadata_path: Path,
    tasks: Sequence[ScoreTask],
) -> tuple[list[ScoreRecord], list[dict], int]:
    """Load a durable prefix of a matching interrupted run."""
    metadata = read_json(metadata_path)
    fingerprint = _task_fingerprint(tasks)
    if metadata.get("task_fingerprint") != fingerprint:
        raise ValueError(
            "Incomplete score run does not match this request; rerun with --overwrite "
            "to discard its partial results."
        )
    records = [ScoreRecord.from_row(row) for row in read_jsonl(score_path)]
    errors = read_jsonl(error_path)
    completed_ids = [record.candidate_id for record in records] + [
        str(row["candidate_id"]) for row in errors
    ]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("Incomplete score run contains duplicate candidate results")
    expected_ids = [task.candidate_id for task in tasks[: len(completed_ids)]]
    if set(completed_ids) != set(expected_ids):
        raise ValueError("Incomplete score run is not a complete prefix of this request")
    return records, errors, len(completed_ids)


def _load_failed_run(
    *, score_path: Path, error_path: Path, metadata_path: Path, tasks: Sequence[ScoreTask]
) -> tuple[list[ScoreRecord], list[ScoreTask], dict]:
    """Validate a completed run and select exactly its failed candidates."""
    metadata = read_json(metadata_path)
    if metadata.get("task_fingerprint") != _task_fingerprint(tasks):
        raise ValueError("Completed score run does not match this request")
    records = [ScoreRecord.from_row(row) for row in read_jsonl(score_path)]
    errors = read_jsonl(error_path)
    task_ids = {task.candidate_id for task in tasks}
    scored_ids = [record.candidate_id for record in records]
    failed_ids = [str(row["candidate_id"]) for row in errors]
    all_ids = scored_ids + failed_ids
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != task_ids:
        raise ValueError("Completed score run does not cover these tasks exactly once")
    failed = set(failed_ids)
    return records, [task for task in tasks if task.candidate_id in failed], metadata


def select_original_ids(
    original_ids: Iterable[str],
    *,
    sample_limit: int | None,
    seed: int,
) -> list[str]:
    """Choose source IDs reproducibly; ``None`` means use all IDs."""
    ids = sorted(set(original_ids))
    if sample_limit is None or sample_limit >= len(ids):
        return ids
    if sample_limit <= 0:
        raise ValueError("sample_limit must be a positive integer when provided.")
    return sorted(random.Random(seed).sample(ids, sample_limit))


def load_score_tasks(
    *,
    repository: DatasetRepository,
    methods: Iterable[str] | None,
    run_ids: Iterable[str] | None,
    target_layers: Iterable[int] | None,
    sample_limit: int | None,
    seed: int,
    include_originals: bool = True,
    reference_policy: str = "original",
    source_partitions: Iterable[str] | None = None,
) -> tuple[list[ScoreTask], list[str]]:
    """Load manifest-selected canonical candidates and their exact references."""
    if reference_policy not in {"original", "parent"}:
        raise ValueError("reference_policy must be 'original' or 'parent'")
    repository.validate_lineage()
    originals = {record.base_text_id: record for record in repository.read_originals()}

    if source_partitions is not None:
        requested_partitions = set(source_partitions)
        if not requested_partitions:
            raise ValueError("source_partitions must not be empty when supplied")
        partition_by_original = repository.read_split_assignments()
        if partition_by_original is None:
            raise FileNotFoundError(
                "source_partitions requires split_assignments.jsonl"
            )
        originals = {
            base_text_id: record
            for base_text_id, record in originals.items()
            if partition_by_original.get(base_text_id) in requested_partitions
        }
        if not originals:
            raise ValueError("No originals match the requested source_partitions")

    selected_ids = select_original_ids(
        originals,
        sample_limit=sample_limit,
        seed=seed,
    )
    selected = set(selected_ids)
    tasks: list[ScoreTask] = []
    original_candidate_ids = {
        base_id: make_original_candidate_id(
            dataset_name=repository.dataset_name,
            base_text_id=base_id,
        )
        for base_id in selected_ids
    }
    if include_originals:
        for base_id in selected_ids:
            tasks.append(
                ScoreTask(
                    dataset_name=repository.dataset_name,
                    base_text_id=base_id,
                    candidate_id=original_candidate_ids[base_id],
                    perturbation_method="original",
                    perturbation_run_id="original",
                    source_layer=0,
                    target_layer=0,
                    reference_candidate_id=original_candidate_ids[base_id],
                    source_text=originals[base_id].text,
                    target_text=originals[base_id].text,
                )
            )

    entries = repository.list_layers(
        methods=methods,
        run_ids=run_ids,
        target_layers=target_layers,
    )
    if not entries:
        raise FileNotFoundError("No canonical perturbation layers match the score selection")
    candidate_texts: dict[str, str] = {}
    for entry in repository.list_layers():
        for candidate in repository.read_candidates(entry):
            candidate_texts[candidate.candidate_id] = candidate.text
    for entry in entries:
        for candidate in repository.read_candidates(entry):
            if candidate.base_text_id not in selected:
                continue
            if reference_policy == "parent":
                reference_id = candidate.parent_candidate_id
                reference_text = (
                    originals[candidate.base_text_id].text
                    if candidate.source_layer == 0
                    else candidate_texts[reference_id]
                )
                reference_layer = candidate.source_layer
            else:
                reference_id = original_candidate_ids[candidate.base_text_id]
                reference_text = originals[candidate.base_text_id].text
                reference_layer = 0
            tasks.append(
                ScoreTask(
                    dataset_name=repository.dataset_name,
                    base_text_id=candidate.base_text_id,
                    candidate_id=candidate.candidate_id,
                    perturbation_method=candidate.perturbation_method,
                    perturbation_run_id=candidate.run_id,
                    source_layer=reference_layer,
                    target_layer=candidate.target_layer,
                    reference_candidate_id=reference_id,
                    source_text=reference_text,
                    target_text=candidate.text,
                )
            )
    return tasks, selected_ids


class BERTScoreScorer:
    """Thin wrapper around Hugging Face Evaluate's standard BERTScore metric."""

    def __init__(self, *, language: str, batch_size: int) -> None:
        try:
            import evaluate
        except ImportError as exc:
            raise ImportError(
                "bertscore_f1 requires the Hugging Face 'evaluate' package and "
                "its BERTScore dependencies."
            ) from exc

        self.metric = evaluate.load("bertscore")
        self.language = language
        self.batch_size = batch_size

    def score(self, tasks: Sequence[ScoreTask]) -> list[float]:
        """Return Evaluate's F1 values using its normal model/layer defaults."""
        result = self.metric.compute(
            predictions=[task.target_text for task in tasks],
            references=[task.source_text for task in tasks],
            lang=self.language,
            batch_size=self.batch_size,
        )
        scores = [float(score) for score in result["f1"]]
        if len(scores) != len(tasks):
            raise RuntimeError("BERTScore returned a different number of scores than inputs.")
        return scores


class BLEURTScorer:
    """Thin wrapper around Hugging Face Evaluate's BLEURT metric."""

    def __init__(self, *, checkpoint: str) -> None:
        print(
            f"[score_custom_dataset] Loading BLEURT checkpoint {checkpoint!r}...",
            file=sys.stderr,
            flush=True,
        )
        try:
            import evaluate
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "BLEURT requires the Hugging Face 'evaluate' package, BLEURT, "
                "and TensorFlow."
            ) from exc

        gpu_devices = tf.config.list_physical_devices("GPU")
        print(
            f"[score_custom_dataset] TensorFlow {tf.__version__}; visible GPUs: "
            f"{[device.name for device in gpu_devices] or 'none'}.",
            file=sys.stderr,
            flush=True,
        )
        if not gpu_devices:
            raise RuntimeError(
                "BLEURT cannot see a GPU. Load CSC's python-tensorflow module "
                "and ensure the job requests a GPU before scoring."
            )

        # BLEURT-20 is the checkpoint recommended by the BLEURT authors.  Pass
        # it as Evaluate's configuration name so the selected model is explicit
        # and recorded alongside the generated supervision.
        self.metric = evaluate.load("bleurt", checkpoint)
        print(
            "[score_custom_dataset] BLEURT checkpoint loaded.",
            file=sys.stderr,
            flush=True,
        )

    def score(self, tasks: Sequence[ScoreTask]) -> list[float]:
        result = self.metric.compute(
            predictions=[task.target_text for task in tasks],
            references=[task.source_text for task in tasks],
        )
        scores = [float(score) for score in result["scores"]]
        if len(scores) != len(tasks):
            raise RuntimeError("BLEURT returned a different number of scores than inputs.")
        return scores


class TokenNormalizedPerplexityScorer:
    """Store negative mean token NLL (equivalently, ``-log(perplexity)``)."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        max_tokens: int,
        device: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "token_normalized_perplexity requires torch and transformers."
            ) from exc

        self._torch = torch
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_kwargs = {"torch_dtype": "auto"} if self.device.type == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def score(self, tasks: Sequence[ScoreTask]) -> list[float]:
        torch = self._torch
        results: list[float] = []
        for offset in range(0, len(tasks), self.batch_size):
            batch = tasks[offset : offset + self.batch_size]
            encoded = self.tokenizer(
                [task.target_text for task in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            ).to(self.device)
            labels = encoded["input_ids"].masked_fill(encoded["attention_mask"] == 0, -100)
            if labels.shape[1] < 2:
                raise ValueError("Text must contain at least two tokens to score perplexity.")
            with torch.no_grad():
                logits = self.model(**encoded).logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            token_losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                shifted_labels,
                reduction="none",
                ignore_index=-100,
            )
            valid_tokens = (shifted_labels != -100).sum(dim=1)
            if (valid_tokens == 0).any():
                raise ValueError("Text has no scorable tokens after tokenization.")
            mean_nll = (token_losses * (shifted_labels != -100)).sum(dim=1) / valid_tokens
            results.extend((-mean_nll).detach().float().cpu().tolist())
        return [float(value) for value in results]


BatchScorer = Callable[[Sequence[ScoreTask]], list[float]]


def score_with_failure_isolation(
    tasks: Sequence[ScoreTask],
    scorer: BatchScorer,
) -> tuple[list[float | None], list[ScoreFailure]]:
    """Split failed batches until every failed task has its own error record."""
    scores: list[float | None] = [None] * len(tasks)
    failures: list[ScoreFailure] = []

    def score_indices(indices: list[int]) -> None:
        try:
            values = scorer([tasks[index] for index in indices])
            if len(values) != len(indices):
                raise RuntimeError("Scorer returned a different number of values than tasks.")
            for index, value in zip(indices, values):
                if not math.isfinite(value):
                    raise ValueError(f"Scorer returned a non-finite score: {value!r}")
                scores[index] = float(value)
        except Exception as exc:  # preserve one structured record per failed input
            if len(indices) > 1:
                midpoint = len(indices) // 2
                score_indices(indices[:midpoint])
                score_indices(indices[midpoint:])
                return
            task = tasks[indices[0]]
            failures.append(
                ScoreFailure(
                    task=task,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    if tasks:
        score_indices(list(range(len(tasks))))
    return scores, failures


def score_failed_until_success(
    tasks: Sequence[ScoreTask],
    scorer: BatchScorer,
    *,
    max_retries: int,
    score_once: Callable[
        [Sequence[ScoreTask]], tuple[list[float | None], list[ScoreFailure]]
    ] | None = None,
) -> tuple[list[float | None], list[ScoreFailure]]:
    """Retry only unresolved tasks, retaining every successful score."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    scores: list[float | None] = [None] * len(tasks)
    pending = list(range(len(tasks)))
    failures_by_index: dict[int, ScoreFailure] = {}
    for attempt in range(max_retries + 1):
        if not pending:
            break
        print(
            f"[score_custom_dataset] Retry round {attempt + 1}/{max_retries + 1}: "
            f"submitting {len(pending)} task(s).",
            file=sys.stderr,
            flush=True,
        )
        started = time.monotonic()
        attempt_tasks = [tasks[index] for index in pending]
        attempt_scores, attempt_failures = (
            score_once(attempt_tasks)
            if score_once is not None
            else score_with_failure_isolation(attempt_tasks, scorer)
        )
        failed_by_id = {failure.task.candidate_id: failure for failure in attempt_failures}
        next_pending = []
        for index, value in zip(pending, attempt_scores):
            if value is not None:
                scores[index] = value
                failures_by_index.pop(index, None)
            else:
                next_pending.append(index)
                failures_by_index[index] = failed_by_id[tasks[index].candidate_id]
        pending = next_pending
        print(
            f"[score_custom_dataset] Retry round {attempt + 1}/{max_retries + 1}: "
            f"{len(pending)} task(s) still failed after {time.monotonic() - started:.1f}s.",
            file=sys.stderr,
            flush=True,
        )
    return scores, [failures_by_index[index] for index in pending]


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def score_custom_dataset(
    *,
    dataset_name: str,
    scoring_type: str,
    scoring_run_id: str = "default",
    sample_limit: int | None,
    seed: int,
    language: str,
    batch_size: int,
    scoring_chunk_size: int = 1000,
    model_name: str | None,
    bleurt_checkpoint: str = DEFAULT_BLEURT_CHECKPOINT,
    metricx_model_name: str = DEFAULT_METRICX_MODEL,
    metricx_tokenizer_name: str = DEFAULT_METRICX_TOKENIZER,
    metricx_max_input_length: int = 1536,
    gptscore_model_name: str | None = None,
    gptscore_tokenizer_name: str | None = None,
    gptscore_model_type: str = "auto",
    gptscore_source_prompt_template: str | None = None,
    gptscore_device: str | None = None,
    gptscore_device_map: str | None = None,
    gptscore_dtype: str = "auto",
    gptscore_tp_plan: str | None = "auto",
    geval_cache_path: str | None = None,
    geval_batch_size: int = 10000,
    geval_batch_action: str = "prepare",
    geval_client: Any | None = None,
    themis_model_name: str = DEFAULT_THEMIS_MODEL,
    themis_tensor_parallel_size: int = 1,
    themis_max_model_len: int | None = None,
    themis_max_tokens: int = 512,
    themis_gpu_memory_utilization: float = 0.9,
    themis_trust_remote_code: bool = False,
    max_tokens: int = 8192,
    device: str | None = None,
    methods: Iterable[str] | None = None,
    perturbation_run_ids: Iterable[str] | None = None,
    target_layers: Iterable[int] | None = None,
    include_originals: bool = True,
    reference_policy: str = "original",
    source_partitions: Iterable[str] | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
    retry_failed_max_retries: int = 100,
    dataset_root: Path = Path("data/custom_datasets"),
) -> dict:
    """Score originals and perturbations and write reproducibility records."""
    if scoring_type not in SUPPORTED_SCORING_TYPES:
        raise ValueError(f"Unsupported scoring_type: {scoring_type!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if scoring_chunk_size <= 0:
        raise ValueError("scoring_chunk_size must be positive.")
    if themis_tensor_parallel_size <= 0:
        raise ValueError("themis_tensor_parallel_size must be positive.")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2.")
    if metricx_max_input_length < 2:
        raise ValueError("metricx_max_input_length must be at least 2.")
    if overwrite and retry_failed:
        raise ValueError("--overwrite and --retry-failed cannot be used together")
    if retry_failed_max_retries < 0:
        raise ValueError("retry_failed_max_retries must be non-negative")

    methods = tuple(methods) if methods is not None else None
    perturbation_run_ids = (
        tuple(perturbation_run_ids) if perturbation_run_ids is not None else None
    )
    target_layers = tuple(target_layers) if target_layers is not None else None
    repository = DatasetRepository.from_root(dataset_root, dataset_name)
    if scoring_type == "geval_gpt54mini_fluency":
        from clumsification_code.scoring.geval_batch import score_geval_batch

        return score_geval_batch(
            repository=repository,
            scoring_run_id=scoring_run_id,
            methods=methods,
            perturbation_run_ids=perturbation_run_ids,
            target_layers=target_layers,
            sample_limit=sample_limit,
            seed=seed,
            include_originals=include_originals,
            reference_policy=reference_policy,
            source_partitions=source_partitions,
            batch_size=geval_batch_size,
            action=geval_batch_action,
            client=geval_client,
            retry_failed=retry_failed,
            overwrite_prepared=overwrite,
        )
    score_destinations = (
        repository.score_path(scoring_type, scoring_run_id),
        repository.score_error_path(scoring_type, scoring_run_id),
        repository.score_metadata_path(scoring_type, scoring_run_id),
    )
    if retry_failed and not all(path.exists() for path in score_destinations):
        raise FileNotFoundError("--retry-failed requires a completed score run")
    if not overwrite and not retry_failed and any(path.exists() for path in score_destinations):
        existing = next(path for path in score_destinations if path.exists())
        raise FileExistsError(f"Score run output already exists: {existing}")
    progress_root = repository.score_method_root(scoring_type) / ".in_progress"
    progress_score_path = progress_root / f"{scoring_run_id}.jsonl"
    progress_error_path = progress_root / f"{scoring_run_id}.errors.jsonl"
    progress_metadata_path = progress_root / f"{scoring_run_id}.metadata.json"
    progress_paths = (progress_score_path, progress_error_path, progress_metadata_path)
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
    previous_records: list[ScoreRecord] = []
    previous_metadata: dict = {}
    if retry_failed:
        previous_records, tasks, previous_metadata = _load_failed_run(
            score_path=score_destinations[0],
            error_path=score_destinations[1],
            metadata_path=score_destinations[2],
            tasks=full_tasks,
        )
        if not tasks:
            return {
                "score_path": str(score_destinations[0]),
                "error_path": str(score_destinations[1]),
                "metadata_path": str(score_destinations[2]),
                "num_successful_scores": len(previous_records),
                "num_failures": 0,
            }
    print(
        f"[score_custom_dataset] Loaded {len(tasks):,} scoring tasks; "
        f"chunk size is {scoring_chunk_size:,}.",
        file=sys.stderr,
        flush=True,
    )
    resumed_records: list[ScoreRecord] = []
    resumed_errors: list[dict] = []
    resume_offset = 0
    if any(path.exists() for path in progress_paths):
        if overwrite:
            for path in progress_paths:
                path.unlink(missing_ok=True)
        elif not all(path.exists() for path in progress_paths):
            raise FileExistsError(
                "Incomplete score-run checkpoint is missing one or more required files; "
                "rerun with --overwrite to discard it."
            )
        else:
            resumed_records, resumed_errors, resume_offset = _load_progress(
                score_path=progress_score_path,
                error_path=progress_error_path,
                metadata_path=progress_metadata_path,
                tasks=tasks,
            )
    retry_score_once = None
    if scoring_type == "bertscore_f1":
        scorer = BERTScoreScorer(language=language, batch_size=batch_size).score
        scorer_config = {
            "implementation": "evaluate.load('bertscore')",
            "language": language,
            "uses_metric_defaults": True,
            "uses_reference": True,
        }
        direction_description = "Raw BERTScore F1; higher is better; no transformation."
    elif scoring_type == "bleurt":
        scorer = BLEURTScorer(checkpoint=bleurt_checkpoint).score
        scorer_config = {
            "implementation": "evaluate.load('bleurt', checkpoint)",
            "checkpoint": bleurt_checkpoint,
            "uses_reference": True,
        }
        direction_description = "Raw BLEURT score; higher is better; no transformation."
    elif scoring_type == "metricx24_source_qe":
        from clumsification_code.evals.inference.metricx import MetricX24QEInferenceModel

        teacher = MetricX24QEInferenceModel(
            model_name_or_path=metricx_model_name,
            tokenizer_name=metricx_tokenizer_name,
            batch_size=batch_size,
            max_input_length=metricx_max_input_length,
        )

        def scorer(task_batch: Sequence[ScoreTask]) -> list[float]:
            return teacher.score_pairs(
                [task.source_text for task in task_batch],
                [task.target_text for task in task_batch],
            ).tolist()

        scorer_config = {
            "model_name": metricx_model_name,
            "tokenizer_name": metricx_tokenizer_name,
            "input_mode": "source_allowed",
            "uses_reference": False,
            "format": "source: <source> candidate: <candidate>",
            "max_input_length": metricx_max_input_length,
        }
        direction_description = (
            "MetricX-24 raw QE error is lower-is-better; stored value is its "
            "negation, so higher is better."
        )
    elif scoring_type == "menlo_themis_fluency":
        from clumsification_code.evals.inference.vllm_scorer import VLLMTextScorer

        teacher = VLLMTextScorer(
            themis_model_name,
            tensor_parallel_size=themis_tensor_parallel_size,
            max_model_len=themis_max_model_len,
            max_tokens=themis_max_tokens,
            temperature=0.0,
            gpu_memory_utilization=themis_gpu_memory_utilization,
            trust_remote_code=themis_trust_remote_code,
            protocol="themis_direct_assessment.json",
            rubric="menlo_fluency.json",
            task="custom_dataset",
            aspect="fluency",
        )

        def scorer(task_batch: Sequence[ScoreTask]) -> list[float]:
            return teacher.score_texts(
                [task.target_text for task in task_batch],
                batch_size=batch_size,
            ).tolist()

        def retry_score_once(
            task_batch: Sequence[ScoreTask],
        ) -> tuple[list[float | None], list[ScoreFailure]]:
            values, errors = teacher.score_texts_once(
                [task.target_text for task in task_batch]
            )
            if len(values) != len(task_batch) or len(errors) != len(task_batch):
                raise RuntimeError("vLLM returned a different number of results than tasks")
            scores = []
            failures = []
            for task, value, error in zip(task_batch, values, errors):
                if error is None and not math.isfinite(value):
                    error = ValueError(f"Scorer returned a non-finite score: {value!r}")
                if error is None:
                    scores.append(float(value))
                else:
                    scores.append(None)
                    failures.append(ScoreFailure(
                        task=task,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    ))
            return scores, failures

        scorer_config = {
            "model_name": themis_model_name,
            "input_mode": "candidate_only",
            "uses_reference": False,
            "protocol": "themis_direct_assessment.json",
            "protocol_id": "themis.direct_assessment.no_reference",
            "protocol_version": "1",
            "rubric": "menlo_fluency.json",
            "rubric_id": "menlo.fluency",
            "rubric_version": "1",
            "parser": "themis_rating",
            "temperature": 0.0,
            "max_tokens": themis_max_tokens,
        }
        direction_description = "Themis 1-5 score with MENLO fluency rubric; higher is better."
    elif scoring_type == "gptscore_source_fluency":
        from clumsification_code.evals.inference.gptscore import (
            DEFAULT_SOURCE_AWARE_FLUENCY_PROMPT,
            LocalHFGPTScoreInferenceModel,
        )
        import torch

        if not gptscore_model_name:
            raise ValueError(
                "gptscore_source_fluency requires --gptscore-model-name."
            )
        teacher = LocalHFGPTScoreInferenceModel(
            model_name_or_path=gptscore_model_name,
            tokenizer_name_or_path=gptscore_tokenizer_name,
            model_type=gptscore_model_type,
            task_name="custom_dataset",
            aspect="fluency",
            batch_size=batch_size,
            max_input_length=max_tokens,
            dtype=gptscore_dtype,
            device=None if gptscore_device is None else torch.device(gptscore_device),
            device_map=gptscore_device_map,
            tp_plan=gptscore_tp_plan,
            source_prompt_template=(
                gptscore_source_prompt_template
                or DEFAULT_SOURCE_AWARE_FLUENCY_PROMPT
            ),
        )

        def scorer(task_batch: Sequence[ScoreTask]) -> list[float]:
            return teacher.score_pairs(
                [task.source_text for task in task_batch],
                [task.target_text for task in task_batch],
                batch_size=batch_size,
                max_length=max_tokens,
            ).tolist()

        scorer_config = {
            "model_name": gptscore_model_name,
            "tokenizer_name": gptscore_tokenizer_name or gptscore_model_name,
            "model_type": gptscore_model_type,
            "input_mode": "source_allowed",
            "uses_reference": False,
            "source_prompt_template": teacher.source_prompt_template,
            "length_normalization": "mean",
            "max_input_length": max_tokens,
        }
        direction_description = (
            "Negative mean candidate-token NLL conditioned on the source; "
            "higher is better."
        )
    else:
        scorer_config = {"model_name": model_name or DEFAULT_PPL_MODEL}
        local_scorer = TokenNormalizedPerplexityScorer(
            model_name=scorer_config["model_name"],
            batch_size=batch_size,
            max_tokens=max_tokens,
            device=device,
        )
        scorer = local_scorer.score
        direction_description = (
            "Raw token-normalized perplexity is lower-is-better. Stored value is "
            "-log(perplexity), equivalently negative mean token NLL, so higher is better."
        )

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("Chunked scoring requires the 'tqdm' package.") from exc

    print(
        f"[score_custom_dataset] Starting {scoring_type} scoring.",
        file=sys.stderr,
        flush=True,
    )
    score_records: list[ScoreRecord] = resumed_records
    error_rows: list[dict] = resumed_errors
    task_fingerprint = _task_fingerprint(tasks)
    # VLLMTextScorer performs inference in ``batch_size`` groups.  Persisting
    # only after scoring_chunk_size tasks made a large Themis run appear
    # stalled and lost all completed work if the job was pre-empted while the
    # chunk was still running.  Keep the larger chunk for other scorers, but
    # checkpoint Themis at the same granularity as its actual vLLM calls.
    checkpoint_chunk_size = (
        min(scoring_chunk_size, batch_size)
        if scoring_type == "menlo_themis_fluency"
        else scoring_chunk_size
    )
    if checkpoint_chunk_size != scoring_chunk_size:
        print(
            "[score_custom_dataset] Themis checkpointing after each inference "
            f"batch ({checkpoint_chunk_size} tasks).",
            file=sys.stderr,
            flush=True,
        )
    with tqdm(
        total=len(tasks), initial=resume_offset, desc=f"{scoring_type} scoring", unit="pair"
    ) as progress:
        for offset in range(resume_offset, len(tasks), checkpoint_chunk_size):
            task_chunk = tasks[offset : offset + checkpoint_chunk_size]
            if retry_failed:
                scores, failures = score_failed_until_success(
                    task_chunk, scorer, max_retries=retry_failed_max_retries,
                    score_once=retry_score_once,
                )
            else:
                scores, failures = score_with_failure_isolation(task_chunk, scorer)
            record_chunk = [
                ScoreRecord(
                    dataset_name=task.dataset_name,
                    base_text_id=task.base_text_id,
                    candidate_id=task.candidate_id,
                    perturbation_method=task.perturbation_method,
                    scoring_method=scoring_type,
                    scoring_run_id=scoring_run_id,
                    score_value=score,
                    source_layer=task.source_layer,
                    target_layer=task.target_layer,
                    reference_candidate_id=task.reference_candidate_id,
                    metadata={"perturbation_run_id": task.perturbation_run_id},
                )
                for task, score in zip(task_chunk, scores)
                if score is not None
            ]
            error_chunk = [
                {
                    "schema_version": 1,
                    "base_text_id": failure.task.base_text_id,
                    "dataset_name": failure.task.dataset_name,
                    "candidate_id": failure.task.candidate_id,
                    "perturbation_method": failure.task.perturbation_method,
                    "perturbation_run_id": failure.task.perturbation_run_id,
                    "source_layer": failure.task.source_layer,
                    "target_layer": failure.task.target_layer,
                    "reference_candidate_id": failure.task.reference_candidate_id,
                    "scoring_method": scoring_type,
                    "scoring_run_id": scoring_run_id,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                }
                for failure in failures
            ]
            append_jsonl_durable(
                progress_score_path, [record.to_row() for record in record_chunk]
            )
            append_jsonl_durable(progress_error_path, error_chunk)
            score_records.extend(record_chunk)
            error_rows.extend(error_chunk)
            progress.update(len(task_chunk))
            write_json_atomic(
                progress_metadata_path,
                {
                    "status": "in_progress",
                    "dataset_name": dataset_name,
                    "scoring_method": scoring_type,
                    "scoring_run_id": scoring_run_id,
                    "scoring_chunk_size": scoring_chunk_size,
                    "checkpoint_chunk_size": checkpoint_chunk_size,
                    "task_fingerprint": task_fingerprint,
                    "num_candidate_tasks": len(tasks),
                    "num_completed_tasks": min(offset + len(task_chunk), len(tasks)),
                    "num_successful_scores": len(score_records),
                    "num_failures": len(error_rows),
                },
                overwrite=True,
            )
    if retry_failed:
        score_records = previous_records + score_records
    metadata = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "scoring_method": scoring_type,
        "scoring_run_id": scoring_run_id,
        "selected_methods": sorted(methods) if methods is not None else None,
        "selected_perturbation_run_ids": (
            sorted(perturbation_run_ids) if perturbation_run_ids is not None else None
        ),
        "selected_target_layers": (
            sorted(target_layers) if target_layers is not None else None
        ),
        "reference_policy": reference_policy,
        "score_direction": "higher_is_better",
        "score_transform": direction_description,
        "include_originals": include_originals,
        "num_original_tasks": sum(task.target_layer == 0 for task in full_tasks),
        "num_perturbation_tasks": sum(task.target_layer > 0 for task in full_tasks),
        "failures": "written_to_errors_jsonl; no null or NaN score values are written",
        "sample_limit": sample_limit,
        "seed": seed,
        "selected_original_ids": selected_ids,
        "num_selected_originals": len(selected_ids),
        "num_candidate_tasks": len(full_tasks),
        "num_successful_scores": len(score_records),
        "num_failures": len(error_rows),
        "language": language,
        "teacher_input_mode": scorer_config.get("input_mode", "candidate_only"),
        "uses_reference": scorer_config.get("uses_reference", False),
        "batch_size": batch_size,
        "scoring_chunk_size": scoring_chunk_size,
        "checkpoint_chunk_size": checkpoint_chunk_size,
        "task_fingerprint": _task_fingerprint(full_tasks),
        "max_tokens": max_tokens,
        "device": device,
        "scorer_config": scorer_config,
        "package_versions": {
            "python": sys.version.split()[0],
            "evaluate": _package_version("evaluate"),
            "bert-score": _package_version("bert-score"),
            "bleurt": _package_version("BLEURT"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
            "openai": _package_version("openai"),
        },
    }
    if retry_failed:
        metadata["retry_attempts"] = previous_metadata.get("retry_attempts", 0) + 1
        metadata["retried_failed_tasks"] = len(tasks)
        metadata["retry_failed_max_retries"] = retry_failed_max_retries
    score_path, error_path, metadata_path = repository.write_scores(
        score_records,
        scoring_method=scoring_type,
        scoring_run_id=scoring_run_id,
        errors=error_rows,
        metadata=metadata,
        overwrite=overwrite or retry_failed,
    )
    for path in progress_paths:
        path.unlink(missing_ok=True)
    return {
        "score_path": str(score_path),
        "error_path": str(error_path),
        "metadata_path": str(metadata_path),
        "num_successful_scores": len(score_records),
        "num_failures": len(error_rows),
    }
