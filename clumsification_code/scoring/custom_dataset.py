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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from clumsification_code.data.candidate_identity import make_original_candidate_id
from clumsification_code.data.io import write_json_atomic, write_jsonl_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import ScoreRecord

DEFAULT_PPL_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_BLEURT_CHECKPOINT = "BLEURT-20"
DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
SUPPORTED_SCORING_TYPES = frozenset(
    {
        "token_normalized_perplexity",
        "bertscore_f1",
        "bleurt",
        "metricx24_source_qe",
        "gptscore_source_fluency",
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
) -> tuple[list[ScoreTask], list[str]]:
    """Load manifest-selected canonical candidates and their exact references."""
    if reference_policy not in {"original", "parent"}:
        raise ValueError("reference_policy must be 'original' or 'parent'")
    repository.validate_lineage()
    originals = {record.base_text_id: record for record in repository.read_originals()}

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
        try:
            import evaluate
        except ImportError as exc:
            raise ImportError(
                "BLEURT requires the Hugging Face 'evaluate' package, BLEURT, "
                "and TensorFlow."
            ) from exc

        # BLEURT-20 is the checkpoint recommended by the BLEURT authors.  Pass
        # it as Evaluate's configuration name so the selected model is explicit
        # and recorded alongside the generated supervision.
        self.metric = evaluate.load("bleurt", checkpoint)

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
    max_tokens: int = 8192,
    device: str | None = None,
    methods: Iterable[str] | None = None,
    perturbation_run_ids: Iterable[str] | None = None,
    target_layers: Iterable[int] | None = None,
    include_originals: bool = True,
    reference_policy: str = "original",
    overwrite: bool = False,
    dataset_root: Path = Path("data/custom_datasets"),
) -> dict:
    """Score originals and perturbations and write reproducibility records."""
    if scoring_type not in SUPPORTED_SCORING_TYPES:
        raise ValueError(f"Unsupported scoring_type: {scoring_type!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2.")
    if metricx_max_input_length < 2:
        raise ValueError("metricx_max_input_length must be at least 2.")

    methods = tuple(methods) if methods is not None else None
    perturbation_run_ids = (
        tuple(perturbation_run_ids) if perturbation_run_ids is not None else None
    )
    target_layers = tuple(target_layers) if target_layers is not None else None
    repository = DatasetRepository.from_root(dataset_root, dataset_name)
    score_destinations = (
        repository.score_path(scoring_type, scoring_run_id),
        repository.score_error_path(scoring_type, scoring_run_id),
        repository.score_metadata_path(scoring_type, scoring_run_id),
    )
    if not overwrite and any(path.exists() for path in score_destinations):
        existing = next(path for path in score_destinations if path.exists())
        raise FileExistsError(f"Score run output already exists: {existing}")
    tasks, selected_ids = load_score_tasks(
        repository=repository,
        methods=methods,
        run_ids=perturbation_run_ids,
        target_layers=target_layers,
        sample_limit=sample_limit,
        seed=seed,
        include_originals=include_originals,
        reference_policy=reference_policy,
    )

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

    scores, failures = score_with_failure_isolation(tasks, scorer)
    score_records = [
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
        for task, score in zip(tasks, scores)
        if score is not None
    ]
    error_rows = [
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
        "num_original_tasks": sum(task.target_layer == 0 for task in tasks),
        "num_perturbation_tasks": sum(task.target_layer > 0 for task in tasks),
        "failures": "written_to_errors_jsonl; no null or NaN score values are written",
        "sample_limit": sample_limit,
        "seed": seed,
        "selected_original_ids": selected_ids,
        "num_selected_originals": len(selected_ids),
        "num_candidate_tasks": len(tasks),
        "num_successful_scores": len(score_records),
        "num_failures": len(error_rows),
        "language": language,
        "teacher_input_mode": scorer_config.get("input_mode", "candidate_only"),
        "uses_reference": scorer_config.get("uses_reference", False),
        "batch_size": batch_size,
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
        },
    }
    score_path, error_path, metadata_path = repository.write_scores(
        score_records,
        scoring_method=scoring_type,
        scoring_run_id=scoring_run_id,
        errors=error_rows,
        metadata=metadata,
        overwrite=overwrite,
    )
    return {
        "score_path": str(score_path),
        "error_path": str(error_path),
        "metadata_path": str(metadata_path),
        "num_successful_scores": len(score_records),
        "num_failures": len(error_rows),
    }
