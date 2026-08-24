# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Score perturbations in a custom dataset against their original texts.

The public entry point is :func:`score_custom_dataset`; command-line parsing
lives in ``scripts/score_custom_dataset.py``. Score records deliberately only
contain successful scores; originals are self-scored by default. Failures live
in a separate error JSONL file.
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from clumsification_code.data.candidate_identity import (
    candidate_id_from_raw_row,
    canonical_perturbation_source,
    make_original_candidate_id,
)

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

    base_text_id: int
    perturbation_source: str
    candidate_id: str
    source_layer: int
    target_layer: int
    source_text: str
    target_text: str


@dataclass(frozen=True)
class ScoreFailure:
    """A task that could not be evaluated, kept outside the score JSONL."""

    task: ScoreTask
    error_type: str
    error_message: str


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_no}")
            rows.append(row)
    return rows


def _merge_source_rows(
    path: Path,
    new_rows: list[dict],
    *,
    perturbation_source: str,
    overwrite: bool,
) -> list[dict]:
    """Replace one source's rows while preserving the other source."""
    if not path.exists():
        return new_rows

    existing_rows = _read_jsonl(path)
    for row_number, row in enumerate(existing_rows, start=1):
        missing = {"perturbation_source", "candidate_id"} - row.keys()
        if missing:
            raise ValueError(
                f"{path}:{row_number} is not a canonical score record; "
                f"missing {sorted(missing)}. Remove the old score file "
                "before running the new scorer."
            )

    existing_sources = {row["perturbation_source"] for row in existing_rows}
    if perturbation_source in existing_sources and not overwrite:
        raise FileExistsError(
            f"{path} already contains {perturbation_source!r} records. "
            "Pass --overwrite to replace them."
        )

    retained_rows = [
        row
        for row in existing_rows
        if row["perturbation_source"] != perturbation_source
    ]
    return retained_rows + new_rows


def _as_int(value: object, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer {context}, got {value!r}.") from exc


def select_original_ids(
    original_ids: Iterable[int],
    *,
    sample_limit: int | None,
    seed: int,
) -> list[int]:
    """Choose source IDs reproducibly; ``None`` means use all IDs."""
    ids = sorted(set(original_ids))
    if sample_limit is None or sample_limit >= len(ids):
        return ids
    if sample_limit <= 0:
        raise ValueError("sample_limit must be a positive integer when provided.")
    return sorted(random.Random(seed).sample(ids, sample_limit))


def load_score_tasks(
    *,
    dataset_dir: Path,
    layer_directory: str,
    sample_limit: int | None,
    seed: int,
) -> tuple[list[ScoreTask], list[int]]:
    """Load all candidates for a sampled set of original document IDs."""
    perturbation_source = canonical_perturbation_source(layer_directory)
    original_path = dataset_dir / "original.jsonl"
    if not original_path.is_file():
        raise FileNotFoundError(f"Missing original dataset file: {original_path}")

    originals: dict[int, str] = {}
    for row_no, row in enumerate(_read_jsonl(original_path), start=1):
        if "custom_id" not in row or "text" not in row:
            raise ValueError(
                f"{original_path}:{row_no} must contain custom_id and text."
            )
        original_id = _as_int(row["custom_id"], context=f"at {original_path}:{row_no}")
        if original_id in originals:
            raise ValueError(f"Duplicate custom_id={original_id} in {original_path}.")
        if not isinstance(row["text"], str):
            raise ValueError(f"{original_path}:{row_no}: text must be a string.")
        originals[original_id] = row["text"]

    selected_ids = select_original_ids(
        originals,
        sample_limit=sample_limit,
        seed=seed,
    )
    selected_set = set(selected_ids)
    layer_dir = dataset_dir / layer_directory
    if not layer_dir.is_dir():
        raise FileNotFoundError(f"Missing perturbation layer directory: {layer_dir}")

    tasks: list[ScoreTask] = []
    dataset_name = dataset_dir.name

    # Score clean originals by default. Reference-based scorers compare the
    # original with itself; candidate-only scorers score the original directly.
    for original_id in selected_ids:
        tasks.append(
            ScoreTask(
                base_text_id=original_id,
                perturbation_source="original",
                candidate_id=make_original_candidate_id(
                    dataset_name=dataset_name,
                    base_text_id=original_id,
                ),
                source_layer=0,
                target_layer=0,
                source_text=originals[original_id],
                target_text=originals[original_id],
            )
        )

    candidate_indices: dict[tuple[int, int], int] = {}
    for layer_path in sorted(layer_dir.glob("*.jsonl"), key=lambda path: path.name):
        try:
            target_layer = int(layer_path.stem)
        except ValueError:
            continue
        for row_no, row in enumerate(_read_jsonl(layer_path), start=1):
            if "head_id" not in row or "text" not in row:
                raise ValueError(f"{layer_path}:{row_no} must contain head_id and text.")
            original_id = _as_int(row["head_id"], context=f"at {layer_path}:{row_no}")
            if original_id not in originals:
                raise ValueError(
                    f"{layer_path}:{row_no} has head_id={original_id}, which is not "
                    "present in original.jsonl."
                )
            if original_id not in selected_set:
                continue
            if not isinstance(row["text"], str):
                raise ValueError(f"{layer_path}:{row_no}: text must be a string.")
            candidate_key = (original_id, target_layer)
            candidate_index = candidate_indices.get(candidate_key, 0)
            candidate_indices[candidate_key] = candidate_index + 1
            tasks.append(
                ScoreTask(
                    base_text_id=original_id,
                    perturbation_source=perturbation_source,
                    candidate_id=candidate_id_from_raw_row(
                        row,
                        perturbation_source=perturbation_source,
                        base_text_id=original_id,
                        target_layer=target_layer,
                        candidate_index=candidate_index,
                    ),
                    source_layer=0,
                    target_layer=target_layer,
                    source_text=originals[original_id],
                    target_text=row["text"],
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


def _write_jsonl_atomically(path: Path, rows: Iterable[dict], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as writer:
        temporary_path = Path(writer.name)
        for row in rows:
            json.dump(row, writer, ensure_ascii=False, allow_nan=False)
            writer.write("\n")
    temporary_path.replace(path)


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def _write_metadata(path: Path, metadata: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as writer:
        temporary_path = Path(writer.name)
        json.dump(metadata, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")
    temporary_path.replace(path)


def score_custom_dataset(
    *,
    dataset_name: str,
    scoring_type: str,
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
    layer_directory: str = "perturbed_layers",
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

    dataset_dir = dataset_root / dataset_name
    perturbation_source = canonical_perturbation_source(layer_directory)
    tasks, selected_ids = load_score_tasks(
        dataset_dir=dataset_dir,
        layer_directory=layer_directory,
        sample_limit=sample_limit,
        seed=seed,
    )
    score_dir = dataset_dir / "scores"
    score_path = score_dir / f"{scoring_type}.jsonl"
    error_path = score_dir / f"{scoring_type}.errors.jsonl"
    metadata_path = score_dir / f"{scoring_type}.{perturbation_source}.metadata.json"

    if scoring_type == "bertscore_f1":
        scorer = BERTScoreScorer(language=language, batch_size=batch_size).score
        scorer_config = {
            "implementation": "evaluate.load('bertscore')",
            "language": language,
            "uses_metric_defaults": True,
        }
        direction_description = "Raw BERTScore F1; higher is better; no transformation."
    elif scoring_type == "bleurt":
        scorer = BLEURTScorer(checkpoint=bleurt_checkpoint).score
        scorer_config = {
            "implementation": "evaluate.load('bleurt', checkpoint)",
            "checkpoint": bleurt_checkpoint,
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
    score_rows = [
        {
            "base_text_id": task.base_text_id,
            "perturbation_source": task.perturbation_source,
            "candidate_id": task.candidate_id,
            "source_layer": task.source_layer,
            "target_layer": task.target_layer,
            "score_name": scoring_type,
            "score_value": score,
        }
        for task, score in zip(tasks, scores)
        if score is not None
    ]
    error_rows = [
        {
            "base_text_id": failure.task.base_text_id,
            "perturbation_source": failure.task.perturbation_source,
            "candidate_id": failure.task.candidate_id,
            "source_layer": failure.task.source_layer,
            "target_layer": failure.task.target_layer,
            "score_name": scoring_type,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
        }
        for failure in failures
    ]
    _write_jsonl_atomically(
        score_path,
        _merge_source_rows(
            score_path,
            score_rows,
            perturbation_source=perturbation_source,
            overwrite=overwrite,
        ),
        overwrite=True,
    )
    _write_jsonl_atomically(
        error_path,
        _merge_source_rows(
            error_path,
            error_rows,
            perturbation_source=perturbation_source,
            overwrite=overwrite,
        ),
        overwrite=True,
    )
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "input_layer_directory": layer_directory,
        "perturbation_source": perturbation_source,
        "candidate_id_schema": (
            "{source}__base_{base_id}__layer_{target_layer}__candidate_{index}"
        ),
        "scoring_type": scoring_type,
        "score_direction": "higher_is_better",
        "score_transform": direction_description,
        "scored_against": "original_and_candidate",
        "original_self_scores": "written_by_default",
        "num_original_tasks": len(selected_ids),
        "num_perturbation_tasks": len(tasks) - len(selected_ids),
        "failures": "written_to_errors_jsonl; no null or NaN score values are written",
        "sample_limit": sample_limit,
        "seed": seed,
        "selected_original_ids": selected_ids,
        "num_selected_originals": len(selected_ids),
        "num_candidate_tasks": len(tasks),
        "num_successful_scores": len(score_rows),
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
    _write_metadata(metadata_path, metadata, overwrite=overwrite)
    return {
        "score_path": str(score_path),
        "error_path": str(error_path),
        "metadata_path": str(metadata_path),
        "num_successful_scores": len(score_rows),
        "num_failures": len(error_rows),
    }
