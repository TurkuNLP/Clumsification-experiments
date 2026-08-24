# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Regression-specific preparation, training, and evaluation utilities."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import Trainer

from .utils import logger, tensor_debug_summary


def _is_valid_score(value: Any) -> bool:
    """Return True only for finite numeric scalar score values."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def flatten_regression_split(
    dataset: Dataset,
    score_name: str,
    split_name: str,
) -> Tuple[Dataset, Dict[str, int]]:
    """
    Flatten valid scored chain items into independent text regression rows.

    Every text in a chain is considered, including layer 0. An item becomes a
    regression example only when its selected score is a finite numeric value.
    """
    if score_name not in dataset.column_names:
        raise ValueError(
            f"Split {split_name!r} does not contain score field {score_name!r}. "
            f"Available fields: {dataset.column_names}"
        )

    rows: List[Dict[str, Any]] = []
    missing_count = 0
    invalid_count = 0

    for chain in dataset:
        texts = chain["texts"]
        labels = chain["labels"]
        scores = chain[score_name]
        candidate_ids = chain.get("candidate_ids", [])
        perturbation_sources = chain.get("perturbation_sources", [])

        if (
            len(texts) != len(labels)
            or len(texts) != len(scores)
            or len(texts) != len(candidate_ids)
            or len(texts) != len(perturbation_sources)
        ):
            raise ValueError(
                f"Misaligned fields in chain {chain.get('id', '<unknown>')!r}: "
                f"texts={len(texts)}, labels={len(labels)}, "
                f"{score_name}={len(scores)}, candidate_ids={len(candidate_ids)}, "
                f"perturbation_sources={len(perturbation_sources)}."
            )

        for item_index, (text, layer, raw_score) in enumerate(
            zip(texts, labels, scores)
        ):
            if raw_score is None:
                missing_count += 1
                continue

            if not _is_valid_score(raw_score):
                invalid_count += 1
                continue

            rows.append(
                {
                    "text": text,
                    "raw_target": float(raw_score),
                    "chain_id": str(chain["id"]),
                    "dataset_name": chain.get("dataset_name"),
                    "source_original_ids": chain.get("source_original_ids", []),
                    "layer": int(layer),
                    "candidate_id": candidate_ids[item_index],
                    "perturbation_source": perturbation_sources[item_index],
                    "item_index_in_chain": item_index,
                }
            )

    statistics = {
        "accepted": len(rows),
        "skipped_missing": missing_count,
        "skipped_invalid": invalid_count,
    }

    if not rows:
        raise ValueError(
            f"No valid regression examples in split {split_name!r} for "
            f"score field {score_name!r}. Statistics: {statistics}"
        )

    logger.info(
        "Regression split %s for %s: accepted=%d, missing=%d, invalid=%d",
        split_name,
        score_name,
        statistics["accepted"],
        statistics["skipped_missing"],
        statistics["skipped_invalid"],
    )

    return Dataset.from_list(rows), statistics


def build_regression_dataset_dict(
    grouped_dataset_dict: DatasetDict,
    score_name: str,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """
    Flatten shared chain datasets and add min-max scaled targets.

    Training-split min/max values are used for all splits. Development and test
    values are intentionally not clipped, so they may be outside [0, 1].
    """
    flattened: Dict[str, Dataset] = {}
    split_statistics: Dict[str, Dict[str, int]] = {}

    for split_name in ("train", "dev", "test"):
        flattened[split_name], split_statistics[split_name] = (
            flatten_regression_split(
                dataset=grouped_dataset_dict[split_name],
                score_name=score_name,
                split_name=split_name,
            )
        )

    train_values = [float(value) for value in flattened["train"]["raw_target"]]
    train_min = min(train_values)
    train_max = max(train_values)

    if train_min == train_max:
        raise ValueError(
            f"Regression target {score_name!r} is constant in the training "
            f"split ({train_min}). Cannot perform min-max scaling."
        )

    denominator = train_max - train_min

    def add_scaled_target(example: Dict[str, Any]) -> Dict[str, float]:
        return {
            "target": (float(example["raw_target"]) - train_min) / denominator,
        }

    regression_dataset_dict = DatasetDict(
        {
            split_name: split_dataset.map(add_scaled_target)
            for split_name, split_dataset in flattened.items()
        }
    )

    metadata = {
        "training_method": "regression",
        "score_name": score_name,
        "target_scaling": {
            "method": "minmax",
            "fit_split": "train",
            "train_min": train_min,
            "train_max": train_max,
            "clip": False,
        },
        "split_statistics": split_statistics,
    }

    return regression_dataset_dict, metadata


def _pearson(values_a: List[float], values_b: List[float]) -> float:
    """Calculate Pearson correlation without an additional SciPy dependency."""
    if len(values_a) < 2:
        return 0.0

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)

    numerator = sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(values_a, values_b)
    )

    norm_a = math.sqrt(sum((value - mean_a) ** 2 for value in values_a))
    norm_b = math.sqrt(sum((value - mean_b) ** 2 for value in values_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return numerator / (norm_a * norm_b)


def _average_ranks(values: List[float]) -> List[float]:
    """Return one-based average ranks; tied values receive equal ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        end = start + 1

        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1

        average_rank = ((start + 1) + end) / 2.0

        for position in range(start, end):
            ranks[order[position]] = average_rank

        start = end

    return ranks


@torch.no_grad()
def evaluate_regression_distributed(
    model,
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    text_prefix: str,
    train_min: float,
    train_max: float,
) -> Dict[str, float]:
    """
    Evaluate independent regression examples on their original score scale.

    Each distributed worker receives a strided subset of examples. Shorter
    worker subsets are padded with harmless duplicate inputs so every FSDP rank
    executes the same number of forward passes.
    """
    model.eval()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if len(dataset) == 0:
        raise ValueError("Regression evaluation received an empty dataset.")

    device = next(model.parameters()).device

    local_indices = list(range(rank, len(dataset), world_size))
    local_count = len(local_indices)

    local_count_tensor = torch.tensor(
        [local_count],
        dtype=torch.long,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_count_tensor, op=dist.ReduceOp.MAX)

    max_local_count = int(local_count_tensor.item())
    padded_indices = local_indices + [-1] * (max_local_count - local_count)

    model_dtype = next(
        (
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )

    use_autocast = (
        device.type == "cuda"
        and model_dtype in {torch.float16, torch.bfloat16}
    )

    local_pairs: List[Tuple[float, float]] = []
    score_scale = train_max - train_min

    for start in range(0, max_local_count, batch_size):
        indices = padded_indices[start : start + batch_size]
        real_mask = [index >= 0 for index in indices]
        fetch_indices = [index if index >= 0 else 0 for index in indices]

        rows = dataset[fetch_indices]

        tokenized = tokenizer(
            [f"{text_prefix}{text}" for text in rows["text"]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                output = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    group_sizes=None,
                    labels=None,
                )
        else:
            output = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                group_sizes=None,
                labels=None,
            )

        scaled_predictions = output["flat_scores"].float().cpu().tolist()

        for is_real, prediction, raw_target in zip(
            real_mask,
            scaled_predictions,
            rows["raw_target"],
        ):
            if is_real:
                local_pairs.append(
                    (
                        prediction * score_scale + train_min,
                        float(raw_target),
                    )
                )

    if dist.is_available() and dist.is_initialized():
        gathered_pairs = [None] * world_size
        dist.all_gather_object(gathered_pairs, local_pairs)
    else:
        gathered_pairs = [local_pairs]

    if rank == 0:
        pairs = [
            pair
            for rank_pairs in gathered_pairs
            for pair in rank_pairs
        ]

        predictions = [prediction for prediction, _ in pairs]
        targets = [target for _, target in pairs]

        errors = [
            prediction - target
            for prediction, target in pairs
        ]

        metrics = {
            "num_examples": len(pairs),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "rmse": math.sqrt(
                sum(error * error for error in errors) / len(errors)
            ),
            "pearson": _pearson(predictions, targets),
            "spearman": _pearson(
                _average_ranks(predictions),
                _average_ranks(targets),
            ),
        }
    else:
        metrics = None

    if dist.is_available() and dist.is_initialized():
        payload = [metrics]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]

    return metrics


class RegressionFETrainer(Trainer):
    """Hugging Face Trainer for independent Huber regression."""

    def __init__(
        self,
        regression_tokenizer,
        regression_max_length: int,
        text_prefix: str,
        train_min: float,
        train_max: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.regression_tokenizer = regression_tokenizer
        self.regression_max_length = regression_max_length
        self.text_prefix = text_prefix
        self.train_min = train_min
        self.train_max = train_max

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            group_sizes=None,
            labels=inputs["labels"],
        )

        loss = outputs["loss"]

        if not torch.isfinite(loss):
            debug = {
                key: tensor_debug_summary(value)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            }

            raise FloatingPointError(
                f"Non-finite regression loss at step {self.state.global_step}. "
                f"Inputs: {debug}"
            )

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if eval_dataset is None:
            return {}

        metrics = evaluate_regression_distributed(
            model=self.model,
            dataset=eval_dataset,
            tokenizer=self.regression_tokenizer,
            max_length=self.regression_max_length,
            batch_size=max(1, self.args.per_device_eval_batch_size),
            text_prefix=self.text_prefix,
            train_min=self.train_min,
            train_max=self.train_max,
        )

        metrics = {
            f"{metric_key_prefix}_{name}": value
            for name, value in metrics.items()
        }

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )

        return metrics
