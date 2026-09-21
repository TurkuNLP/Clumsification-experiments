# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Independent vLLM replicas for candidate-only benchmark inference."""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import tempfile
import uuid
from typing import Any, List

import numpy as np


_worker_scorer = None


def _visible_device_ids(device_count: int, accelerator: str) -> list[str]:
    variables = (
        ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        if accelerator == "rocm" else ("CUDA_VISIBLE_DEVICES",)
    )
    for name in variables:
        value = os.environ.get(name)
        if value:
            ids = [item.strip() for item in value.split(",")]
            if len(ids) == device_count and all(ids):
                return ids
    return [str(index) for index in range(device_count)]


def _device_groups(device_ids: list[str], data_parallel_size: int, tensor_parallel_size: int) -> list[str]:
    if data_parallel_size < 1 or tensor_parallel_size < 1:
        raise ValueError("vLLM data and tensor parallel sizes must be positive")
    required = data_parallel_size * tensor_parallel_size
    if required > len(device_ids):
        raise ValueError(
            f"Requested vLLM DP={data_parallel_size} × TP={tensor_parallel_size} "
            f"needs {required} GPUs, but only {len(device_ids)} are visible"
        )
    return [
        ",".join(device_ids[start:start + tensor_parallel_size])
        for start in range(0, required, tensor_parallel_size)
    ]


def _initialize_worker(
    device_group: str, accelerator: str, scorer_kwargs: dict[str, Any],
    cache_root: str,
) -> None:
    global _worker_scorer
    if accelerator == "rocm":
        os.environ["ROCR_VISIBLE_DEVICES"] = device_group
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_group
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Independent replicas otherwise compile the same model into one vLLM/
    # Inductor cache concurrently, which can leave truncated cache entries.
    os.makedirs(cache_root, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = cache_root
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")

    # Import vLLM only after narrowing the worker's visible devices.
    from .vllm_scorer import VLLMTextScorer

    _worker_scorer = VLLMTextScorer(**scorer_kwargs)


def _score_worker(
    texts: list[str], task_name: str | None, aspect: str,
    batch_size: int, max_length: int,
) -> np.ndarray:
    if _worker_scorer is None:
        raise RuntimeError("vLLM replica was not initialized")
    _worker_scorer.set_prompt_context(task_name, aspect)
    return _worker_scorer.score_texts(texts, batch_size=batch_size, max_length=max_length)


class ParallelVLLMTextScorer:
    """Replicate a scorer across disjoint GPU groups and preserve input order."""

    def __init__(
        self, model_name_or_path: str, *, data_parallel_size: int,
        tensor_parallel_size: int = 1, **scorer_kwargs: Any,
    ) -> None:
        if data_parallel_size < 2:
            raise ValueError("ParallelVLLMTextScorer requires at least two replicas")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.task = scorer_kwargs.get("task")
        self.aspect = scorer_kwargs.get("aspect", "fluency")
        self.protocol = scorer_kwargs.get("protocol", "prometheus_direct_assessment.json")
        self.rubric = scorer_kwargs.get("rubric", "menlo_fluency.json")
        self._scorer_kwargs = {
            "model_name_or_path": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            **scorer_kwargs,
        }
        self._executors: list[ProcessPoolExecutor] = []
        self._cache_run_id = uuid.uuid4().hex
        atexit.register(self.close)

    def _start_workers(self) -> None:
        if self._executors:
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("vLLM data parallel inference requires visible GPUs")
        accelerator = "rocm" if torch.version.hip else "cuda"
        groups = _device_groups(
            _visible_device_ids(torch.cuda.device_count(), accelerator),
            self.data_parallel_size,
            self.tensor_parallel_size,
        )
        print(f"Starting {len(groups)} vLLM replicas (TP={self.tensor_parallel_size}) on {groups}", flush=True)
        base_cache_root = os.environ.get("VLLM_CACHE_ROOT")
        if not base_cache_root:
            base_cache_root = os.path.join(
                os.environ.get("SLURM_TMPDIR")
                or os.environ.get("TMPDIR")
                or tempfile.gettempdir(),
                "vllm",
            )
        context = multiprocessing.get_context("spawn")
        self._executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=context)
            for _ in groups
        ]
        try:
            ready = [
                executor.submit(
                    _initialize_worker, group, accelerator,
                    self._scorer_kwargs,
                    os.path.join(
                        base_cache_root, "eval_replicas", self._cache_run_id,
                        f"replica_{index}",
                    ),
                )
                for index, (executor, group) in enumerate(zip(self._executors, groups))
            ]
            for future in ready:
                future.result()
        except BaseException:
            self.close()
            raise

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task = task_name
        self.aspect = aspect

    def score_cache_context(self) -> tuple[str, str, str]:
        """Match the single-process scorer's prompt identity across dimensions."""
        from clumsification_code.evals.geval.prompts import render_rubric, rubric_for
        from clumsification_code.prompts import load_prompt_data

        rubric_data = load_prompt_data(f"evaluation/rubrics/{self.rubric}")
        rubric = (
            rubric_data["rubric"]
            if "rubric" in rubric_data
            else render_rubric(rubric_for(task=self.task, aspect=self.aspect))
        )
        return (self.protocol, self.rubric, rubric)

    def score_texts(
        self, texts: List[str], device=None, batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        del device
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.asarray([], dtype=np.float32)
        self._start_workers()

        shards: list[list[tuple[int, str]]] = [[] for _ in self._executors]
        for index, candidate in enumerate(texts):
            shards[index % len(shards)].append((index, str(candidate)))

        futures = [
            (
                shard,
                executor.submit(
                    _score_worker, [text for _, text in shard], self.task,
                    self.aspect, batch_size, max_length,
                ),
            )
            for shard, executor in zip(shards, self._executors)
            if shard
        ]
        scores = np.empty(len(texts), dtype=np.float32)
        for shard, future in futures:
            values = np.asarray(future.result(), dtype=np.float32)
            if values.shape != (len(shard),):
                raise RuntimeError("vLLM replica returned the wrong number of scores")
            for (index, _), value in zip(shard, values):
                scores[index] = value
        return scores

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self._executors = []
