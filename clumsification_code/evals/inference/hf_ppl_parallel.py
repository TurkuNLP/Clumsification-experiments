# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Independent Hugging Face PPL replicas on separate GPUs."""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from typing import Any, List

import numpy as np


_worker_scorer = None


def _visible_device_ids(device_count: int, accelerator: str) -> list[str]:
    names = (
        ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        if accelerator == "rocm" else ("CUDA_VISIBLE_DEVICES",)
    )
    for name in names:
        value = os.environ.get(name)
        if value:
            ids = [item.strip() for item in value.split(",")]
            if len(ids) == device_count and all(ids):
                return ids
    return [str(index) for index in range(device_count)]


def _initialize_worker(device_id: str, accelerator: str, scorer_kwargs: dict[str, Any]) -> None:
    global _worker_scorer
    if accelerator == "rocm":
        os.environ["ROCR_VISIBLE_DEVICES"] = device_id
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    import torch
    from .hf_ppl import HFCausalLMPerplexityInferenceModel

    _worker_scorer = HFCausalLMPerplexityInferenceModel(
        **scorer_kwargs,
        device=torch.device("cuda:0"),
        show_progress=False,
    )


def _score_worker(texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    if _worker_scorer is None:
        raise RuntimeError("PPL replica was not initialized")
    return _worker_scorer.score_texts(texts, batch_size=batch_size, max_length=max_length)


class ParallelHFPPLScorer:
    """Score disjoint text shards on one model replica per GPU."""

    def __init__(self, *, data_parallel_size: int, **scorer_kwargs: Any) -> None:
        if data_parallel_size < 2:
            raise ValueError("ParallelHFPPLScorer requires at least two replicas")
        if scorer_kwargs.get("device_map"):
            raise ValueError("--device-map cannot be combined with PPL data parallelism")
        self.data_parallel_size = data_parallel_size
        self._scorer_kwargs = scorer_kwargs
        self._executors: list[ProcessPoolExecutor] = []
        atexit.register(self.close)

    def _start_workers(self) -> None:
        if self._executors:
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PPL data parallel inference requires visible GPUs")
        count = torch.cuda.device_count()
        if self.data_parallel_size > count:
            raise ValueError(
                f"Requested {self.data_parallel_size} PPL replicas, but only {count} GPUs are visible"
            )
        accelerator = "rocm" if torch.version.hip else "cuda"
        device_ids = _visible_device_ids(count, accelerator)[:self.data_parallel_size]
        print(f"Starting {len(device_ids)} PPL replicas on GPUs {device_ids}", flush=True)
        context = multiprocessing.get_context("spawn")
        self._executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=context)
            for _ in device_ids
        ]
        try:
            futures = [
                executor.submit(_initialize_worker, device_id, accelerator, self._scorer_kwargs)
                for executor, device_id in zip(self._executors, device_ids)
            ]
            for future in futures:
                future.result()
        except BaseException:
            self.close()
            raise

    def score_texts(
        self, texts: List[str], device=None, batch_size: int = 32, max_length: int = 512,
    ) -> np.ndarray:
        del device
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.asarray([], dtype=np.float32)
        self._start_workers()

        shards: list[list[tuple[int, str]]] = [[] for _ in self._executors]
        for index, text in enumerate(texts):
            shards[index % len(shards)].append((index, str(text)))
        futures = [
            (shard, executor.submit(_score_worker, [text for _, text in shard], batch_size, max_length))
            for shard, executor in zip(shards, self._executors)
            if shard
        ]
        scores = np.empty(len(texts), dtype=np.float32)
        for shard, future in futures:
            values = np.asarray(future.result(), dtype=np.float32)
            if values.shape != (len(shard),):
                raise RuntimeError("PPL replica returned the wrong number of scores")
            for (index, _), value in zip(shard, values):
                scores[index] = value
        if not np.isfinite(scores).all():
            raise RuntimeError("PPL replica returned non-finite scores")
        return scores

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self._executors = []
