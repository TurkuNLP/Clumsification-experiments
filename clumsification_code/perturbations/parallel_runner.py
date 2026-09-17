"""Independent model replicas with disjoint device groups and a single caller/writer."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import json
import math
import multiprocessing
import os
import time
from itertools import chain, groupby

from .vllm_runner import VLLMRunner, _key
from .schemas import GenerationRuntime
from .length_planning import GenerationBatch

_worker = None


def _initialize(group, accelerator):
    global _worker
    if accelerator == "rocm":
        os.environ["ROCR_VISIBLE_DEVICES"] = group
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = group
    _worker = VLLMRunner()
    from multiprocessing.util import Finalize
    Finalize(None, _worker.close, exitpriority=10)


def _execute(model, prompts, temperature, max_tokens, kwargs, budgets):
    _worker._budgets = budgets
    _worker._plan_key = json.dumps({k: v for k, v in kwargs["config"].items()
                                   if k != "engine_max_model_len"}, sort_keys=True)
    # A tokenizer is needed only to interpret reasoning boundaries/diagnostics.
    if _worker.tokenizer is None:
        from .length_planning import load_tokenizer
        _worker.tokenizer = load_tokenizer(model, kwargs["config"])
    outputs = _worker(model, prompts, temperature, max_tokens, **kwargs)
    return outputs, _worker.last_context_stats


class ParallelLLMRunner(VLLMRunner):
    """Explicit replicas: e.g. four engines, each using a topology-checked GPU pair."""

    def __init__(self, groups, *, accelerator="cuda"):
        super().__init__()
        self.groups = [group.strip() for group in groups.split(";")]
        ids = [device.strip() for group in self.groups for device in group.split(",")]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("Device groups must be nonempty and disjoint")
        if len({len(group.split(',')) for group in self.groups}) != 1:
            raise ValueError("All replicas must have the same tensor-parallel size")
        self.accelerator, self.executors = accelerator, []

    def _start_workers(self):
        if not self.executors:
            self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                                  initializer=_initialize, initargs=(group,self.accelerator))
                              for group in self.groups]

    def generate_batches(self, batches, *, adapter, runtime, batch_size):
        """Keep one chunk in flight per replica, yielding completed chunks to the writer.

        Coordinator threads only wait on their dedicated process. Model execution
        remains isolated in those processes; validation and commits stay in the
        calling thread. Refill a free replica before yielding to disk I/O.
        Drain and commit the current bucket before submitting the next bucket.
        """
        chunk_size = max(1, math.ceil(batch_size / len(self.groups)))
        chunks = (
            (getattr(batch, "bucket_key", None), GenerationBatch(
                batch[start:start + chunk_size], bucket_key=getattr(batch, "bucket_key", None),
                context_limit=getattr(batch, "context_limit", None)))
            for batch in batches
            for start in range(0, len(batch), chunk_size)
        )
        # Consume the first chunk before starting threads: this also completes
        # global token planning and populates the adapter's prompt cache.
        first = next(chunks, None)
        if first is None:
            return
        self._start_workers()

        def execute(rank, batch):
            stats = {}

            def chat(model, prompts, temperature, max_tokens, **kwargs):
                kwargs = dict(kwargs)
                kwargs["config"] = kwargs["config"] | {
                    "tensor_parallel_size": len(self.groups[rank].split(","))
                }
                if batch.context_limit is not None:
                    kwargs["config"]["engine_max_model_len"] = batch.context_limit
                values, measured = self.executors[rank].submit(
                    _execute, model, prompts, temperature, max_tokens, kwargs,
                    {_key(p): self._budgets[_key(p)] for p in prompts},
                ).result()
                stats.update(measured)
                return values

            chat.supports_generation_config = True
            local_runtime = GenerationRuntime(chat_runner=chat, attempts=runtime.attempts)
            return list(adapter.generate(batch, local_runtime)), stats

        last_completion = time.monotonic()
        first_bucket = True
        with ThreadPoolExecutor(max_workers=len(self.groups)) as coordinators:
            for bucket_key, bucket in groupby(chain([first], chunks), key=lambda entry: entry[0]):
                if not first_bucket:
                    self.close()
                    self._start_workers()
                first_bucket = False
                bucket_chunks = (entry[1] for entry in bucket)
                pending = {}
                if bucket_key is not None:
                    print(f"Starting source bucket: {bucket_key[1]} tokens"
                          + (" (over context)" if bucket_key[0] else ""), flush=True)
                try:
                    for rank in range(len(self.groups)):
                        batch = next(bucket_chunks, None)
                        if batch is None:
                            break
                        pending[coordinators.submit(execute, rank, batch)] = (rank, batch)
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            rank, batch = pending.pop(future)
                            results, stats = future.result()
                            following = next(bucket_chunks, None)
                            if following is not None:
                                pending[coordinators.submit(execute, rank, following)] = (rank, following)
                            now = time.monotonic()
                            # Sum wall intervals, not overlapping worker durations.
                            stats = stats | {"elapsed_seconds": now - last_completion}
                            last_completion = now
                            yield batch, results, stats
                finally:
                    for future in pending:
                        future.cancel()

    def __call__(self, model, prompts, temperature, max_tokens, **kwargs):
        self._start_workers()
        config = kwargs["config"]
        if not self._budgets or any(_key(p) not in self._budgets for p in prompts):
            self.prepare(prompts,kwargs["source_texts"],config)
        jobs=[]
        for rank, executor in enumerate(self.executors):
            indices=list(range(rank,len(prompts),len(self.groups)))
            if not indices: continue
            subconfig=config | {"tensor_parallel_size":len(self.groups[rank].split(','))}
            args={"config":subconfig}
            for name in ("source_texts","request_ids","attempts"):
                args[name]=[kwargs[name][i] for i in indices]
            if kwargs.get("requested_edits") is not None:
                args["requested_edits"] = [kwargs["requested_edits"][i] for i in indices]
            subprompts=[prompts[i] for i in indices]
            jobs.append((indices,executor.submit(_execute,model,subprompts,temperature,max_tokens,args,
                                                {_key(p):self._budgets[_key(p)] for p in subprompts})))
        outputs=[None]*len(prompts)
        counts={}; elapsed=0.
        for indices, future in jobs:
            values,stats=future.result()
            if len(values)!=len(indices): raise RuntimeError("Replica returned the wrong output count")
            for i,value in zip(indices,values): outputs[i]=value
            for key,count in stats["bucket_counts"].items(): counts[key]=counts.get(key,0)+count
            elapsed=max(elapsed,stats["elapsed_seconds"])
        self.last_context_stats={"bucket_counts":counts,"elapsed_seconds":elapsed}
        return outputs

    def close(self):
        for executor in self.executors: executor.shutdown(wait=True,cancel_futures=True)
        self.executors=[]
        super().close()
