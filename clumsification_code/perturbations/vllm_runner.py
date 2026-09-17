"""Persistent vLLM inference with per-request thinking and answer budgets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time
from typing import Any

from .length_planning import load_tokenizer, plan_token_budgets, budget_report
from .output_parsing import parse_completion
from .schemas import ChatCompletion, SkippedGeneration

OUTPUT_SCHEMA = {
    "type": "object", "properties": {
        "text": {"type": "string"},
        "applied_edits": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
    }, "required": ["text", "applied_edits"], "additionalProperties": False,
}


def output_schema(edit_ids=None):
    """Constrain counts to the requested operations when assignments are available."""
    if edit_ids is None:
        return OUTPUT_SCHEMA
    counts = {
        "type": "object",
        "properties": {edit_id: {"type": "integer", "minimum": 0} for edit_id in edit_ids},
        "required": list(edit_ids),
        "additionalProperties": False,
    }
    return OUTPUT_SCHEMA | {"properties": OUTPUT_SCHEMA["properties"] | {"applied_edits": counts}}


def _key(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


class VLLMRunner:
    supports_generation_config = True

    def __init__(self, *, tokenizer=None):
        self.tokenizer = tokenizer
        self._engine = None
        self._engine_key = None
        self._budgets = {}
        self._plan_key = None
        self.last_context_stats = {}
        self.profile = {}

    def prepare(self, prompts, source_texts, config):
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(config["model"], config)
        self._plan_key = json.dumps(config, sort_keys=True)
        budgets, boundaries = plan_token_budgets(prompts, source_texts, self.tokenizer, config)
        self._budgets = {_key(prompt): budget for prompt, budget in zip(prompts, budgets)}
        self.profile = budget_report(budgets, boundaries)
        print("Token plan: " + json.dumps(self.profile), flush=True)
        return budgets

    def close(self):
        if self._engine is not None:
            engine_core = getattr(getattr(self._engine, "llm_engine", None), "engine_core", None)
            if engine_core is not None and callable(getattr(engine_core, "shutdown", None)):
                engine_core.shutdown()
            self._engine = None
            self._engine_key = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_batches(self, batches, *, adapter, runtime, batch_size):
        """Reinitialize once per bucket with its measured engine context limit."""
        from .schemas import GenerationRuntime
        previous = object()
        for batch in batches:
            key = getattr(batch, "bucket_key", None)
            if key != previous:
                self.close()
                previous = key
            context_limit = getattr(batch, "context_limit", None)

            def chat(model, prompts, temperature, max_tokens, **kwargs):
                if context_limit is not None:
                    kwargs["config"] = kwargs["config"] | {"engine_max_model_len": context_limit}
                return self(model, prompts, temperature, max_tokens, **kwargs)

            chat.supports_generation_config = True
            results = list(adapter.generate(batch, GenerationRuntime(chat_runner=chat, attempts=runtime.attempts)))
            yield batch, results, self.last_context_stats

    def _get_engine(self, model, config):
        import torch
        from vllm import LLM, SamplingParams
        thinking = config.get("enable_thinking", True)
        if thinking:
            try:
                SamplingParams(thinking_token_budget=1)
                from vllm.config import ReasoningConfig
            except (ImportError, TypeError) as exc:
                raise RuntimeError("This vLLM build lacks thinking_token_budget/ReasoningConfig; use a compatible build") from exc
        tp = int(config.get("tensor_parallel_size", 1))
        if tp > torch.cuda.device_count():
            raise ValueError(f"Requested TP={tp}, but only {torch.cuda.device_count()} devices are visible")
        args = dict(model=model, max_model_len=int(config.get("max_model_len", 32768)),
                    tensor_parallel_size=tp, language_model_only=True,
                    gpu_memory_utilization=float(config.get("gpu_memory_utilization", .9)),
                    max_num_seqs=int(config.get("max_num_seqs", 64)),
                    max_num_batched_tokens=int(config.get("max_num_batched_tokens", 8192)),
                    enable_prefix_caching=config.get("enable_prefix_caching", True),
                    enable_chunked_prefill=config.get("enable_chunked_prefill", True),
                    dtype=config.get("dtype", "bfloat16"))
        if config.get("revision"):
            args["revision"] = config["revision"]
        if config.get("tokenizer"):
            args["tokenizer"] = config["tokenizer"]
        if config.get("structured_output", True):
            args["structured_outputs_config"] = {
                "reasoning_parser": config.get("reasoning_parser", "qwen3"),
                "enable_in_reasoning": False,
            }
        key = json.dumps(args | {"thinking": thinking}, sort_keys=True)
        if self._engine_key != key:
            self.close()
            if thinking:
                args["reasoning_config"] = ReasoningConfig(reasoning_start_str="<think>", reasoning_end_str="</think>")
            self._engine = LLM(**args)
            self._engine_key = key
        return self._engine

    def __call__(self, model, prompts, temperature, max_tokens, *, config=None, source_texts=None,
                 request_ids=None, attempts=None, requested_edits=None, **kwargs):
        config = dict(config or {}) | {"model": model} | kwargs
        engine_limit = int(config.pop("engine_max_model_len", config.get("max_model_len", 32768)))
        if source_texts is None:
            raise ValueError("The vLLM runner requires explicit source_texts")
        for name, values in (("source_texts", source_texts), ("request_ids", request_ids),
                             ("attempts", attempts), ("requested_edits", requested_edits)):
            if values is not None and len(values) != len(prompts):
                raise ValueError(f"{name} must have one entry per prompt")
        if self._plan_key != json.dumps(config, sort_keys=True) or any(_key(p) not in self._budgets for p in prompts):
            self.prepare(prompts, source_texts, config)
        from vllm import SamplingParams
        results = [None] * len(prompts)
        indices, params = [], []
        stats = {}
        for i, prompt in enumerate(prompts):
            budget = self._budgets[_key(prompt)]
            stats[str(budget.source_bucket)] = stats.get(str(budget.source_bucket), 0) + 1
            if not budget.fits:
                results[i] = SkippedGeneration(budget.prompt_tokens, budget.required_tokens)
                continue
            attempt = (attempts or [0]*len(prompts))[i]
            identity = (request_ids or [_key(p) for p in prompts])[i]
            seed = int.from_bytes(hashlib.sha256(f"{config.get('seed',42)}:{identity}:{attempt}".encode()).digest()[:4], "big")
            # Failed-only retries get more answer room, never less reasoning room.
            limit = min(engine_limit-budget.prompt_tokens,
                        budget.max_tokens + attempt * max(512, budget.answer_tokens // 2))
            sampling = dict(max_tokens=limit, temperature=temperature, seed=seed,
                            top_p=float(config.get("top_p", .95)), top_k=int(config.get("top_k", 20)))
            if config.get("enable_thinking", True):
                sampling["thinking_token_budget"] = budget.thinking_tokens
            if config.get("structured_output", True):
                from vllm.sampling_params import StructuredOutputsParams
                edits = requested_edits[i] if requested_edits is not None else None
                sampling["structured_outputs"] = StructuredOutputsParams(json=output_schema(edits))
            params.append(SamplingParams(**sampling))
            indices.append(i)
        started = time.monotonic()
        if indices:
            outputs = self._get_engine(model, config | {"max_model_len": engine_limit}).chat(
                [prompts[i] for i in indices], sampling_params=params,
                chat_template_kwargs={"enable_thinking": config.get("enable_thinking", True)},
                use_tqdm=False,
            )
            if len(outputs) != len(indices):
                raise RuntimeError("vLLM did not return exactly one output per input")
            for param, i, output in zip(params, indices, outputs):
                completion = parse_completion(output, thinking=config.get("enable_thinking", True))
                choices = getattr(output, "outputs", None) or []
                raw_tokens = list(getattr(choices[0], "token_ids", []) or []) if choices else []
                end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                if end_ids:
                    boundary = next((n for n in range(len(raw_tokens)-len(end_ids)+1)
                                     if raw_tokens[n:n+len(end_ids)] == end_ids), None)
                    completion.metadata["reasoning_tokens_before_end"] = boundary
                completion.metadata.update(asdict(self._budgets[_key(prompts[i])]))
                completion.metadata["sampling_seed"] = param.seed
                completion.metadata["generation_max_tokens"] = param.max_tokens
                completion.metadata["engine_max_model_len"] = engine_limit
                completion.metadata["attempt"] = (attempts or [0]*len(prompts))[i]
                results[i] = completion
        self.last_context_stats = {"bucket_counts": stats, "elapsed_seconds": time.monotonic()-started}
        return results


_default_runner = VLLMRunner()


def run_vllm(model, prompts, temperature, max_tokens, **kwargs):
    results = _default_runner(model, prompts, temperature, max_tokens, **kwargs)
    run_vllm.last_context_stats = _default_runner.last_context_stats
    return results


run_vllm.supports_generation_config = True


def _prepare_default(*args, **kwargs):
    budgets = _default_runner.prepare(*args, **kwargs)
    run_vllm.profile = _default_runner.profile
    return budgets


run_vllm.prepare = _prepare_default
