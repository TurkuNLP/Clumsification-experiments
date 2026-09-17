"""Tokenizer-measured reasoning/answer budgets and frozen source-length buckets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from itertools import groupby
from typing import Any, Iterator

from .schemas import ChatRunner, PerturbationInput, PerturbationMethod


@dataclass(frozen=True)
class TokenBudget:
    source_tokens: int
    prompt_tokens: int
    source_bucket: int
    thinking_tokens: int
    answer_tokens: int
    extra_character_tokens: int
    transition_tokens: int
    max_tokens: int
    required_tokens: int
    fits: bool


def load_tokenizer(model: str, config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        config.get("tokenizer") or model, revision=config.get("revision"),
        trust_remote_code=False,
    )


def estimate_chat_prompt_tokens(model, messages, *, tokenizer=None, enable_thinking=True):
    tokenizer = tokenizer or load_tokenizer(model, {})
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1: raise ValueError("Expected a single rendered prompt")
        encoded = encoded[0]
    return len(encoded)


def source_boundaries(lengths: list[int]) -> tuple[int, ...]:
    """Split the short majority at p30, then p60/p90, with a separate long tail."""
    ordered = sorted(lengths)
    if not ordered:
        return ()
    return tuple(sorted({max(64, math.ceil(ordered[round((len(ordered)-1)*q)] / 64)*64)
                         for q in (0.3, 0.6, 0.9, 1.0)}))


def plan_token_budgets(prompts, source_texts, tokenizer, config, *, boundaries=None):
    if len(prompts) != len(source_texts):
        raise ValueError("Every prompt needs its source text")
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in source_texts]
    boundaries = tuple(boundaries or config.get("source_buckets") or source_boundaries(lengths))
    ceiling = int(config.get("max_model_len", 32768))
    thinking = config.get("enable_thinking", True)
    # Calibrate the 100-character target from actual text; this is not a hard limit.
    ratios = [len(tokenizer.encode(text[:100], add_special_tokens=False)) / len(text[:100])
              for text in source_texts if text]
    extra = int(config.get("extra_character_tokens") or max(32, math.ceil(100 * max(ratios, default=0.5))))
    transition = len(tokenizer.encode("</think>\n\n", add_special_tokens=False)) + 8
    budgets = []
    for messages, size in zip(prompts, lengths):
        prompt = estimate_chat_prompt_tokens("", messages, tokenizer=tokenizer, enable_thinking=thinking)
        bucket = next((b for b in boundaries if b >= size), size)
        cap = int(config.get("thinking_token_cap", 0))
        thought = min(size, cap) if cap else size
        thought = max(1, thought) if thinking else 0
        answer = bucket + extra
        # Escaping/metadata reserve is separate from edited-text space.
        overhead = int(config.get("output_metadata_tokens", 0))
        minimum = prompt + thought + answer + overhead + transition
        if minimum > ceiling:
            bucket, answer = size, size + extra  # Try exact sizing before declaring overflow.
            minimum = prompt + thought + answer + overhead + transition
        elastic = max(int(config.get("answer_reserve_tokens", 256)), math.ceil(answer * .15))
        max_tokens = min(ceiling - prompt, thought + answer + overhead + transition + elastic)
        budgets.append(TokenBudget(size, prompt, bucket, thought, answer, extra, transition,
                                   max(0, max_tokens), minimum, minimum <= ceiling))
    return budgets, boundaries


def budget_report(budgets, boundaries):
    def quantiles(values):
        values = sorted(values)
        return {str(q): values[round((len(values)-1)*q)] for q in (.1,.3,.5,.6,.9,.99,1)} if values else {}
    return {
        "input_count": len(budgets), "source_buckets": list(boundaries),
        "extra_character_tokens": max((b.extra_character_tokens for b in budgets), default=32),
        "bucket_counts": {str(b): sum(x.source_bucket == b for x in budgets) for b in sorted({x.source_bucket for x in budgets})},
        "source_token_quantiles": quantiles([x.source_tokens for x in budgets]),
        "prompt_token_quantiles": quantiles([x.prompt_tokens for x in budgets]),
        "required_token_quantiles": quantiles([x.required_tokens for x in budgets]),
        "over_context_count": sum(not x.fits for x in budgets),
        "reserved_rounding_tokens": sum(x.source_bucket-x.source_tokens for x in budgets),
    }


def plan_context_buckets(model, prompts, *, max_model_len, source_texts=None, tokenizer=None, **config):
    if source_texts is None:
        raise ValueError("Pass source_texts explicitly; source text is never extracted from prompts")
    tokenizer = tokenizer or load_tokenizer(model, config)
    budgets, _ = plan_token_budgets(prompts, source_texts, tokenizer, config | {"max_model_len": max_model_len})
    groups, skipped = {}, []
    for index, budget in enumerate(budgets):
        if budget.fits:
            groups.setdefault(budget.source_bucket, []).append(index)
        else:
            skipped.append({"prompt_index": index, "prompt_tokens": budget.prompt_tokens,
                            "required_tokens": budget.required_tokens})
    return groups, skipped


class GenerationBatch(list):
    """List-compatible checkpoint batch with an explicit scheduling bucket."""

    def __init__(self, items, *, bucket_key, context_limit=None):
        super().__init__(items)
        self.bucket_key = bucket_key
        self.context_limit = context_limit


def iter_generation_batches(items, *, adapter, runner, method_config, perturbation_source, batch_size, on_plan=None, attempts=None):
    if perturbation_source != "LLM":
        yield items
        return
    if hasattr(runner, "prepare"):
        rows = [item.metadata | {"base_text_id": item.base_text_id, "candidate_id": item.candidate_id,
                                 "text": item.text} for item in items]
        prompts = adapter.build_prompts(rows)
        budgets = runner.prepare(prompts, [str(item.text).replace("\n", " ") for item in items], method_config)
        if on_plan is not None: on_plan(runner.profile)
        def bucket_key(index):
            return (not budgets[index].fits, budgets[index].source_bucket)
        order = sorted(range(len(items)), key=bucket_key)
        for key, indices in groupby(order, key=bucket_key):
            indices = list(indices)
            bucket_items = [items[i] for i in indices]
            ceiling = int(method_config.get("max_model_len", 32768))
            context_limit = None
            if not key[0] and all(hasattr(budgets[i], "max_tokens") for i in indices):
                needed = max(
                    budgets[i].prompt_tokens + budgets[i].max_tokens
                    + (attempts or {}).get(str(items[i].candidate_id), 0)
                    * max(512, budgets[i].answer_tokens // 2)
                    for i in indices
                )
                context_limit = min(ceiling, math.ceil(needed / 256) * 256)
                print(f"Source bucket {key[1]}: engine max_model_len={context_limit}", flush=True)
            for start in range(0, len(bucket_items), batch_size):
                yield GenerationBatch(bucket_items[start:start + batch_size], bucket_key=key,
                                      context_limit=context_limit)
        return
    for start in range(0, len(items), batch_size):
        yield items[start:start+batch_size]
