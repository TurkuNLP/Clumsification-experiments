# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from clumsification_code.evals.geval.parser import parse_score_response
from clumsification_code.evals.geval.prompts import (
    GEVAL_QE_PROMPT_VERSION,
    build_messages,
    build_response_format_json_schema,
)


class GEvalScorer:
    """
    No-reference / QE-style G-Eval scorer.

    This class exposes score_texts(...), matching the interface expected by the
    benchmark runner and the existing FE/MetricX/GPTScore adapters.

    Scores are higher-is-better, on the raw 1-5 G-Eval scale.
    """

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_path: Optional[str] = None,
        score_min: float = 1.0,
        score_max: float = 5.0,
        temperature: float = 0.0,
        n_samples: int = 1,
        max_output_tokens: int = 256,
        max_input_chars: int = 12000,
        sleep_seconds: float = 0.0,
        max_retries: int = 8,
        concurrency: Optional[int] = None,
        response_format: str = "json_schema",
        task: Optional[str] = None,
        aspect: Optional[str] = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        if score_min >= score_max:
            raise ValueError(f"score_min must be < score_max, got {score_min} >= {score_max}")

        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")

        if concurrency is not None and concurrency < 1:
            raise ValueError(f"concurrency must be >= 1 when provided, got {concurrency}")

        if response_format not in {"json_schema", "json_object", "none"}:
            raise ValueError(
                "response_format must be one of {'json_schema', 'json_object', 'none'}, "
                f"got {response_format!r}"
            )

        self.model_name = model_name
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.temperature = float(temperature)
        self.n_samples = int(n_samples)
        self.max_output_tokens = int(max_output_tokens)
        self.max_input_chars = int(max_input_chars)
        self.sleep_seconds = float(sleep_seconds)
        self.max_retries = int(max_retries)
        self.concurrency = int(concurrency) if concurrency is not None else None
        self.response_format = response_format
        self.task = task
        self.aspect = aspect

        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.async_client = AsyncOpenAI(**client_kwargs)

        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    @classmethod
    def from_args(cls, args: Any) -> "GEvalScorer":
        return cls(
            model_name=getattr(args, "geval_model", "gpt-4o-mini"),
            api_key=getattr(args, "api_key", None),
            base_url=getattr(args, "base_url", None),
            cache_path=getattr(args, "cache_path", None),
            score_min=getattr(args, "geval_score_min", 1.0),
            score_max=getattr(args, "geval_score_max", 5.0),
            temperature=getattr(args, "temperature", 0.0),
            n_samples=getattr(args, "n_samples", 1),
            max_output_tokens=getattr(args, "max_output_tokens", 256),
            max_input_chars=getattr(args, "max_input_chars", 12000),
            sleep_seconds=getattr(args, "sleep_seconds", 0.0),
            max_retries=getattr(args, "max_retries", 8),
            concurrency=getattr(args, "geval_concurrency", None),
            response_format=getattr(args, "geval_response_format", "json_schema"),
            task=getattr(args, "geval_task", None),
            aspect=getattr(args, "geval_aspect", None),
        )

    def _load_cache(self) -> None:
        if self.cache_path is None:
            return

        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.cache_path.parent),
            delete=False,
        ) as tmp:
            json.dump(self.cache, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)

        tmp_path.replace(self.cache_path)

    def _cache_key(self, text: str) -> str:
        payload = {
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "model_name": self.model_name,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "max_output_tokens": self.max_output_tokens,
            "max_input_chars": self.max_input_chars,
            "response_format": self.response_format,
            "task": self.task,
            "aspect": self.aspect,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        return build_messages(
            text,
            max_input_chars=self.max_input_chars,
            task=self.task,
            aspect=self.aspect,
        )

    def _get_response_format(self) -> Optional[Dict[str, Any]]:
        if self.response_format == "json_schema":
            return build_response_format_json_schema()

        if self.response_format == "json_object":
            return {"type": "json_object"}

        return None
    
    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task = task_name
        self.aspect = aspect

    async def _score_one_uncached_async(self, text: str) -> Dict[str, Any]:
        messages = self._build_messages(text)
        response_format = self._get_response_format()

        scores: List[float] = []
        raw_responses: List[str] = []
        parsed_payloads: List[Dict[str, Any]] = []

        for sample_idx in range(self.n_samples):
            for attempt in range(self.max_retries):
                try:
                    request_kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_output_tokens,
                    }
                    if response_format is not None:
                        request_kwargs["response_format"] = response_format

                    completion = await self.async_client.chat.completions.create(
                        **request_kwargs
                    )

                    raw_content = completion.choices[0].message.content or ""
                    parsed = parse_score_response(
                        raw_content,
                        score_min=self.score_min,
                        score_max=self.score_max,
                        clamp=True,
                    )

                    scores.append(parsed.score)
                    raw_responses.append(raw_content)
                    parsed_payloads.append(parsed.payload)

                    if self.sleep_seconds > 0:
                        await asyncio.sleep(self.sleep_seconds)

                    break

                except Exception as exc:
                    is_last = attempt == self.max_retries - 1
                    if is_last:
                        raise RuntimeError(
                            f"G-Eval API/parse call failed after {self.max_retries} attempts. "
                            f"sample_idx={sample_idx}; error={exc}"
                        ) from exc

                    wait_s = min(60.0, (2.0**attempt) + random.random())
                    print(
                        f"G-Eval call failed on attempt {attempt + 1}/"
                        f"{self.max_retries}: {exc}. Retrying in {wait_s:.1f}s."
                    )
                    await asyncio.sleep(wait_s)

        mean_score = float(np.mean(scores))

        return {
            "score": mean_score,
            "scores": scores,
            "raw_responses": raw_responses,
            "parsed_payloads": parsed_payloads,
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "response_format": self.response_format,
            "task": self.task,
            "aspect": self.aspect,
        }

    async def _score_texts_async(
        self,
        texts: List[str],
        *,
        concurrency: int,
    ) -> np.ndarray:
        semaphore = asyncio.Semaphore(concurrency)
        cache_lock = asyncio.Lock()
        results: List[Optional[float]] = [None] * len(texts)
        cache_changed = False

        key_to_text: Dict[str, str] = {}
        key_to_indices: Dict[str, List[int]] = defaultdict(list)

        for i, text in enumerate(texts):
            text = "" if text is None else str(text)
            key = self._cache_key(text)

            if key in self.cache:
                results[i] = float(self.cache[key]["score"])
            else:
                key_to_text.setdefault(key, text)
                key_to_indices[key].append(i)

        async def score_key(key: str) -> None:
            nonlocal cache_changed

            async with semaphore:
                async with cache_lock:
                    cached_record = self.cache.get(key)

                if cached_record is not None:
                    score = float(cached_record["score"])
                else:
                    record = await self._score_one_uncached_async(key_to_text[key])
                    score = float(record["score"])

                    async with cache_lock:
                        if key not in self.cache:
                            self.cache[key] = record
                            cache_changed = True

                if not np.isfinite(score):
                    raise RuntimeError(f"Non-finite G-Eval score for cache key {key}")

                for idx in key_to_indices[key]:
                    results[idx] = score

        tasks = [asyncio.create_task(score_key(key)) for key in key_to_text]

        if tasks:
            for fut in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"G-Eval scoring async x{concurrency}",
            ):
                await fut

        if cache_changed:
            self._save_cache()

        if any(score is None for score in results):
            missing = sum(score is None for score in results)
            raise RuntimeError(f"Internal G-Eval error: {missing} scores were not filled")

        return np.asarray(results, dtype=np.float64)

    @torch.no_grad()
    def score_texts(
        self,
        texts: Iterable[str],
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Benchmark-runner-compatible interface.

        device and max_length are accepted for compatibility with local scorers.
        For G-Eval, use --max-input-chars to control prompt length.

        If --geval-concurrency/--concurrency is set, that controls API
        concurrency. Otherwise batch_size is used.
        """
        del device, max_length

        texts_list = list(texts)
        concurrency = self.concurrency if self.concurrency is not None else batch_size
        concurrency = max(1, int(concurrency))

        return asyncio.run(
            self._score_texts_async(
                texts_list,
                concurrency=concurrency,
            )
        )
