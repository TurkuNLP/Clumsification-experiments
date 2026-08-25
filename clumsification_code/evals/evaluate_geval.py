# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""
Evaluate a G-Eval no-reference baseline on the same benchmark suite used by
evaluate_model_on_benchmark.py, so results are directly comparable.

Expected layout:
- this file lives alongside evaluate_model_on_benchmark.py
- benchmark data files are in data/benchmarks/...
- OPENAI_API_KEY is set, or --api-key is passed

Example:
python evaluate_geval_on_benchmark.py \
  --model-name GEval_gpt4o_mini_QE \
  --training-dataset baseline \
  --perturbation-type none \
  --num-layers 0 \
  --context-length 12000 \
  --geval-model gpt-4o-mini \
  --batch-size 1 \
  --max-input-chars 12000 \
  --temperature 0 \
  --n-samples 1 \
  --cache-path data/evals/geval_cache_gpt4o_mini.json \
  --api-key 

Notes:
- This is a no-reference / QE-style G-Eval adapter. It only sees the candidate
  text, because the shared benchmark runner provides only texts to score.
- Scores are higher-is-better.
- Correlation metrics are invariant to affine rescaling, so the raw 1-5 G-Eval
  score is returned directly.
"""

# This script has been co-created, refactored, and cleaned using GPT 5.6.
import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import torch
from openai import OpenAI, AsyncOpenAI
import asyncio
from tqdm.auto import tqdm

# Reuse your existing benchmark loaders/metrics/writer.
import clumsification_code.evals.evaluate_model_on_benchmark as bench
from clumsification_code.evals.geval.prompts import (
    GEVAL_QE_PROMPT_VERSION,
    build_messages,
    build_response_format_json_schema,
)


# ──────────────────────────────────────────────────────────────────────
#  G-Eval no-reference / QE adapter
# ──────────────────────────────────────────────────────────────────────


class GEvalQEInferenceModel:
    """
    Adapter exposing the same score_texts(...) interface expected by
    evaluate_model_on_benchmark.py.

    The model assigns a no-reference text-quality score from 1 to 5:

      1 = very poor
      2 = poor / weak
      3 = acceptable
      4 = good
      5 = excellent

    Returned scores are higher-is-better.

    Why no-reference?
    -----------------
    The original FE benchmark code only passes a single string per candidate
    to getModelPreds(...). Therefore this adapter intentionally does not use
    task sources or references. This keeps the comparison fair against the
    custom QE model and the MetricX QE adapter.
    """

    def __init__(
        self,
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
        concurrency: int = 8,
    ):
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        if score_min >= score_max:
            raise ValueError(
                f"score_min must be < score_max, got {score_min} >= {score_max}"
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
        self.concurrency = int(concurrency)

        client_kwargs: Dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load deterministic local cache if available."""
        if self.cache_path is None:
            return

        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self) -> None:
        """Atomically save local cache."""
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
        """
        Cache key includes all scoring settings that can affect the score.

        This is important for reproducibility: changing prompt version,
        judge model, sampling, or truncation creates a different cache entry.
        """
        payload = {
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "model_name": self.model_name,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "max_output_tokens": self.max_output_tokens,
            "max_input_chars": self.max_input_chars,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _truncate_text(self, text: str) -> str:
        """
        Character-level truncation for API cost/control.

        We use chars rather than tokenizer-specific tokens to keep this adapter
        model-agnostic across OpenAI-compatible endpoints.
        """
        text = "" if text is None else str(text)

        if len(text) <= self.max_input_chars:
            return text

        return text[: self.max_input_chars] + "\n\n[TRUNCATED]"

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        return build_messages(
            text,
            max_input_chars=self.max_input_chars,
            aspect="fluency",
        )

    @staticmethod
    def _extract_json_object(raw: str) -> Dict[str, Any]:
        """
        Robust fallback parser.

        Structured outputs should return valid JSON. This fallback exists only
        to protect long experiments from occasional provider/proxy formatting
        deviations.
        """
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Could not parse JSON object from response: {raw[:500]!r}")

    def _parse_score(self, raw_content: str) -> float:
        obj = self._extract_json_object(raw_content)

        if "score" not in obj:
            raise ValueError(f"G-Eval response missing 'score': {obj}")

        score = float(obj["score"])

        if not np.isfinite(score):
            raise ValueError(f"G-Eval score is non-finite: {score}")

        # Clamp rather than fail: protects long benchmark runs from rare
        # near-boundary outputs such as 5.00001.
        score = float(np.clip(score, self.score_min, self.score_max))
        return score

    async def _score_one_uncached_async(self, text: str) -> Dict[str, Any]:
        messages = self._build_messages(text)

        response_format = build_response_format_json_schema()

        scores: List[float] = []
        raw_responses: List[str] = []

        for sample_idx in range(self.n_samples):
            for attempt in range(self.max_retries):
                try:
                    completion = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                        response_format=response_format,
                    )

                    raw_content = completion.choices[0].message.content or ""
                    score = self._parse_score(raw_content)

                    scores.append(score)
                    raw_responses.append(raw_content)

                    if self.sleep_seconds > 0:
                        await asyncio.sleep(self.sleep_seconds)

                    break

                except Exception as e:
                    is_last = attempt == self.max_retries - 1
                    if is_last:
                        raise RuntimeError(
                            f"G-Eval API call failed after {self.max_retries} attempts. "
                            f"sample_idx={sample_idx}; error={e}"
                        ) from e

                    wait_s = min(60.0, (2.0**attempt) + random.random())
                    print(
                        f"G-Eval call failed on attempt {attempt + 1}/"
                        f"{self.max_retries}: {e}. Retrying in {wait_s:.1f}s."
                    )
                    await asyncio.sleep(wait_s)

        mean_score = float(np.mean(scores))

        return {
            "score": mean_score,
            "scores": scores,
            "raw_responses": raw_responses,
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
        }
    
    def _score_one_uncached(self, text: str) -> Dict[str, Any]:
        messages = self._build_messages(text)

        response_format = build_response_format_json_schema()

        scores: List[float] = []
        raw_responses: List[str] = []

        # We make n_samples separate calls instead of using n=... so this works
        # with more OpenAI-compatible providers and keeps retry granularity small.
        for sample_idx in range(self.n_samples):
            for attempt in range(self.max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                        response_format=response_format,
                    )

                    raw_content = completion.choices[0].message.content or ""
                    score = self._parse_score(raw_content)

                    scores.append(score)
                    raw_responses.append(raw_content)

                    if self.sleep_seconds > 0:
                        time.sleep(self.sleep_seconds)

                    break

                except Exception as e:
                    is_last = attempt == self.max_retries - 1
                    if is_last:
                        raise RuntimeError(
                            "G-Eval API call failed after "
                            f"{self.max_retries} attempts. "
                            f"sample_idx={sample_idx}; error={e}"
                        ) from e

                    # Exponential backoff with jitter, useful for rate limits and
                    # transient API errors.
                    wait_s = min(60.0, (2.0**attempt) + random.random())
                    print(
                        f"G-Eval call failed on attempt {attempt + 1}/"
                        f"{self.max_retries}: {e}. Retrying in {wait_s:.1f}s."
                    )
                    time.sleep(wait_s)

        mean_score = float(np.mean(scores))

        return {
            "score": mean_score,
            "scores": scores,
            "raw_responses": raw_responses,
            "prompt_version": GEVAL_QE_PROMPT_VERSION,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
        }

    async def _score_texts_async(
        self,
        texts: List[str],
        concurrency: int,
    ) -> np.ndarray:
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[float]] = [None] * len(texts)
        cache_changed = False

        async def score_index(i: int, text: str):
            nonlocal cache_changed

            text = "" if text is None else str(text)
            key = self._cache_key(text)

            if key in self.cache:
                results[i] = float(self.cache[key]["score"])
                return

            async with semaphore:
                # Double-check after waiting for semaphore.
                if key in self.cache:
                    results[i] = float(self.cache[key]["score"])
                    return

                record = await self._score_one_uncached_async(text)
                self.cache[key] = record
                cache_changed = True
                results[i] = float(record["score"])

        tasks = [score_index(i, text) for i, text in enumerate(texts)]

        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"G-Eval scoring async x{concurrency}",
        ):
            await fut

        if cache_changed:
            self._save_cache()

        return np.asarray(results, dtype=np.float64)
    
    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Adapter expected by evaluate_model_on_benchmark.py.

        Parameters device, batch_size, and max_length are accepted for interface
        compatibility with the learned model and MetricX adapter. For API-based
        G-Eval, scoring is sequential by default for rate-limit safety.

        max_length is ignored; use --max-input-chars instead.
        """
        del device, max_length

        concurrency = max(1, int(batch_size))

        output_scores: List[float] = []
        cache_changed = False
        if concurrency <= 1:
            for text in tqdm(texts, desc="G-Eval scoring"):
                text = "" if text is None else str(text)
                key = self._cache_key(text)

                if key in self.cache:
                    record = self.cache[key]
                else:
                    record = self._score_one_uncached(text)
                    self.cache[key] = record
                    cache_changed = True

                    # Save after every new score to avoid losing expensive results
                    # if the benchmark run is interrupted.
                    self._save_cache()

                score = float(record["score"])

                if not np.isfinite(score):
                    raise RuntimeError(f"Non-finite G-Eval score for cache key {key}")

                output_scores.append(score)
        

            if cache_changed:
                self._save_cache()

            return np.asarray(output_scores, dtype=np.float64)
        
        else:
            return asyncio.run(self._score_texts_async(texts, concurrency))


# ──────────────────────────────────────────────────────────────────────
#  Argument parser
# ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a G-Eval no-reference baseline on the full benchmark suite "
        "and log JSONL results."
    )

    # Output metadata.
    parser.add_argument("--model-name", type=str, default="GEval_QE")
    parser.add_argument("--training-dataset", type=str, default="baseline")
    parser.add_argument("--perturbation-type", type=str, default="none")
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=12000)

    # G-Eval / OpenAI config.
    parser.add_argument("--geval-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, the OpenAI client uses OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature. Use 0 for deterministic-ish scoring.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of independent judge samples per text. Scores are averaged.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Maximum tokens for each judge response.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=12000,
        help="Maximum candidate-text characters sent to the judge.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep after successful API calls for rate-limit control.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries per API call.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/evals/geval_cache.json",
        help="JSON cache path. Set to empty string to disable cache.",
    )

    # Interface-compatible settings used by shared benchmark functions.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Accepted for compatibility. G-Eval scoring is sequential.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Accepted for compatibility. Use --max-input-chars for G-Eval.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent G-Eval API requests.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Main benchmark runner
# ──────────────────────────────────────────────────────────────────────


def legacy_main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_path = args.cache_path if args.cache_path else None

    print(f"Device: {device}")
    print(f"G-Eval model: {args.geval_model}")
    print(f"Temperature: {args.temperature}")
    print(f"n_samples: {args.n_samples}")
    print(f"max_input_chars: {args.max_input_chars}")
    print(f"cache_path: {cache_path}")
    print()

    model = GEvalQEInferenceModel(
        model_name=args.geval_model,
        api_key=args.api_key,
        base_url=args.base_url,
        cache_path=cache_path,
        temperature=args.temperature,
        n_samples=args.n_samples,
        max_output_tokens=args.max_output_tokens,
        max_input_chars=args.max_input_chars,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
    )

    # Sanity probe.
    probe = [
        "This is a fluent, grammatical sentence.",
        "This sentence broken grammar bad.",
    ]
    probe_scores = model.score_texts(probe)
    print("G-Eval probe scores, higher is better:", probe_scores)
    print()

    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size

    all_results: Dict[str, Any] = {}

    # ==================================================================
    # Preference-style HF benchmarks
    # ==================================================================

    print("=" * 60)
    print(" Preference-style HF benchmarks")
    print("=" * 60)

    # ── JFLEG ─────────────────────────────────────────────────────
    # Skip for geval so that there won't be unnecessary costs
    """
    jfleg_results = bench.eval_jfleg_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    for bench_name, metrics in jfleg_results.items():
        all_results.update(bench._flatten_preference_metrics(bench_name, metrics))

    # ── MultiBLiMP ────────────────────────────────────────────────
    multiblimp_metrics = bench.eval_multiblimp_english_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench._flatten_preference_metrics(
            "MultiBLiMP_eng_minimal_pair_preference",
            multiblimp_metrics,
        )
    )

    # ── Story Cloze ───────────────────────────────────────────────
    storycloze_results = bench.eval_story_cloze_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    for bench_name, metrics in storycloze_results.items():
        all_results.update(bench._flatten_preference_metrics(bench_name, metrics))
    """

    # ==================================================================
    # Scalar human-score benchmarks
    # ==================================================================

    print()
    print("=" * 60)
    print(" Scalar human-score benchmarks")
    print("=" * 60)

    # ── SummEval ──────────────────────────────────────────────────
    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]
    print(f"\nSummEval texts: {len(summeval_texts)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        summeval_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    all_results.update(
        bench.correlation_bundle(
            [x for y in ds["fluency"] for x in y],
            raw_preds,
            "summeval_fluency",
        )
    )
    all_results.update(
        bench.correlation_bundle(
            [x for y in ds["coherence"] for x in y],
            raw_preds,
            "summeval_coherence",
        )
    )
    all_results.update(
        bench.correlation_bundle(
            [x for y in ds["consistency"] for x in y],
            raw_preds,
            "summeval_consistency",
        )
    )

    # ── ELLIPSE ───────────────────────────────────────────────────
    ellipse_ds = bench.load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    print(f"\nELLIPSE texts: {len(ellipse_ds)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        ellipse_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(ellipse_ds["overall"], raw_preds, "ellipse_overall")
    )
    all_results.update(
        bench.correlation_bundle(ellipse_ds["cohesion"], raw_preds, "ellipse_cohesion")
    )

    # ── USR – Topical Chat ────────────────────────────────────────
    tc_texts, tc_overall_labels = bench.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Overall",
    )
    print(f"\nTopicalChat texts: {len(tc_texts)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        tc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(tc_overall_labels, raw_preds, "tc_overall")
    )

    _, tc_natural_labels = bench.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Natural",
    )
    all_results.update(
        bench.correlation_bundle(tc_natural_labels, raw_preds, "tc_natural")
    )

    # ── USR – Persona Chat ────────────────────────────────────────
    pc_texts, pc_overall_labels = bench.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Overall",
    )
    print(f"\nPersonaChat texts: {len(pc_texts)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        pc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(pc_overall_labels, raw_preds, "pc_overall")
    )

    _, pc_natural_labels = bench.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Natural",
    )
    all_results.update(
        bench.correlation_bundle(pc_natural_labels, raw_preds, "pc_natural")
    )

    # ── OpenMEVA ──────────────────────────────────────────────────
    meva_texts_roc, meva_labels_roc = bench.load_data_openmeva(
        "data/benchmarks/mans_roc.json"
    )
    meva_texts_wp, meva_labels_wp = bench.load_data_openmeva(
        "data/benchmarks/mans_wp.json"
    )
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    print(f"\nOpenMEVA texts: {len(meva_texts)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        meva_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(meva_labels, raw_preds, "OpenMEVA_overall")
    )

    # ── WebNLG ────────────────────────────────────────────────────
    webnlg_texts, webnlg_labels = bench.load_data_webnlg(
        "data/benchmarks/web_nlg_2020_human_evals_en.json"
    )
    print(f"\nWebNLG texts: {len(webnlg_texts)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        webnlg_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(webnlg_labels, raw_preds, "WebNLG_fluency")
    )

    # ── HANNA ─────────────────────────────────────────────────────
    hanna_ds = bench.load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    print(f"\nHANNA texts: {len(hanna_ds)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        hanna_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(hanna_ds["coherence"], raw_preds, "HANNA_coherence")
    )
    all_results.update(
        bench.correlation_bundle(hanna_ds["complexity"], raw_preds, "HANNA_complexity")
    )

    # ── ARG-ESSAY ─────────────────────────────────────────────────
    arge_ds = bench.load_argessay_data("data/benchmarks/arg-essay.csv")
    print(f"\nARG-ESSAY texts: {len(arge_ds)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        arge_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(
            arge_ds["language_mastery"],
            raw_preds,
            "ARG-ESSAY_language_mastery",
        )
    )
    all_results.update(
        bench.correlation_bundle(
            arge_ds["complexity"],
            raw_preds,
            "ARG-ESSAY_complexity",
        )
    )
    all_results.update(
        bench.correlation_bundle(
            arge_ds["vocabulary"],
            raw_preds,
            "ARG-ESSAY_vocabulary",
        )
    )
    all_results.update(
        bench.correlation_bundle(
            arge_ds["language_constructs"],
            raw_preds,
            "ARG-ESSAY_language_constructs",
        )
    )

    # ── Human Ratings of NLG ──────────────────────────────────────
    hr_ds = bench.load_human_ratings_of_nlg_data(
        "data/benchmarks/human_ratings_of_nlg.csv"
    )
    print(f"\nHumanRatings texts: {len(hr_ds)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        hr_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(hr_ds["quality"], raw_preds, "HumanRatings_quality")
    )
    all_results.update(
        bench.correlation_bundle(
            hr_ds["naturalness"],
            raw_preds,
            "HumanRatings_naturalness",
        )
    )

    # ── FED ───────────────────────────────────────────────────────
    turn_ds, whole_ds = bench.load_fed_data("data/benchmarks/fed_data.json")
    print(f"\nFED turn-level texts: {len(turn_ds)}")
    raw_preds_turn = bench.getModelPreds(
        device,
        model,
        turn_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    print(f"FED whole-dialogue texts: {len(whole_ds)}")
    raw_preds_whole = bench.getModelPreds(
        device,
        model,
        whole_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    all_results.update(
        bench.correlation_bundle(turn_ds["fluent"], raw_preds_turn, "FED_turn_fluency")
    )
    all_results.update(
        bench.correlation_bundle(turn_ds["overall"], raw_preds_turn, "FED_turn_overall")
    )
    all_results.update(
        bench.correlation_bundle(
            whole_ds["overall"],
            raw_preds_whole,
            "FED_whole_overall",
        )
    )

    # ── E2E ───────────────────────────────────────────────────────
    """
    e2e_ds = bench.load_e2e_data("data/benchmarks/E2E_data")
    print(f"\nE2E texts: {len(e2e_ds)}")
    raw_preds = bench.getModelPreds(
        device,
        model,
        e2e_ds["text"],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    all_results.update(
        bench.correlation_bundle(e2e_ds["naturalness"], raw_preds, "E2E_naturalness")
    )
    all_results.update(
        bench.correlation_bundle(e2e_ds["quality"], raw_preds, "E2E_quality")
    )
    """

    # ==================================================================
    # Write JSONL
    # ==================================================================

    bench.write_results_jsonl(
        model_name=args.model_name,
        training_dataset=args.training_dataset,
        perturbation_type=args.perturbation_type,
        num_layers=args.num_layers,
        context_length=args.context_length,
        model_dir=args.geval_model,
        results=all_results,
    )


def main(argv=None):
    """Compatibility entry point using the shared registry-backed runner.

    The previous implementation remains available as ``legacy_main`` while
    callers transition.  Keeping this wrapper preserves the familiar script
    path but ensures G-Eval uses the same benchmark selection as FE, GPTScore,
    and MetricX.
    """
    import sys

    from clumsification_code.evals.run_benchmark import main as shared_main

    forwarded = list(sys.argv[1:] if argv is None else argv)
    if not any(argument == "--scorer" or argument.startswith("--scorer=") for argument in forwarded):
        forwarded.insert(0, "--scorer=geval")
    shared_main(forwarded)


if __name__ == "__main__":
    main()
