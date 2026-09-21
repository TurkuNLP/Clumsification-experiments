# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""General vLLM-backed absolute-grading scorer."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clumsification_code.evals.geval.prompts import render_rubric, rubric_for
from clumsification_code.evals.geval.parser import parse_score_response
from clumsification_code.prompts import load_prompt_data, load_prompt_spec


_RESULT_RE = re.compile(
    r"(?:\[\s*RESULT\s*\]|\bRESULT\b)\s*:?\s*([1-5])\b",
    flags=re.IGNORECASE,
)
_RATING_RE = re.compile(r"\bRating\s*:\s*([1-5])\b", flags=re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_MAX_RETRIES = 5
_THEMIS_RATING_REPAIR = (
    "\n\nUsing the target text and criterion above, answer with exactly one line: "
    "Rating: N (replace N with one integer from 1 to 5). "
    "Do not include analysis or repeat the rubric."
)


class VLLMTextScorer:
    """Score candidate texts with a configurable vLLM protocol and rubric."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        protocol: str = "prometheus_direct_assessment.json",
        rubric: str = "menlo_fluency.json",
        task: Optional[str] = None,
        aspect: str = "fluency",
    ) -> None:
        #vllm is loaded here so that the package can still be used in environments without vllm
        #Useful in HPC envs
        from vllm import LLM, SamplingParams

        self.model_name_or_path = model_name_or_path
        self.task = task
        self.aspect = aspect
        self.protocol = protocol
        self.rubric = rubric
        self.protocol_spec = load_prompt_spec(f"evaluation/protocols/{protocol}")
        self.rubric_data = load_prompt_data(f"evaluation/rubrics/{rubric}")
        self.output_parser = self.protocol_spec.metadata.get(
            "output_parser", "prometheus_result"
        )
        # Themis ships without a tokenizer chat template.  Its official
        # evaluator sends the evaluation prompt as a plain completion, so do
        # not route this protocol through LLM.chat(), which requires a chat
        # template in recent Transformers/vLLM versions.
        self.input_mode = (
            "raw_completion"
            if self.protocol_spec.metadata.get("method") == "themis"
            else "chat"
        )
        self.enable_thinking = enable_thinking
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.rating_repair_sampling_params = (
            SamplingParams(
                temperature=0.2,
                repetition_penalty=1.15,
                max_tokens=min(max_tokens, 128),
            )
            if self.output_parser == "themis_rating" else None
        )
        llm_kwargs: Dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "language_model_only": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task = task_name
        self.aspect = aspect

    def score_cache_context(self) -> str:
        """Identify the rendered prompt apart from the candidate text."""
        return repr(self._messages(""))

    def _messages(self, text: str) -> List[Dict[str, str]]:
        if "rubric" in self.rubric_data:
            rubric = self.rubric_data["rubric"]
        elif "aspects" in self.rubric_data:
            rubric = render_rubric(rubric_for(task=self.task, aspect=self.aspect))
        else:
            raise ValueError(f"Rubric file has no usable rubric content: {self.rubric}")
        values = {
            "instruction": "Assess the quality of the response for the requested criterion.",
            "candidate_text": "" if text is None else str(text),
            "rubric": rubric,
        }
        return self.protocol_spec.render_messages(
            {name: values[name] for name in self.protocol_spec.required_variables}
        )

    @staticmethod
    def _parse_score(text: str) -> float:
        visible_text = _THINK_BLOCK_RE.sub("", text or "")
        match = _RESULT_RE.search(visible_text)
        if match is None:
            raise ValueError(f"Could not find [RESULT] score in vLLM output: {text}")
        return float(match.group(1))

    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        del device, max_length
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        prompts = [self._messages(text) for text in texts]
        scores = np.full(len(texts), np.nan, dtype=np.float32)
        pending = list(range(len(texts)))
        errors: Dict[int, BaseException] = {}

        for attempt in range(_MAX_RETRIES + 1):
            if not pending:
                break

            next_pending: List[int] = []
            # vLLM schedules a submitted request list against its own KV-cache
            # capacity. A small outer loop would cap the scheduler at 32
            # requests even when hundreds are waiting on this replica.
            submission_size = len(pending) if attempt == 0 else batch_size
            starts = range(0, len(pending), submission_size)
            for start in starts:
                indices = pending[start : start + submission_size]
                retry_rating = attempt > 0 and self.output_parser == "themis_rating"
                batch_prompts = [prompts[index] for index in indices]
                if retry_rating:
                    batch_prompts = [self._rating_repair_prompt(prompt) for prompt in batch_prompts]
                batch_scores, batch_errors = self._score_batch(
                    batch_prompts,
                    sampling_params=(
                        self.rating_repair_sampling_params
                        if retry_rating else None
                    ),
                )
                for index, score, error in zip(indices, batch_scores, batch_errors):
                    if error is None:
                        scores[index] = score
                    else:
                        next_pending.append(index)
                        errors[index] = error
            pending = next_pending

        if pending:
            details = "; ".join(
                f"index {index}: {errors[index]}" for index in pending
            )
            raise RuntimeError(
                f"vLLM scoring failed for {len(pending)} text(s) after "
                f"{_MAX_RETRIES} retries: {details}"
            )
        return scores

    def score_texts_once(
        self, texts: List[str]
    ) -> Tuple[List[float], List[Optional[BaseException]]]:
        """Submit one batch once and retain each text's parse or request error."""
        if not texts:
            return [], []
        return self._score_batch([self._messages(text) for text in texts])

    @staticmethod
    def _rating_repair_prompt(prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        repaired = [dict(message) for message in prompt]
        repaired[-1]["content"] += _THEMIS_RATING_REPAIR
        return repaired

    def _score_batch(
        self, prompts: List[List[Dict[str, str]]], *, sampling_params=None,
    ) -> Tuple[List[float], List[Optional[BaseException]]]:
        """Score one request batch without allowing one output to abort it."""
        sampling_params = sampling_params or self.sampling_params
        try:
            if self.input_mode == "raw_completion":
                outputs = self.llm.generate(
                    ["\n\n".join(message["content"] for message in prompt) for prompt in prompts],
                    sampling_params=sampling_params,
                    # Benchmark results provide the useful progress signal.
                    use_tqdm=False,
                )
            else:
                outputs = self.llm.chat(
                    prompts,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": self.enable_thinking},
                    # Avoid per-request progress output in batch logs.
                    use_tqdm=False,
                )
        except Exception as exc:
            return [float("nan")] * len(prompts), [exc] * len(prompts)

        scores: List[float] = []
        errors: List[Optional[BaseException]] = []
        for output in outputs:
            try:
                scores.append(self._parse_output(output.outputs[0].text))
                errors.append(None)
            except Exception as exc:
                scores.append(float("nan"))
                errors.append(exc)

        # A malformed provider response can contain fewer outputs than prompts.
        for _ in range(len(prompts) - len(scores)):
            scores.append(float("nan"))
            errors.append(ValueError("vLLM returned fewer outputs than prompts"))
        return scores, errors

    def _parse_output(self, text: str) -> float:
        if self.output_parser == "json_score":
            visible_text = _THINK_BLOCK_RE.sub("", text or "").strip()
            return parse_score_response(visible_text).score
        if self.output_parser == "themis_rating":
            visible_text = _THINK_BLOCK_RE.sub("", text or "")
            match = _RATING_RE.search(visible_text)
            if match is None:
                raise ValueError(
                    f"Could not find Rating score in Themis output: {text!r}"
                )
            return float(match.group(1))
        return self._parse_score(text)


VLLMAbsoluteGrader = VLLMTextScorer
