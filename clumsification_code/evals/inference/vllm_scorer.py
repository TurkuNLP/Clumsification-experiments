# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""General vLLM-backed absolute-grading scorer."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from clumsification_code.evals.geval.prompts import render_rubric, rubric_for
from clumsification_code.evals.geval.parser import parse_score_response
from clumsification_code.prompts import load_prompt_data, load_prompt_spec


_RESULT_RE = re.compile(
    r"(?:\[\s*RESULT\s*\]|\bRESULT\b)\s*:?\s*([1-5])\b",
    flags=re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


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
        self.enable_thinking = enable_thinking
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_kwargs: Dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task = task_name
        self.aspect = aspect

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
            raise ValueError(f"Could not find [RESULT] score in vLLM output: {text[:300]!r}")
        return float(match.group(1))

    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        del device, batch_size, max_length
        prompts = [self._messages(text) for text in texts]
        outputs = self.llm.chat(
            prompts,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        return np.asarray(
            [self._parse_output(output.outputs[0].text) for output in outputs],
            dtype=np.float32,
        )

    def _parse_output(self, text: str) -> float:
        if self.output_parser == "json_score":
            visible_text = _THINK_BLOCK_RE.sub("", text or "").strip()
            return parse_score_response(visible_text).score
        return self._parse_score(text)


VLLMAbsoluteGrader = VLLMTextScorer
