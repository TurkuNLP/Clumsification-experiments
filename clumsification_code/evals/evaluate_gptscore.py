"""
Evaluate a local GPTScore-style no-reference baseline on the same benchmark
suite used by evaluate_model_on_benchmark.py, so results are directly comparable.

This adapter supports arbitrary Hugging Face local / Hub checkpoints that can be
loaded as either:

  - AutoModelForCausalLM
  - AutoModelForSeq2SeqLM

No-reference GPTScore convention used here
──────────────────────────────────────────
For each candidate text x, return a scalar quality score:

  score(x) = - mean_token_NLL(x | no_reference_prompt)

Higher is better.

For decoder-only / causal LMs:

  input  = <no_reference_prompt> + <candidate text>
  labels = ignore prompt tokens, score only candidate tokens

For encoder-decoder / seq2seq LMs:

  encoder input = <no_reference_prompt>
  decoder label = <candidate text>

Important implementation detail
───────────────────────────────
The causal LM path defaults to an OPT/GPTScore-compatible tokenization mode that
is much closer to the original GPTScore repo and to the SummEval sanity check:

  full_ids   = tokenizer.encode(prompt + candidate)
  target_ids = tokenizer.encode(candidate), with a leading BOS/EOS stripped if present
  labels     = -100 everywhere except the final len(target_ids) positions

This reproduces the original style used by opt_score.py much more closely than
tokenizing prompt and candidate independently.

Expected layout:
- this file lives alongside evaluate_model_on_benchmark.py
- benchmark data files are in data/benchmarks/...

Example OPT-350M run:

python evaluate_gptscore_local_on_benchmark.py \
  --model-name GPTScore_OPT350M_NoRef \
  --training-dataset baseline \
  --perturbation-type none \
  --num-layers 0 \
  --context-length 1024 \
  --hf-model-name-or-path facebook/opt-350m \
  --model-type auto \
  --batch-size 8 \
  --max-input-length 1024 \
  --dtype float16 \
  --tp-plan none
"""

import argparse
import contextlib
import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers


# ──────────────────────────────────────────────────────────────────────
#  Distributed / rank helpers
# ──────────────────────────────────────────────────────────────────────

def setup_distributed() -> Dict[str, Any]:
    """
    Initialize torch.distributed when launched with torchrun.

    Returns rank/world/local_rank/device info.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": distributed,
        "device": device,
    }


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs) -> None:
    if is_rank0():
        print(*args, **kwargs)


@contextlib.contextmanager
def mute_non_rank0_output(rank: int):
    """
    Suppress stdout/stderr on nonzero ranks.

    This catches print() and most tqdm output from benchmark code without
    preventing nonzero ranks from participating in tensor-parallel forwards.
    """
    if rank == 0:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield


def patch_progress_bars_for_rank(rank: int) -> None:
    """
    Disable common progress bars/logging on nonzero ranks.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if rank != 0:
        os.environ["TQDM_DISABLE"] = "1"
        datasets.disable_progress_bars()
        transformers.utils.logging.set_verbosity_error()

    try:
        import tqdm as tqdm_pkg
        import tqdm.auto as tqdm_auto

        original_tqdm = tqdm_pkg.tqdm
        original_trange = tqdm_pkg.trange

        def rank_aware_tqdm(*args, **kwargs):
            if rank != 0:
                kwargs["disable"] = True
            return original_tqdm(*args, **kwargs)

        def rank_aware_trange(*args, **kwargs):
            if rank != 0:
                kwargs["disable"] = True
            return original_trange(*args, **kwargs)

        tqdm_pkg.tqdm = rank_aware_tqdm
        tqdm_pkg.trange = rank_aware_trange
        tqdm_auto.tqdm = rank_aware_tqdm
        tqdm_auto.trange = rank_aware_trange
    except Exception:
        pass


def patch_benchmark_module_for_rank0_only_output(bench_module, rank: int) -> None:
    """
    If evaluate_model_on_benchmark imported tqdm directly, patch its local tqdm.
    """
    if rank == 0:
        return

    for attr_name in ("tqdm", "trange"):
        if hasattr(bench_module, attr_name):
            original = getattr(bench_module, attr_name)

            def quiet_progress(*args, _original=original, **kwargs):
                kwargs["disable"] = True
                return _original(*args, **kwargs)

            setattr(bench_module, attr_name, quiet_progress)


# ──────────────────────────────────────────────────────────────────────
#  dtype helpers
# ──────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_torch_dtype(dtype_name: str) -> Union[str, torch.dtype]:
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {dtype_name!r}. Expected one of {sorted(_DTYPE_MAP)}."
        )
    return _DTYPE_MAP[dtype_name]


# ──────────────────────────────────────────────────────────────────────
#  No-reference GPTScore prompts
# ──────────────────────────────────────────────────────────────────────

"""
Prompt format:

Each entry has:

  {
      "instruction_prefix": "...",
      "prompt_text": "\n\nTl;dr",
  }

The actual no-reference prompt is:

  instruction_prefix + " " + prompt_text

This mirrors the sanity check's no-reference construction:

  source_side = " "
  prompt_text = "\n\nTl;dr"

or, for instruction-style SummEval:

  source_side = instruction_prefix + " "
  prompt_text = "\n\nTl;dr"

Then candidate is appended and only candidate tokens are scored.
"""

SUMMEVAL_ASPECTS = {
    "coherence": {
        "abbr": "COH",
        "instruction_prefix": "Generate a coherent summary for the following text: ",
    },
    "consistency": {
        "abbr": "CON",
        "instruction_prefix": "Generate factually consistent summary for the following text: ",
    },
    "fluency": {
        "abbr": "FLU",
        "instruction_prefix": "Generate a fluent and grammatical summary for the following text: ",
    },
    "relevance": {
        "abbr": "REL",
        "instruction_prefix": "Generate a relevant summary with consistent details for the following text: ",
    },
}


DEFAULT_NOREF_PROMPT = {
    "instruction_prefix": "Generate a fluent, coherent, grammatical, high-quality English text: ",
    "prompt_text": "",
}


GPTSCORE_NOREF_PROMPTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "default": {
        "quality": DEFAULT_NOREF_PROMPT,
        "overall": DEFAULT_NOREF_PROMPT,
        "fluency": {
            "instruction_prefix": "Generate a fluent and grammatical English text: ",
            "prompt_text": "",
        },
        "coherence": {
            "instruction_prefix": "Generate a coherent English text: ",
            "prompt_text": "",
        },
        "consistency": {
            "instruction_prefix": "Generate a factually consistent English text: ",
            "prompt_text": "",
        },
        "naturalness": {
            "instruction_prefix": "Generate a natural English text: ",
            "prompt_text": "",
        },
        "complexity": {
            "instruction_prefix": "Generate a complex and well-written English text: ",
            "prompt_text": "",
        },
    },

    # SummEval: keep the prompt wording from the sanity check / GPTScore table.
    "summeval": {
        "coherence": {
            "instruction_prefix": SUMMEVAL_ASPECTS["coherence"]["instruction_prefix"],
            "prompt_text": "\n\nTl;dr",
        },
        "consistency": {
            "instruction_prefix": SUMMEVAL_ASPECTS["consistency"]["instruction_prefix"],
            "prompt_text": "\n\nTl;dr",
        },
        "fluency": {
            "instruction_prefix": SUMMEVAL_ASPECTS["fluency"]["instruction_prefix"],
            "prompt_text": "\n\nTl;dr",
        },
        "relevance": {
            "instruction_prefix": SUMMEVAL_ASPECTS["relevance"]["instruction_prefix"],
            "prompt_text": "\n\nTl;dr",
        },
        # Vanilla no-reference SummEval, matching:
        #   source_side = " "
        #   prompt_text = "\n\nTl;dr"
        "vanilla": {
            "instruction_prefix": "",
            "prompt_text": "\n\nTl;dr",
        },
    },

    # Preference-style grammaticality / acceptability.
    "jfleg": {
        "fluency": {
            "instruction_prefix": "Generate a fluent and grammatical sentence: ",
            "prompt_text": "",
        },
        "grammar": {
            "instruction_prefix": "Generate a fluent and grammatical sentence: ",
            "prompt_text": "",
        },
    },
    "multiblimp": {
        "grammar": {
            "instruction_prefix": "Generate a fluent and grammatical sentence: ",
            "prompt_text": "",
        },
        "acceptability": {
            "instruction_prefix": "Generate an acceptable English sentence: ",
            "prompt_text": "",
        },
    },

    # Story / narrative preference.
    "story_cloze": {
        "coherence": {
            "instruction_prefix": "Generate a coherent and sensible story ending: ",
            "prompt_text": "",
        },
    },

    # Essay / writing quality.
    "ellipse": {
        "overall": {
            "instruction_prefix": "Generate a high-quality student essay: ",
            "prompt_text": "",
        },
        "cohesion": {
            "instruction_prefix": "Generate a cohesive student essay: ",
            "prompt_text": "",
        },
    },
    "argessay": {
        "language_mastery": {
            "instruction_prefix": "Generate an essay with strong language mastery: ",
            "prompt_text": "",
        },
        "complexity": {
            "instruction_prefix": "Generate a complex and well-written essay: ",
            "prompt_text": "",
        },
        "vocabulary": {
            "instruction_prefix": "Generate an essay with strong vocabulary usage: ",
            "prompt_text": "",
        },
        "language_constructs": {
            "instruction_prefix": "Generate an essay with strong language constructs: ",
            "prompt_text": "",
        },
    },

    # Dialogue.
    "topicalchat": {
        "overall": {
            "instruction_prefix": "Generate a high-quality dialogue response: ",
            "prompt_text": "",
        },
        "natural": {
            "instruction_prefix": "Generate a natural dialogue response: ",
            "prompt_text": "",
        },
    },
    "personachat": {
        "overall": {
            "instruction_prefix": "Generate a high-quality dialogue response: ",
            "prompt_text": "",
        },
        "natural": {
            "instruction_prefix": "Generate a natural dialogue response: ",
            "prompt_text": "",
        },
    },
    "fed_turn": {
        "fluency": {
            "instruction_prefix": "Generate a fluent dialogue response: ",
            "prompt_text": "",
        },
        "overall": {
            "instruction_prefix": "Generate a high-quality dialogue response: ",
            "prompt_text": "",
        },
    },
    "fed_whole": {
        "overall": {
            "instruction_prefix": "Generate a high-quality dialogue: ",
            "prompt_text": "",
        },
    },

    # Data-to-text / NLG.
    "webnlg": {
        "fluency": {
            "instruction_prefix": "Generate a fluent and grammatical data-to-text description: ",
            "prompt_text": "",
        },
    },
    "e2e": {
        "naturalness": {
            "instruction_prefix": "Generate a natural restaurant description: ",
            "prompt_text": "",
        },
        "quality": {
            "instruction_prefix": "Generate a high-quality restaurant description: ",
            "prompt_text": "",
        },
    },
    "humanratings": {
        "quality": {
            "instruction_prefix": "Generate a high-quality natural language generation output: ",
            "prompt_text": "",
        },
        "naturalness": {
            "instruction_prefix": "Generate a natural language generation output: ",
            "prompt_text": "",
        },
    },

    # Story generation.
    "openmeva": {
        "overall": {
            "instruction_prefix": "Generate a high-quality story continuation: ",
            "prompt_text": "",
        },
    },
    "hanna": {
        "coherence": {
            "instruction_prefix": "Generate a coherent story: ",
            "prompt_text": "",
        },
        "complexity": {
            "instruction_prefix": "Generate a complex and well-written story: ",
            "prompt_text": "",
        },
    },
}


def load_prompt_overrides(path: Optional[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load optional JSON prompt overrides.

    Expected JSON shape:

    {
      "summeval": {
        "fluency": {
          "instruction_prefix": "...",
          "prompt_text": "\\n\\nTl;dr"
        }
      },
      "my_benchmark": {
        "overall": {
          "instruction_prefix": "...",
          "prompt_text": ""
        }
      }
    }
    """
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("--prompt-config-json must contain a JSON object.")

    return obj


def merge_prompt_overrides(
    base: Dict[str, Dict[str, Dict[str, str]]],
    overrides: Dict[str, Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    merged = {
        task: {
            aspect: dict(prompt)
            for aspect, prompt in aspects.items()
        }
        for task, aspects in base.items()
    }

    for task, aspects in overrides.items():
        merged.setdefault(task, {})
        for aspect, prompt in aspects.items():
            merged[task][aspect] = dict(prompt)

    return merged


# ──────────────────────────────────────────────────────────────────────
#  GPTScore-style local HF inference model
# ──────────────────────────────────────────────────────────────────────

class LocalHFGPTScoreInferenceModel:
    """
    Adapter expected by evaluate_model_on_benchmark.py.

    Public method:
      score_texts(texts, device=None, batch_size=..., max_length=...) -> np.ndarray

    Returned scores are higher-is-better:

      -mean negative log-likelihood over candidate tokens.

    This class is intentionally no-reference: it scores only the candidate text
    conditioned on a task/aspect prompt. No source/reference text is passed.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        model_type: Literal["auto", "causal", "seq2seq"] = "auto",
        task_name: str = "default",
        aspect: str = "quality",
        prompt_table: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        prompt_template: Optional[str] = None,
        batch_size: int = 8,
        max_input_length: int = 1024,
        dtype: Union[str, torch.dtype] = "auto",
        device: Optional[torch.device] = None,
        device_map: Optional[str] = None,
        tp_plan: Optional[str] = "auto",
        trust_remote_code: bool = True,
        add_bos_token: bool = True,
        length_normalization: Literal["mean", "sum"] = "mean",
        original_causal_tokenization: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.model_type_arg = model_type

        self.task_name = task_name
        self.aspect = aspect
        self.prompt_table = prompt_table or GPTSCORE_NOREF_PROMPTS

        # Optional explicit global override. If not None, this wins over the
        # task/aspect prompt table. It must accept {aspect} and {task_name}.
        self.prompt_template = prompt_template

        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.tp_plan = tp_plan
        self.uses_tp = self.tp_plan not in {None, "", "none", "None"}
        self.trust_remote_code = trust_remote_code
        self.add_bos_token = add_bos_token
        self.length_normalization = length_normalization
        self.original_causal_tokenization = original_causal_tokenization

        if self.length_normalization not in {"mean", "sum"}:
            raise ValueError(
                f"length_normalization must be 'mean' or 'sum', got "
                f"{self.length_normalization!r}"
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        config = transformers.AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        if self.model_type_arg == "auto":
            self.is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
        elif self.model_type_arg == "seq2seq":
            self.is_seq2seq = True
        elif self.model_type_arg == "causal":
            self.is_seq2seq = False
        else:
            raise ValueError(
                f"model_type must be 'auto', 'causal', or 'seq2seq', got "
                f"{self.model_type_arg!r}"
            )

        if self.uses_tp and self.device_map is not None:
            raise ValueError(
                "Do not pass both tp_plan and device_map. "
                "Use tp_plan for tensor parallelism."
            )

        model_kwargs = {
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.uses_tp:
            model_kwargs["tp_plan"] = self.tp_plan
        elif self.device_map is not None:
            model_kwargs["device_map"] = self.device_map

        if self.is_seq2seq:
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs,
            )

        # Some decoder-only models, e.g. GPT-2, do not define a pad token.
        # For scoring-only inference, using EOS as PAD is standard and safe.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # With HF tensor parallelism or device_map, from_pretrained places/shards the model.
        # Calling .to(device) afterwards can break the sharded placement.
        if self.device_map is None and not self.uses_tp:
            self.model.to(self.device)

        self.model.eval()

        # With device_map="auto", the model may be sharded. Inputs should go to
        # the first parameter's device.
        self.input_device = next(self.model.parameters()).device

        rank0_print("Local GPTScore model loaded.")
        rank0_print(f"  model                    : {self.model_name_or_path}")
        rank0_print(f"  tokenizer                : {self.tokenizer_name_or_path}")
        rank0_print(f"  detected type            : {'seq2seq' if self.is_seq2seq else 'causal'}")
        rank0_print(f"  input device             : {self.input_device}")
        rank0_print(f"  dtype                    : {self.dtype}")
        rank0_print(f"  tp_plan                  : {self.tp_plan if self.uses_tp else None}")
        rank0_print(f"  device_map               : {self.device_map}")
        rank0_print(f"  max_input_length         : {self.max_input_length}")
        rank0_print(f"  length norm              : {self.length_normalization}")
        rank0_print(f"  original causal tokenize : {self.original_causal_tokenization}")
        rank0_print(f"  task/aspect              : {self.task_name!r}/{self.aspect!r}")
        rank0_print(f"  current prompt           : {self._render_prompt()!r}")

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        """
        Set the no-reference GPTScore prompt context used by subsequent
        score_texts() calls.
        """
        self.task_name = task_name
        self.aspect = aspect
        rank0_print(
            f"[GPTScore prompt] task={task_name!r}, aspect={aspect!r}, "
            f"prompt={self._render_prompt()!r}"
        )

    def _get_prompt_spec(self) -> Dict[str, str]:
        if self.prompt_template is not None:
            rendered = self.prompt_template.format(
                aspect=self.aspect,
                task_name=self.task_name,
            )
            return {
                "instruction_prefix": rendered,
                "prompt_text": "",
            }

        table = self.prompt_table

        if self.task_name in table and self.aspect in table[self.task_name]:
            return table[self.task_name][self.aspect]

        if self.task_name in table and "overall" in table[self.task_name]:
            return table[self.task_name]["overall"]

        if "default" in table and self.aspect in table["default"]:
            return table["default"][self.aspect]

        if "default" in table and "quality" in table["default"]:
            return table["default"]["quality"]

        return DEFAULT_NOREF_PROMPT

    def _render_prompt(self) -> str:
        """
        Render no-reference GPTScore prompt.

        Actual no-reference form:

          instruction_prefix + " " + prompt_text

        If instruction_prefix already ends with whitespace, this does not add
        another visible non-space token; it simply preserves the original
        no-reference "blank source" style.
        """
        spec = self._get_prompt_spec()
        instruction_prefix = spec.get("instruction_prefix", "")
        prompt_text = spec.get("prompt_text", "")

        # This is the no-reference source placeholder. It is intentionally a
        # single space to mirror the sanity check:
        #
        #   src = " "
        #   text = src + prompt_text + tgt
        #
        # For instruction prompts, this becomes:
        #
        #   instruction_prefix + " " + prompt_text
        #
        if instruction_prefix:
            source_side = instruction_prefix + " "
        else:
            source_side = " "

        return source_side + prompt_text

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""
        return str(text).strip()

    def _strip_leading_special_if_present(self, ids: List[int]) -> List[int]:
        """
        Original OPT GPTScore code uses:

          tgt_ids = tokenizer.encode(tgt)[1:]

        For generic tokenizers, blindly dropping index 0 can be wrong. So we
        drop the first token only when it is a likely auto-added BOS/EOS-style
        special token.
        """
        if not ids:
            return ids

        possible_leading_specials = {
            x for x in [
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                getattr(self.tokenizer, "cls_token_id", None),
                getattr(self.tokenizer, "decoder_start_token_id", None),
            ]
            if x is not None
        }

        if ids[0] in possible_leading_specials:
            return ids[1:]

        return ids

    # ──────────────────────────────────────────────────────────────
    #  Causal LM scoring
    # ──────────────────────────────────────────────────────────────

    def _encode_causal_one_original_style(
        self,
        text: str,
        max_length: int,
    ) -> Dict[str, List[int]]:
        """
        GPTScore/OPT-style causal encoding, matching the sanity-check logic.

        Original style:

          text = prompt + tgt
          input_ids = tokenizer.encode(text)
          tgt_ids = tokenizer.encode(tgt)[1:]
          output_ids = [-100] * len(input_ids)
          output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids

        Here prompt is always no-reference.
        """
        prompt = self._render_prompt()
        candidate = self._clean_text(text)

        if not candidate:
            candidate = (
                self.tokenizer.eos_token
                if self.tokenizer.eos_token is not None
                else self.tokenizer.pad_token
            )

        full_text = prompt + candidate

        input_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=False,
        )

        target_ids = self.tokenizer.encode(
            candidate,
            add_special_tokens=True,
            truncation=False,
        )
        target_ids = self._strip_leading_special_if_present(target_ids)

        if len(target_ids) == 0:
            fallback_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else self.tokenizer.pad_token_id
            )
            target_ids = [int(fallback_id)]

        max_length = int(max_length)
        if max_length < 2:
            raise ValueError("max_length must be at least 2 for causal scoring.")

        # Prefer preserving all target tokens. If the full sequence is too long,
        # remove tokens from the left/prompt side first.
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            target_len = len(target_ids)

            if target_len >= max_length:
                # Extremely long candidate: keep the first max_length target
                # tokens and score them. This is the only case where candidate
                # truncation is unavoidable.
                target_ids = target_ids[:max_length]
                input_ids = target_ids[:]
            else:
                # Reconstruct approximately from prompt + target so that the
                # target tail is preserved.
                prefix_len = max(len(input_ids) - target_len, 0)
                prefix_ids = input_ids[:prefix_len]
                keep_prefix = max_length - target_len
                prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []
                input_ids = prefix_ids + target_ids

        labels = [-100] * len(input_ids)
        labels[len(input_ids) - len(target_ids):] = target_ids

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode_causal_one_separate_prompt_target(
        self,
        text: str,
        max_length: int,
    ) -> Dict[str, List[int]]:
        """
        Generic causal LM scoring path.

        Kept as an optional fallback, but the original-style path above is the
        default because it matches GPTScore/OPT more closely.
        """
        prompt = self._render_prompt()
        candidate = self._clean_text(text)

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            candidate,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(target_ids) == 0:
            fallback_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else self.tokenizer.pad_token_id
            )
            target_ids = [int(fallback_id)]

        bos_ids: List[int] = []
        if self.add_bos_token and self.tokenizer.bos_token_id is not None:
            bos_ids = [int(self.tokenizer.bos_token_id)]

        max_length = int(max_length)
        if max_length < 2:
            raise ValueError("max_length must be at least 2 for causal scoring.")

        prefix_ids = bos_ids + prompt_ids
        max_prefix_len = max_length - 1
        if len(prefix_ids) > max_prefix_len:
            prefix_ids = prefix_ids[-max_prefix_len:]

        available_for_target = max_length - len(prefix_ids)
        target_ids = target_ids[:available_for_target]

        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode_causal_one(self, text: str, max_length: int) -> Dict[str, List[int]]:
        if self.original_causal_tokenization:
            return self._encode_causal_one_original_style(text, max_length)

        return self._encode_causal_one_separate_prompt_target(text, max_length)

    def _collate_causal(
        self,
        features: List[Dict[str, List[int]]],
    ) -> Dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for feat in features:
            n_pad = max_len - len(feat["input_ids"])
            input_ids.append(feat["input_ids"] + [pad_id] * n_pad)
            attention_mask.append(feat["attention_mask"] + [0] * n_pad)
            labels.append(feat["labels"] + [-100] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.input_device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.input_device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.input_device),
        }

    @torch.no_grad()
    def _score_causal_batch(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        all_scores: List[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            features = [
                self._encode_causal_one(text=t, max_length=max_length)
                for t in batch_texts
            ]
            batch = self._collate_causal(features)

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )

            logits = outputs.logits.float()
            labels = batch["labels"]

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in causal batch starting at {start}."
                )

            # HF causal LM loss predicts token t from logits at t-1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            vocab_size = shift_logits.size(-1)
            token_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)

            token_mask = shift_labels.ne(-100)
            nll_sum = (token_loss * token_mask).sum(dim=1)
            token_count = token_mask.sum(dim=1).clamp_min(1)

            if self.length_normalization == "mean":
                scores = -nll_sum / token_count
            else:
                scores = -nll_sum

            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"Non-finite GPTScore values detected in causal batch "
                    f"starting at {start}. Texts: {batch_texts[:3]}"
                )

            all_scores.append(scores.detach().cpu())

        return torch.cat(all_scores, dim=0).numpy().astype(np.float64)

    # ──────────────────────────────────────────────────────────────
    #  Seq2seq scoring
    # ──────────────────────────────────────────────────────────────

    def _encode_seq2seq_batch(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        prompt = self._render_prompt()
        sources = [prompt for _ in texts]
        targets = [self._clean_text(t) for t in texts]

        targets = [
            t if t else (
                self.tokenizer.eos_token
                if self.tokenizer.eos_token is not None
                else self.tokenizer.pad_token
            )
            for t in targets
        ]

        encoded_src = self.tokenizer(
            sources,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        encoded_tgt = self.tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoded_src["input_ids"].to(self.input_device)
        attention_mask = encoded_src["attention_mask"].to(self.input_device)
        labels = encoded_tgt["input_ids"].to(self.input_device)

        labels = labels.masked_fill(
            labels.eq(self.tokenizer.pad_token_id),
            -100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @torch.no_grad()
    def _score_seq2seq_batch(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        all_scores: List[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch = self._encode_seq2seq_batch(
                texts=batch_texts,
                max_length=max_length,
            )

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )

            logits = outputs.logits.float()
            labels = batch["labels"]

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in seq2seq batch starting at {start}."
                )

            vocab_size = logits.size(-1)
            token_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)

            token_mask = labels.ne(-100)
            nll_sum = (token_loss * token_mask).sum(dim=1)
            token_count = token_mask.sum(dim=1).clamp_min(1)

            if self.length_normalization == "mean":
                scores = -nll_sum / token_count
            else:
                scores = -nll_sum

            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"Non-finite GPTScore values detected in seq2seq batch "
                    f"starting at {start}. Texts: {batch_texts[:3]}"
                )

            all_scores.append(scores.detach().cpu())

        return torch.cat(all_scores, dim=0).numpy().astype(np.float64)

    # ──────────────────────────────────────────────────────────────
    #  Public benchmark adapter
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def raw_gptscore_scores(self, texts: List[str]) -> np.ndarray:
        """
        Return raw local GPTScore values.

        Higher is better because scores are negative NLL/log-loss.
        """
        return self.score_texts(
            texts=texts,
            batch_size=self.batch_size,
            max_length=self.max_input_length,
        )

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Adapter expected by evaluate_model_on_benchmark.py.

        `device` is ignored because the model owns its device/device_map.
        """
        if texts is None:
            return np.asarray([], dtype=np.float64)

        texts = [self._clean_text(t) for t in texts]
        effective_batch_size = int(batch_size or self.batch_size)
        effective_max_length = int(max_length or self.max_input_length)

        if self.is_seq2seq:
            return self._score_seq2seq_batch(
                texts=texts,
                batch_size=effective_batch_size,
                max_length=effective_max_length,
            )

        return self._score_causal_batch(
            texts=texts,
            batch_size=effective_batch_size,
            max_length=effective_max_length,
        )


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-reference local HF GPTScore baseline on the full "
            "benchmark suite and log JSONL results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output metadata.
    parser.add_argument("--model-name", type=str, default="GPTScore_LocalHF_NoRef")
    parser.add_argument("--training-dataset", type=str, default="baseline")
    parser.add_argument("--perturbation-type", type=str, default="none")
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=1024)

    # HF model config.
    parser.add_argument(
        "--hf-model-name-or-path",
        type=str,
        required=True,
        help="HF Hub id or local checkpoint path.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer path/name. Defaults to --hf-model-name-or-path.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "causal", "seq2seq"],
        help="Model family. 'auto' uses config.is_encoder_decoder.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=list(_DTYPE_MAP.keys()),
        help="Torch dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional HF accelerate device_map, e.g. 'auto'. If unset, model.to(device) is used.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders.",
    )
    parser.add_argument(
        "--no-add-bos-token",
        action="store_true",
        help="Disable manually prepending BOS for causal LMs in the non-original tokenization path.",
    )

    # GPTScore prompt / scoring config.
    parser.add_argument(
        "--aspect",
        type=str,
        default="quality",
        help=(
            "Initial aspect. The script changes this internally per benchmark; "
            "this only affects the sanity probe/default."
        ),
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="default",
        help=(
            "Initial task name. The script changes this internally per benchmark; "
            "this only affects the sanity probe/default."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Optional global prompt override. If supplied, this prompt is used "
            "for every benchmark instead of the task/aspect prompt table. "
            "Can use {aspect} and {task_name}."
        ),
    )
    parser.add_argument(
        "--prompt-config-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with per-task/per-aspect prompt overrides. "
            "See load_prompt_overrides() for expected shape."
        ),
    )
    parser.add_argument(
        "--length-normalization",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Use negative mean NLL or negative summed NLL. Mean matches the sanity-check loss.",
    )
    parser.add_argument(
        "--no-original-causal-tokenization",
        action="store_true",
        help=(
            "Disable GPTScore/OPT-style causal tokenization. By default the "
            "code tokenizes prompt+candidate together and labels the candidate "
            "tail, matching the original OPT scorer more closely."
        ),
    )

    parser.add_argument(
        "--tp-plan",
        type=str,
        default="auto",
        help=(
            "HF tensor-parallel plan, e.g. 'auto'. "
            "Use 'none' to disable tensor parallelism. "
            "Do not combine with --device-map."
        ),
    )

    # Inference settings.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=1024)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Main benchmark runner
# ──────────────────────────────────────────────────────────────────────

def score_scalar_aspect(
    *,
    bench,
    device,
    model: LocalHFGPTScoreInferenceModel,
    texts: List[str],
    labels: List[float],
    task_name: str,
    aspect: str,
    result_name: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    """
    Helper for scalar human-score benchmarks.

    Unlike the earlier generic version, this explicitly sets the GPTScore
    no-reference prompt per task/aspect before scoring.
    """
    model.set_prompt_context(task_name, aspect)

    raw_preds = bench.getModelPreds(
        device,
        model,
        texts,
        batch_size=batch_size,
        max_length=max_length,
    )

    return bench.correlation_bundle(labels, raw_preds, result_name)


def main():
    args = parse_args()

    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    device = dist_info["device"]

    patch_progress_bars_for_rank(rank)

    # Import after progress/logging patching.
    import clumsification_code.evals.evaluate_model_on_benchmark as bench
    patch_benchmark_module_for_rank0_only_output(bench, rank)

    dtype = parse_torch_dtype(args.dtype)

    prompt_overrides = load_prompt_overrides(args.prompt_config_json)
    prompt_table = merge_prompt_overrides(GPTSCORE_NOREF_PROMPTS, prompt_overrides)

    rank0_print(f"Rank/world size: {rank}/{world_size}")
    rank0_print(f"Local rank: {local_rank}")
    rank0_print(f"Device: {device}")
    rank0_print(f"HF GPTScore model: {args.hf_model_name_or_path}")
    rank0_print(f"Tokenizer: {args.tokenizer_name_or_path or args.hf_model_name_or_path}")
    rank0_print(f"Model type: {args.model_type}")
    rank0_print(f"dtype: {dtype}")
    rank0_print(f"tp_plan: {args.tp_plan}")
    rank0_print()

    model = LocalHFGPTScoreInferenceModel(
        model_name_or_path=args.hf_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        model_type=args.model_type,
        task_name=args.task_name,
        aspect=args.aspect,
        prompt_table=prompt_table,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        dtype=dtype,
        device=device,
        device_map=args.device_map,
        tp_plan=args.tp_plan,
        trust_remote_code=args.trust_remote_code,
        add_bos_token=not args.no_add_bos_token,
        length_normalization=args.length_normalization,
        original_causal_tokenization=not args.no_original_causal_tokenization,
    )

    # Sanity probe.
    model.set_prompt_context("default", "fluency")
    probe = [
        "This is a fluent, grammatical sentence.",
        "This sentence broken grammar bad.",
    ]
    probe_scores = model.score_texts(
        probe,
        batch_size=min(args.batch_size, 2),
        max_length=min(args.max_input_length, 512),
    )
    if rank == 0:
        rank0_print("GPTScore sanity probe scores, higher is better:", probe_scores)
        rank0_print()

    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_input_length

    all_results: Dict[str, Any] = {}

    # ==================================================================
    # Preference-style HF benchmarks
    # ==================================================================

    rank0_print("=" * 60)
    rank0_print(" Preference-style HF benchmarks")
    rank0_print("=" * 60)

    # ── JFLEG ─────────────────────────────────────────────────────
    model.set_prompt_context("jfleg", "grammar")
    jfleg_results = bench.eval_jfleg_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        for bench_name, metrics in jfleg_results.items():
            all_results.update(bench._flatten_preference_metrics(bench_name, metrics))

    # ── MultiBLiMP ────────────────────────────────────────────────
    model.set_prompt_context("multiblimp", "acceptability")
    multiblimp_metrics = bench.eval_multiblimp_english_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        all_results.update(
            bench._flatten_preference_metrics(
                "MultiBLiMP_eng_minimal_pair_preference",
                multiblimp_metrics,
            )
        )

    # ── Story Cloze ───────────────────────────────────────────────
    model.set_prompt_context("story_cloze", "coherence")
    storycloze_results = bench.eval_story_cloze_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        for bench_name, metrics in storycloze_results.items():
            all_results.update(bench._flatten_preference_metrics(bench_name, metrics))

    # ==================================================================
    # Scalar human-score benchmarks
    # ==================================================================

    rank0_print()
    rank0_print("=" * 60)
    rank0_print(" Scalar human-score benchmarks")
    rank0_print("=" * 60)

    # ── SummEval ──────────────────────────────────────────────────
    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]
    if rank == 0:
        rank0_print(f"\nSummEval texts: {len(summeval_texts)}")

    # Important: use aspect-specific SummEval prompts, matching the no-reference
    # sanity-check style.
    for aspect in ["fluency", "coherence", "consistency"]:
        labels = [x for y in ds[aspect] for x in y]
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=summeval_texts,
            labels=labels,
            task_name="summeval",
            aspect=aspect,
            result_name=f"summeval_{aspect}",
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── ELLIPSE ───────────────────────────────────────────────────
    ellipse_ds = bench.load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    if rank == 0:
        rank0_print(f"\nELLIPSE texts: {len(ellipse_ds)}")

    for aspect, result_name, label_key in [
        ("overall", "ellipse_overall", "overall"),
        ("cohesion", "ellipse_cohesion", "cohesion"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=ellipse_ds["text"],
            labels=ellipse_ds[label_key],
            task_name="ellipse",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── USR – Topical Chat ────────────────────────────────────────
    tc_texts, tc_overall_labels = bench.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Overall",
    )
    _, tc_natural_labels = bench.load_data_usr(
        "data/benchmarks/tc_usr_data.json",
        "Natural",
    )
    if rank == 0:
        rank0_print(f"\nTopicalChat texts: {len(tc_texts)}")

    for aspect, result_name, labels in [
        ("overall", "tc_overall", tc_overall_labels),
        ("natural", "tc_natural", tc_natural_labels),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=tc_texts,
            labels=labels,
            task_name="topicalchat",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── USR – Persona Chat ────────────────────────────────────────
    pc_texts, pc_overall_labels = bench.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Overall",
    )
    _, pc_natural_labels = bench.load_data_usr(
        "data/benchmarks/pc_usr_data.json",
        "Natural",
    )
    if rank == 0:
        rank0_print(f"\nPersonaChat texts: {len(pc_texts)}")

    for aspect, result_name, labels in [
        ("overall", "pc_overall", pc_overall_labels),
        ("natural", "pc_natural", pc_natural_labels),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=pc_texts,
            labels=labels,
            task_name="personachat",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── OpenMEVA ──────────────────────────────────────────────────
    meva_texts_roc, meva_labels_roc = bench.load_data_openmeva(
        "data/benchmarks/mans_roc.json"
    )
    meva_texts_wp, meva_labels_wp = bench.load_data_openmeva(
        "data/benchmarks/mans_wp.json"
    )
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    if rank == 0:
        rank0_print(f"\nOpenMEVA texts: {len(meva_texts)}")

    metrics = score_scalar_aspect(
        bench=bench,
        device=device,
        model=model,
        texts=meva_texts,
        labels=meva_labels,
        task_name="openmeva",
        aspect="overall",
        result_name="OpenMEVA_overall",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        all_results.update(metrics)

    # ── WebNLG ────────────────────────────────────────────────────
    webnlg_texts, webnlg_labels = bench.load_data_webnlg(
        "data/benchmarks/web_nlg_2020_human_evals_en.json"
    )
    if rank == 0:
        rank0_print(f"\nWebNLG texts: {len(webnlg_texts)}")

    metrics = score_scalar_aspect(
        bench=bench,
        device=device,
        model=model,
        texts=webnlg_texts,
        labels=webnlg_labels,
        task_name="webnlg",
        aspect="fluency",
        result_name="WebNLG_fluency",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        all_results.update(metrics)

    # ── HANNA ─────────────────────────────────────────────────────
    hanna_ds = bench.load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    if rank == 0:
        rank0_print(f"\nHANNA texts: {len(hanna_ds)}")

    for aspect, result_name, label_key in [
        ("coherence", "HANNA_coherence", "coherence"),
        ("complexity", "HANNA_complexity", "complexity"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=hanna_ds["text"],
            labels=hanna_ds[label_key],
            task_name="hanna",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── ARG-ESSAY ─────────────────────────────────────────────────
    arge_ds = bench.load_argessay_data("data/benchmarks/arg-essay.csv")
    if rank == 0:
        rank0_print(f"\nARG-ESSAY texts: {len(arge_ds)}")

    for aspect, result_name, label_key in [
        ("language_mastery", "ARG-ESSAY_language_mastery", "language_mastery"),
        ("complexity", "ARG-ESSAY_complexity", "complexity"),
        ("vocabulary", "ARG-ESSAY_vocabulary", "vocabulary"),
        ("language_constructs", "ARG-ESSAY_language_constructs", "language_constructs"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=arge_ds["text"],
            labels=arge_ds[label_key],
            task_name="argessay",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── Human Ratings of NLG ──────────────────────────────────────
    hr_ds = bench.load_human_ratings_of_nlg_data(
        "data/benchmarks/human_ratings_of_nlg.csv"
    )
    if rank == 0:
        rank0_print(f"\nHumanRatings texts: {len(hr_ds)}")

    for aspect, result_name, label_key in [
        ("quality", "HumanRatings_quality", "quality"),
        ("naturalness", "HumanRatings_naturalness", "naturalness"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=hr_ds["text"],
            labels=hr_ds[label_key],
            task_name="humanratings",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    # ── FED ───────────────────────────────────────────────────────
    turn_ds, whole_ds = bench.load_fed_data("data/benchmarks/fed_data.json")
    if rank == 0:
        rank0_print(f"\nFED turn-level texts: {len(turn_ds)}")

    for aspect, result_name, label_key in [
        ("fluency", "FED_turn_fluency", "fluent"),
        ("overall", "FED_turn_overall", "overall"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=turn_ds["text"],
            labels=turn_ds[label_key],
            task_name="fed_turn",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)

    if rank == 0:
        rank0_print(f"FED whole-dialogue texts: {len(whole_ds)}")

    metrics = score_scalar_aspect(
        bench=bench,
        device=device,
        model=model,
        texts=whole_ds["text"],
        labels=whole_ds["overall"],
        task_name="fed_whole",
        aspect="overall",
        result_name="FED_whole_overall",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    if rank == 0:
        all_results.update(metrics)

    # ── E2E ───────────────────────────────────────────────────────
    """
    e2e_ds = bench.load_e2e_data("data/benchmarks/E2E_data")
    if rank == 0:
        rank0_print(f"\nE2E texts: {len(e2e_ds)}")

    for aspect, result_name, label_key in [
        ("naturalness", "E2E_naturalness", "naturalness"),
        ("quality", "E2E_quality", "quality"),
    ]:
        metrics = score_scalar_aspect(
            bench=bench,
            device=device,
            model=model,
            texts=e2e_ds["text"],
            labels=e2e_ds[label_key],
            task_name="e2e",
            aspect=aspect,
            result_name=result_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        if rank == 0:
            all_results.update(metrics)
    """

    # ==================================================================
    # Write JSONL
    # ==================================================================
    if rank == 0:
        bench.write_results_jsonl(
            model_name=args.model_name,
            training_dataset=args.training_dataset,
            perturbation_type=args.perturbation_type,
            num_layers=args.num_layers,
            context_length=args.context_length,
            model_dir=args.hf_model_name_or_path,
            results=all_results,
        )


if __name__ == "__main__":
    main()