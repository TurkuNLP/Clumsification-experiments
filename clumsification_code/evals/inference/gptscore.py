import builtins
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers


_DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


_ORIGINAL_PRINT = builtins.print
_PRINT_PATCH_INSTALLED = False


def parse_torch_dtype(dtype_name: str) -> Union[str, torch.dtype]:
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {dtype_name!r}. Expected one of {sorted(_DTYPE_MAP)}."
        )
    return _DTYPE_MAP[dtype_name]


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _dist_rank() -> int:
    return _int_env("RANK", 0)


def _dist_local_rank() -> int:
    return _int_env("LOCAL_RANK", 0)


def _dist_world_size() -> int:
    return _int_env("WORLD_SIZE", 1)


def _is_rank0() -> bool:
    return _dist_rank() == 0


def _option_enabled(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "false", "0", "no"}


def _install_rank0_print_filter() -> None:
    """
    Patch builtins.print so non-zero torchrun ranks do not duplicate benchmark
    output. This intentionally affects caller code after this module/model is
    imported/constructed, because the duplicate printing usually happens in the
    surrounding evaluation loop, not inside this class.

    Disable with:
        GPTSCORE_PRINT_ALL_RANKS=1
    """
    global _PRINT_PATCH_INSTALLED

    if _PRINT_PATCH_INSTALLED:
        return
    if _env_flag("GPTSCORE_PRINT_ALL_RANKS", default=False):
        return
    if _dist_world_size() <= 1:
        return

    def _rank0_print(*args, **kwargs):
        if _is_rank0():
            return _ORIGINAL_PRINT(*args, **kwargs)
        return None

    builtins.print = _rank0_print
    _PRINT_PATCH_INSTALLED = True


def _configure_rank_logging() -> None:
    """
    Keep stderr available for real exceptions, but quiet duplicated library logs
    and progress bars on non-zero ranks.
    """
    if _dist_world_size() <= 1 or _is_rank0():
        return

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logging.getLogger().setLevel(logging.ERROR)

    with contextlib_suppress_all():
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()

    with contextlib_suppress_all():
        datasets.utils.logging.set_verbosity_error()
        datasets.disable_progress_bars()


class contextlib_suppress_all:
    """Tiny local suppress context to avoid importing contextlib just for this."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True


def _select_local_device() -> torch.device:
    """
    Under torchrun, each process should use its LOCAL_RANK GPU.  This avoids all
    ranks accidentally allocating on cuda:0 before HF tensor parallel setup.
    """
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 0:
            idx = _dist_local_rank() % n
            torch.cuda.set_device(idx)
            return torch.device("cuda", idx)
    return torch.device("cpu")


def _maybe_init_distributed_for_tp(use_tp: bool) -> torch.device:
    """
    HF tensor parallelism expects a torch.distributed process group when running
    with torchrun.  All ranks must initialize the group and all ranks must run
    the same forward passes.
    """
    device = _select_local_device()

    if not use_tp:
        return device

    world_size = _dist_world_size()
    if world_size <= 1:
        return device

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available, but tp_plan was requested with "
            f"WORLD_SIZE={world_size}."
        )

    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"

        if backend == "nccl":
            try:
                dist.init_process_group(backend=backend, device_id=device)
            except TypeError:
                # Older PyTorch versions do not accept device_id.
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)

    return device


def _barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def build_prompt_table(prompt_config_json=None):
    overrides = load_prompt_overrides(prompt_config_json)
    return merge_prompt_overrides(GPTSCORE_NOREF_PROMPTS, overrides)


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
        "vanilla": {
            "instruction_prefix": "",
            "prompt_text": "\n\nTl;dr",
        },
    },
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
    "story_cloze": {
        "coherence": {
            "instruction_prefix": "Generate a coherent and sensible story ending: ",
            "prompt_text": "",
        },
    },
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
        task: {aspect: dict(prompt) for aspect, prompt in aspects.items()}
        for task, aspects in base.items()
    }

    for task, aspects in overrides.items():
        merged.setdefault(task, {})
        for aspect, prompt in aspects.items():
            merged[task][aspect] = dict(prompt)

    return merged


class LocalHFGPTScoreInferenceModel:
    """
    Adapter expected by evaluate_model_on_benchmark.py.

    Public method:
      score_texts(texts, device=None, batch_size=..., max_length=...) -> np.ndarray

    Returned scores are higher-is-better:
       -mean negative log-likelihood over candidate tokens.

    Multi-GPU behavior:
      - If launched with torchrun and tp_plan is enabled, this initializes
        torch.distributed and lets HF tensor parallelism shard the model.
      - All ranks participate in model forward passes.
      - Non-zero ranks are quiet by default to avoid duplicate terminal output.
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
        quiet_non_main_rank: bool = True,
    ):
        self.rank = _dist_rank()
        self.local_rank = _dist_local_rank()
        self.world_size = _dist_world_size()
        self.is_main_process = self.rank == 0

        if quiet_non_main_rank:
            _install_rank0_print_filter()
            _configure_rank_logging()

        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.model_type_arg = model_type

        self.task_name = task_name
        self.aspect = aspect
        self.prompt_table = prompt_table or GPTSCORE_NOREF_PROMPTS
        self.prompt_template = prompt_template

        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.dtype = dtype

        self.device_map = device_map
        self.requested_tp_plan = tp_plan
        self.tp_plan = tp_plan

        self.requested_tp = _option_enabled(tp_plan)
        self.uses_tp = self.requested_tp and self.world_size > 1

        if self.requested_tp and self.world_size <= 1:
            self.tp_plan = None

        if self.uses_tp and self.device_map is not None:
            raise ValueError(
                "Do not pass both tp_plan and device_map. "
                "Use tp_plan for tensor parallelism, or device_map for HF/Accelerate "
                "model placement, but not both."
            )

        dist_device = _maybe_init_distributed_for_tp(self.uses_tp)
        self.device = device or dist_device

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

        model_kwargs = {
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.uses_tp:
            model_kwargs["tp_plan"] = self.tp_plan
        elif self.device_map is not None:
            model_kwargs["device_map"] = self.device_map

        model_cls = (
            transformers.AutoModelForSeq2SeqLM
            if self.is_seq2seq
            else transformers.AutoModelForCausalLM
        )

        self.model = self._from_pretrained_dtype_compat(model_cls, model_kwargs)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # With HF TP or device_map, from_pretrained has already placed/sharded
        # the model. Calling .to(device) afterwards can break that placement.
        if self.device_map is None and not self.uses_tp:
            self.model.to(self.device)

        self.model.eval()

        self.input_device = self._infer_input_device()

        _barrier_if_distributed()

        if self.is_main_process and _env_flag("GPTSCORE_DEBUG_DISTRIBUTED", False):
            _ORIGINAL_PRINT(
                "[gptscore] distributed debug:",
                {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "world_size": self.world_size,
                    "uses_tp": self.uses_tp,
                    "requested_tp_plan": self.requested_tp_plan,
                    "effective_tp_plan": self.tp_plan,
                    "device": str(self.device),
                    "input_device": str(self.input_device),
                    "device_map": self.device_map,
                    "is_seq2seq": self.is_seq2seq,
                    "dtype": str(self.dtype),
                    "dist_initialized": dist.is_available() and dist.is_initialized(),
                },
            )

    def _from_pretrained_dtype_compat(self, model_cls, model_kwargs):
        """
        Newer Transformers versions use `dtype`; older versions use
        `torch_dtype`. Try `dtype` first, then fall back if necessary.
        """
        try:
            return model_cls.from_pretrained(
                self.model_name_or_path,
                **model_kwargs,
            )
        except TypeError as e:
            msg = str(e)
            if "dtype" not in model_kwargs or "dtype" not in msg:
                raise

            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            return model_cls.from_pretrained(
                self.model_name_or_path,
                **fallback_kwargs,
            )

    def _infer_input_device(self) -> torch.device:
        if self.uses_tp:
            return self.device

        if self.device_map is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return self.device

        return self.device

    def set_prompt_context(self, task_name: str, aspect: str) -> None:
        self.task_name = task_name
        self.aspect = aspect

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
        spec = self._get_prompt_spec()
        instruction_prefix = spec.get("instruction_prefix", "")
        prompt_text = spec.get("prompt_text", "")

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
        if not ids:
            return ids

        possible_leading_specials = {
            x
            for x in [
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
    # Causal LM scoring
    # ──────────────────────────────────────────────────────────────

    def _encode_causal_one_original_style(
        self,
        text: str,
        max_length: int,
    ) -> Dict[str, List[int]]:
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

        if len(input_ids) > max_length:
            target_len = len(target_ids)

            if target_len >= max_length:
                target_ids = target_ids[:max_length]
                input_ids = target_ids[:]
            else:
                prefix_len = max(len(input_ids) - target_len, 0)
                prefix_ids = input_ids[:prefix_len]
                keep_prefix = max_length - target_len
                prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []
                input_ids = prefix_ids + target_ids

        labels = [-100] * len(input_ids)
        labels[len(input_ids) - len(target_ids) :] = target_ids

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
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.input_device,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
                device=self.input_device,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
                device=self.input_device,
            ),
        }

    @torch.no_grad()
    def _score_causal_batch(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float64)

        all_scores: List[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
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
            labels = batch["labels"].to(logits.device)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in causal batch starting at {start}."
                )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            vocab_size = shift_logits.size(-1)
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
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
    # Seq2seq scoring
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
            t
            if t
            else (
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
        if not texts:
            return np.asarray([], dtype=np.float64)

        all_scores: List[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
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
            labels = batch["labels"].to(logits.device)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in seq2seq batch starting at {start}."
                )

            vocab_size = logits.size(-1)
            token_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
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
    # Public benchmark adapter
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def raw_gptscore_scores(self, texts: List[str]) -> np.ndarray:
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
        Under tensor parallelism, all ranks must call this method with the same
        batches in the same order.
        """
        if texts is None:
            return np.asarray([], dtype=np.float64)

        texts = [self._clean_text(t) for t in texts]
        if len(texts) == 0:
            return np.asarray([], dtype=np.float64)

        effective_batch_size = int(batch_size or self.batch_size)
        effective_max_length = int(max_length or self.max_input_length)

        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {effective_batch_size}.")
        if effective_max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {effective_max_length}."
            )

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