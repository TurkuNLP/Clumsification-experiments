# This script has been co-created, refactored, and cleaned using GPT 5.6.
import logging
from typing import Any, Dict

import torch
import torch.nn as nn


LOGGER_NAME = "fsdp_fe"


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(LOGGER_NAME)


logger = logging.getLogger(LOGGER_NAME)


def tensor_debug_summary(t: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        t_float = t.float() if not t.is_floating_point() else t
        finite = torch.isfinite(t_float)

        summary = {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "finite": bool(finite.all().item()),
        }

        if t.is_floating_point():
            finite_values = t_float[finite]
            if finite_values.numel() > 0:
                summary.update(
                    {
                        "min": float(finite_values.min().item()),
                        "max": float(finite_values.max().item()),
                        "mean": float(finite_values.mean().item()),
                    }
                )

            summary["num_nan"] = int(torch.isnan(t_float).sum().item())
            summary["num_posinf"] = int(torch.isposinf(t_float).sum().item())
            summary["num_neginf"] = int(torch.isneginf(t_float).sum().item())

        return summary


def get_preferred_param_dtype() -> torch.dtype:
    """
    Prefer bf16 when available, otherwise fp16 on CUDA, otherwise fp32.

    FSDP requires parameters within a flattened handle to have uniform dtype.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def assert_uniform_floating_dtype(
    module: nn.Module,
    expected_dtype: torch.dtype,
    name: str = "model",
) -> None:
    bad = []

    for param_name, param in module.named_parameters():
        if param.is_floating_point() and param.dtype != expected_dtype:
            bad.append((param_name, param.dtype, tuple(param.shape)))

    for buffer_name, buffer in module.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != expected_dtype:
            bad.append((f"[buffer] {buffer_name}", buffer.dtype, tuple(buffer.shape)))

    if bad:
        preview = "\n".join(
            f"{n}: dtype={dt}, shape={shape}"
            for n, dt, shape in bad[:50]
        )
        raise RuntimeError(
            f"{name} has floating tensors not in expected dtype {expected_dtype}:\n"
            f"{preview}"
        )


def assert_finite_state_dict(state_dict: Dict[str, torch.Tensor], name: str):
    bad = []

    for k, v in state_dict.items():
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():
                finite_mask = torch.isfinite(v)
                num_bad = v.numel() - int(finite_mask.sum().item())
                bad.append((k, tuple(v.shape), str(v.dtype), num_bad))

    if bad:
        preview = "\n".join(
            f"{k}, shape={shape}, dtype={dtype}, nonfinite={num_bad}"
            for k, shape, dtype, num_bad in bad[:20]
        )
        raise FloatingPointError(
            f"Non-finite tensors found in {name}:\n{preview}"
        )


def strip_known_prefixes(key: str) -> str:
    prefixes = (
        "_orig_mod.",
        "module.",
        "_fsdp_wrapped_module.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key
