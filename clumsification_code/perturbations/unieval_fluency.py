# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""UniEval's token-span disfluency transformation.

The edit loop is transcribed from ``pseudo_data_summ.py`` in the official
UniEval repository at commit d33e7b6cfebe97b2bafe435adbd818230d5a416a.
Only the surrounding data adapter differs: this module accepts one arbitrary
text instead of extracting the first three sentences of CNN/DailyMail input.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


UNIEVAL_COMMIT = "d33e7b6cfebe97b2bafe435adbd818230d5a416a"
UNIEVAL_SOURCE = (
    "https://github.com/maszhongming/UniEval/blob/"
    f"{UNIEVAL_COMMIT}/pseudo_data_summ.py#L19-L57"
)


def apply_unieval_disfluency(
    text: str,
    *,
    n_noise: int = 1,
    python_rng: random.Random | Any | None = None,
    numpy_rng: np.random.Generator | None = None,
) -> tuple[str, list[dict]]:
    """Apply UniEval's insert/delete/shuffle loop to whitespace tokens."""
    if n_noise < 1:
        raise ValueError("n_noise must be at least 1")
    py_rng = python_rng or random
    np_rng = numpy_rng or np.random.default_rng()
    tokens = text.split()
    edits: list[dict] = []
    for _ in range(n_noise):
        target_len = len(tokens)
        if target_len == 0:
            break
        span_len = min(target_len, int(np_rng.poisson(5)))
        transform_type = py_rng.randint(1, 3)
        start_idx = py_rng.randint(0, target_len - span_len)
        edit = {
            "transform_type": {1: "insert", 2: "delete", 3: "shuffle"}[
                transform_type
            ],
            "span_len": span_len,
            "start_idx": start_idx,
        }
        if transform_type == 1:
            copy_idx = py_rng.randint(0, target_len - span_len)
            edit["copy_idx"] = copy_idx
            tokens = (
                tokens[:start_idx]
                + tokens[copy_idx : copy_idx + span_len]
                + tokens[start_idx:]
            )
        elif transform_type == 2:
            tokens = tokens[:start_idx] + tokens[start_idx + span_len :]
        else:
            shuffled_span = tokens[start_idx : start_idx + span_len]
            py_rng.shuffle(shuffled_span)
            tokens = tokens[:start_idx] + shuffled_span + tokens[start_idx + span_len :]
        edits.append(edit)
    return " ".join(tokens), edits
