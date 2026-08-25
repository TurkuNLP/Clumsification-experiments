# This script has been co-created, refactored, and cleaned using GPT 5.6.

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EvalMetadata:
    model_name: str
    model_dir: str
    scorer: str
    training_dataset: str = ""
    perturbation_type: str = ""
    num_layers: int = -1
    context_length: int = -1
    protocol: str = ""
    rubric: str = ""


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return json_sanitize(float(x))
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def write_results_jsonl(
    metadata: EvalMetadata,
    results: Dict[str, Any],
    eval_dir: Path = Path("data/evals"),
    filename: Optional[str] = None,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_name = filename or f"{metadata.model_name}.jsonl"
    out_path = eval_dir / out_name

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **asdict(metadata),
    }
    record.update(results)

    with out_path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(json_sanitize(record), ensure_ascii=False) + "\n")

    print(f"\n✓ Results appended to {out_path}")
    return out_path
