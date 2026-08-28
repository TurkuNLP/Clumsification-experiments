# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Train a UniEval encoder or generative Boolean-QA evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.unieval.training import train_unieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=["encoder", "generative"], default="encoder")
    parser.add_argument("--pooling", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallelism", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument(
        "--fsdp-sharding-strategy",
        choices=["shard_grad_op", "full_shard"],
        default="shard_grad_op",
    )
    parser.add_argument("--fsdp-layer-cls", default=None)
    args = parser.parse_args()
    if args.parallelism == "fsdp" and not args.fsdp_layer_cls:
        parser.error("--fsdp-layer-cls is required when --parallelism fsdp.")
    train_unieval(**vars(args))


if __name__ == "__main__":
    main()
