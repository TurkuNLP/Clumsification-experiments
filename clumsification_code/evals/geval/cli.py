# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import argparse


def add_geval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geval-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI/OpenAI-compatible judge model used for G-Eval.",
    )
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
        help="Maximum output tokens for each judge response.",
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
        help="Maximum retries per API/parse call.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/evals/geval_cache.json",
        help="JSON cache path. Set to an empty string to disable cache.",
    )
    parser.add_argument(
        "--geval-score-min",
        type=float,
        default=1.0,
        help="Minimum possible G-Eval score.",
    )
    parser.add_argument(
        "--geval-score-max",
        type=float,
        default=5.0,
        help="Maximum possible G-Eval score.",
    )
    parser.add_argument(
        "--geval-concurrency",
        "--concurrency",
        dest="geval_concurrency",
        type=int,
        default=None,
        help=(
            "Number of concurrent G-Eval API requests. If omitted, the shared "
            "--batch-size value is used."
        ),
    )
    parser.add_argument(
        "--geval-response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
        help=(
            "Response-format mode. Use json_schema for OpenAI structured outputs; "
            "json_object or none may be useful for some OpenAI-compatible endpoints."
        ),
    )
    parser.add_argument(
        "--geval-task",
        type=str,
        default=None,
        help="Optional task key for task/aspect-specific rubric selection.",
    )
    parser.add_argument(
        "--geval-aspect",
        type=str,
        default=None,
        help="Optional aspect key for task/aspect-specific rubric selection.",
    )
