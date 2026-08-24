# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Command-line arguments for custom-dataset scoring."""

import argparse
from pathlib import Path

from clumsification_code.scoring.custom_dataset import (
    DEFAULT_BLEURT_CHECKPOINT,
    DEFAULT_METRICX_MODEL,
    DEFAULT_METRICX_TOKENIZER,
    DEFAULT_PPL_MODEL,
    SUPPORTED_SCORING_TYPES,
)


def parse_score_args():
    """Parse and validate arguments for ``scripts/score_custom_dataset.py``."""
    parser = argparse.ArgumentParser(
        description="Score custom-dataset perturbations for FE regression supervision."
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--scoring-type",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_SCORING_TYPES),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Number of original IDs to sample; scores every perturbation for each one.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Text language passed to Hugging Face Evaluate BERTScore.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_PPL_MODEL,
        help="Causal LM for perplexity scoring; ignored for BERTScore.",
    )
    parser.add_argument(
        "--bleurt-checkpoint",
        type=str,
        default=DEFAULT_BLEURT_CHECKPOINT,
        help="BLEURT checkpoint for BLEURT scoring; ignored by other methods.",
    )
    parser.add_argument("--metricx-model-name", default=DEFAULT_METRICX_MODEL)
    parser.add_argument("--metricx-tokenizer-name", default=DEFAULT_METRICX_TOKENIZER)
    parser.add_argument("--metricx-max-input-length", type=int, default=1536)
    parser.add_argument(
        "--gptscore-model-name",
        default=None,
        help="Local/Hugging Face model for source-aware GPTScore supervision.",
    )
    parser.add_argument("--gptscore-tokenizer-name", default=None)
    parser.add_argument(
        "--gptscore-model-type",
        choices=["auto", "causal", "seq2seq"],
        default="auto",
    )
    parser.add_argument(
        "--gptscore-source-prompt-template",
        default=None,
        help="Template containing {source}; candidate text is scored after it.",
    )
    parser.add_argument("--gptscore-device", default=None)
    parser.add_argument("--gptscore-device-map", default=None)
    parser.add_argument("--gptscore-dtype", default="auto")
    parser.add_argument("--gptscore-tp-plan", default="auto")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for perplexity scoring, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--layer-directory",
        type=str,
        default="perturbed_layers",
        help="Perturbation folder inside the selected custom dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/custom_datasets"),
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive when supplied.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2.")
    if args.metricx_max_input_length < 2:
        parser.error("--metricx-max-input-length must be at least 2.")
    return args
