# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Command-line arguments for custom-dataset scoring."""

import argparse
from pathlib import Path

from clumsification_code.scoring.custom_dataset import (
    DEFAULT_BLEURT_CHECKPOINT,
    DEFAULT_METRICX_MODEL,
    DEFAULT_METRICX_TOKENIZER,
    DEFAULT_PPL_MODEL,
    DEFAULT_GEVAL_MODEL,
    SUPPORTED_SCORING_TYPES,
)
from clumsification_code.data.workflow_splitting import WORKFLOW_METHODS


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
    parser.add_argument("--scoring-run-id", default="default")
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
        "--scoring-chunk-size",
        type=int,
        default=1000,
        help="Number of candidates scored and checkpointed at a time.",
    )
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
    parser.add_argument(
        "--geval-cache-path",
        default=None,
        help="Legacy direct-request cache; unused by GPT-5.4-mini Batch scoring.",
    )
    parser.add_argument(
        "--geval-batch-size",
        type=int,
        default=10000,
        help="GPT-5.4-mini Batch requests per input file (default: 10000).",
    )
    parser.add_argument(
        "--geval-batch-action",
        choices=("prepare", "submit", "collect"),
        default="prepare",
        help="Prepare local Batch files, submit them, or collect finished results.",
    )
    parser.add_argument("--themis-model-name", default="PKU-ONELab/Themis")
    parser.add_argument("--themis-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--themis-max-model-len", type=int, default=None)
    parser.add_argument("--themis-max-tokens", type=int, default=512)
    parser.add_argument("--themis-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--themis-trust-remote-code", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for perplexity scoring, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=WORKFLOW_METHODS,
        default=None,
    )
    parser.add_argument("--perturbation-run-ids", nargs="+", default=None)
    parser.add_argument("--target-layers", nargs="+", type=int, default=None)
    parser.add_argument(
        "--source-partitions",
        nargs="+",
        default=None,
        help="Score only these split assignments; uses embedded partition metadata as fallback.",
    )
    parser.add_argument(
        "--reference-policy",
        choices=["original", "parent"],
        default="original",
    )
    parser.add_argument("--exclude-originals", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/custom_datasets"),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--overwrite", action="store_true")
    output_group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only errors from a completed score run, preserving its successful scores.",
    )
    parser.add_argument(
        "--retry-failed-max-retries",
        type=int,
        default=100,
        help="Additional retry rounds for unresolved entries in this job (default: 100).",
    )

    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive when supplied.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.scoring_chunk_size <= 0:
        parser.error("--scoring-chunk-size must be positive.")
    if not 1 <= args.geval_batch_size <= 50000:
        parser.error("--geval-batch-size must be between 1 and 50000.")
    if args.retry_failed_max_retries < 0:
        parser.error("--retry-failed-max-retries must be non-negative.")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2.")
    if args.metricx_max_input_length < 2:
        parser.error("--metricx-max-input-length must be at least 2.")
    if not args.scoring_run_id.strip():
        parser.error("--scoring-run-id must be non-empty.")
    if args.target_layers is not None and any(layer < 1 for layer in args.target_layers):
        parser.error("--target-layers values must be positive.")
    return args
