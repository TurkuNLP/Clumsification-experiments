# This script has been co-created, refactored, and cleaned using GPT 5.6.
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
from typing import Optional

import torch

from clumsification_code.evals.benchmark_runner import (
    run_external_dev_suite,
    run_formatted_dataset_suite,
    run_standard_benchmark_suite,
)
from clumsification_code.evals.inference.fe import load_fe_inference_model
from clumsification_code.evals.nlg_eval_loader import DEFAULT_NLG_EVAL_PATH
from clumsification_code.evals.result_writer import EvalMetadata, write_results_jsonl
from clumsification_code.evals.standalone_benchmarks import (
    DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH,
    DEFAULT_COHESENTIA_PATH,
    DEFAULT_COHESENTIA_TRAIN_PATH,
    DEFAULT_ELLIPSE_PATH,
    DEFAULT_ELLIPSE_TRAIN_PATH,
)

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

#Parse information from the DS name
def parse_evaluation_run_name(name:str):
    num_layers = 5
    pert_type = "clumsy"

    #Getting model name
    model_name=name[:name.find('_')]
    name=name[name.find('_')+1:]
    #Parsing the language info
    language=name[:name.find('_')]
    name=name[name.find('_')+1:]
    #Parsing num_layers and pert_type
    training_ds_name=name
    if name[name.rfind('_')+1].isnumeric():
        pert_type = name[:name.rfind('_')]
        num_layers = name[name.rfind('_')+1:]
    else:
        pert_type = name
    return model_name, language, pert_type, num_layers, training_ds_name


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evaluation-role",
        default="final",
        choices=["final", "external-dev", "formatted-dataset"],
        help=(
            "Run the untouched final suite or the separate human-labeled "
            "non-test checkpoint-selection panel, or a selected split from a saved formatted dataset."
        ),
    )
    parser.add_argument(
        "--scorer",
        required=True,
        choices=["fe", "gptscore", "metricx", "geval", "vllm", "unieval", "unieval-trained", "ppl"],
        help="Evaluation scorer backend.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--unieval-base-model", default="")
    parser.add_argument("--training-dataset", default="")
    parser.add_argument("--perturbation-type", default="")
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=-1)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Scorer batch size; for vLLM, only caps retry batches after the full initial submission.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--ppl-data-parallel-size", type=int, default=1,
        help="Independent PPL model replicas; each uses one GPU.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=list(_DTYPE_MAP),
        help="Torch dtype for local torch-backed scorers.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation for local FE models.",
    )
    # Keep benchmark paths configurable while making the repository defaults
    # explicit for reproducible FE and direct-evaluator runs.
    parser.add_argument("--nlg-eval-path", default=str(DEFAULT_NLG_EVAL_PATH))
    parser.add_argument("--ellipse-path", default=str(DEFAULT_ELLIPSE_PATH))
    parser.add_argument(
        "--human-chatgpt-essays-path",
        default=str(DEFAULT_HUMAN_CHATGPT_ESSAYS_PATH),
        help="Herbold et al. (2023) human/ChatGPT essay-comparison CSV.",
    )
    parser.add_argument("--cohesentia-path", default=str(DEFAULT_COHESENTIA_PATH))
    parser.add_argument(
        "--external-dev-ellipse-path",
        default=str(DEFAULT_ELLIPSE_TRAIN_PATH),
        help="Audited official ELLIPSE train partition used only for external development.",
    )
    parser.add_argument(
        "--external-dev-cohesentia-path",
        default=str(DEFAULT_COHESENTIA_TRAIN_PATH),
        help="Released CoheSentia train pool used only for external development.",
    )
    parser.add_argument(
        "--include-dev-story-cloze-diagnostic",
        action="store_true",
        help=(
            "Also evaluate Story Cloze train as a secondary diagnostic. Its "
            "result is excluded from checkpoint selection."
        ),
    )
    parser.add_argument(
        "--skip-preferences",
        action="store_true",
        help="Skip JFLEG, MultiBLiMP, and Story Cloze preference evaluation.",
    )
    parser.add_argument(
        "--skip-multilingual",
        action="store_true",
        help="Skip BASSE and Norwegian multilingual evaluation.",
    )
    parser.add_argument(
        "--max-records-per-dimension",
        type=int,
        default=None,
        help="Limit each filtered scalar dimension for a quick local test.",
    )
    parser.add_argument(
        "--formatted-dataset-path",
        default="",
        help="Path to a saved formatted Hugging Face DatasetDict.",
    )
    parser.add_argument(
        "--formatted-dataset-split",
        default="test",
        choices=["train", "dev", "test"],
        help="Split to evaluate for formatted-dataset evaluation.",
    )
    parser.add_argument(
        "--training-method",
        default="regression",
        choices=["regression", "pairwise", "binary"],
        help="Objective used to interpret the formatted dataset split.",
    )
    parser.add_argument(
        "--score-name",
        default="",
        help="Score field for formatted regression evaluation.",
    )
    parser.add_argument(
        "--pair-policy",
        default="all_unequal_layers",
        choices=["original_only", "all_unequal_layers"],
        help="Pair construction policy for formatted pairwise evaluation.",
    )


def add_gptscore_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-model-name-or-path", default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--model-type", default="auto", choices=["auto", "causal", "seq2seq"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--tp-plan", default="auto")
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--prompt-config-json", default=None)
    parser.add_argument("--length-normalization", default="mean", choices=["mean", "sum"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-original-causal-tokenization", action="store_true")

def add_metricx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metricx-model-name-or-path", default=None)
    parser.add_argument("--tokenizer", default="google/mt5-xl")


def add_unieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--unieval-repo",
        default=None,
        help="Path to the official UniEval checkout containing metric/evaluator.py.",
    )
    parser.add_argument(
        "--unieval-cache-dir",
        default=None,
        help="Optional cache directory for the UniEval Hugging Face checkpoint.",
    )


def add_prometheus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vllm-model-name-or-path", default=None)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vllm-data-parallel-size", type=int, default=1,
        help="Independent vLLM model replicas; total GPUs used is DP × TP.",
    )
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-max-tokens", type=int, default=512)
    parser.add_argument("--vllm-temperature", type=float, default=0.0)
    parser.add_argument("--vllm-enable-thinking", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-trust-remote-code", action="store_true")
    parser.add_argument("--vllm-protocol", default="prometheus_direct_assessment.json")
    parser.add_argument("--vllm-rubric", default="menlo_fluency.json")


def build_scorer(args: argparse.Namespace, device: torch.device):
    dtype = _DTYPE_MAP[args.dtype]

    if args.scorer == "fe":
        if not args.model_dir:
            raise ValueError("--model-dir is required with --scorer fe")
        return load_fe_inference_model(
            model_dir=args.model_dir,
            device=device,
            attn_implementation=args.attn_implementation,
            dtype=dtype,
        )

    if args.scorer == "gptscore":
        from clumsification_code.evals.inference.gptscore import (
            LocalHFGPTScoreInferenceModel,
            build_prompt_table,
        )

        hf_path = args.hf_model_name_or_path or args.model_dir
        if not hf_path:
            raise ValueError("--hf-model-name-or-path or --model-dir is required with --scorer gptscore")

        return LocalHFGPTScoreInferenceModel(
            model_name_or_path=hf_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            model_type=args.model_type,
            batch_size=args.batch_size,
            max_input_length=args.max_length,
            dtype=dtype,
            device=device,
            device_map=args.device_map,
            tp_plan=args.tp_plan,
            trust_remote_code=args.trust_remote_code,
            prompt_template=args.prompt_template,
            prompt_table=build_prompt_table(args.prompt_config_json),
            original_causal_tokenization=not args.no_original_causal_tokenization,
            length_normalization=args.length_normalization,
        )

    if args.scorer == "metricx":
        from clumsification_code.evals.inference.metricx import MetricX24QEInferenceModel

        if not args.metricx_model_name_or_path:
            raise ValueError("--metricx-model-name-or-path is required with --scorer metricx")

        return MetricX24QEInferenceModel(
            model_name_or_path=args.metricx_model_name_or_path,
            tokenizer_name=args.tokenizer,
            batch_size=args.batch_size,
            max_input_length=args.max_length,
        )

    if args.scorer == "geval":
        from clumsification_code.evals.geval.scorer import GEvalScorer

        return GEvalScorer.from_args(args)

    if args.scorer == "vllm":
        model_path = args.vllm_model_name_or_path or args.model_dir
        if not model_path:
            raise ValueError("--vllm-model-name-or-path or --model-dir is required with --scorer vllm")
        if args.vllm_data_parallel_size < 1 or args.vllm_tensor_parallel_size < 1:
            raise ValueError("vLLM data and tensor parallel sizes must be positive")
        if args.vllm_data_parallel_size > 1:
            from clumsification_code.evals.inference.vllm_parallel import ParallelVLLMTextScorer

            scorer_type = ParallelVLLMTextScorer
            parallel_kwargs = {"data_parallel_size": args.vllm_data_parallel_size}
        else:
            from clumsification_code.evals.inference.vllm_scorer import VLLMTextScorer

            scorer_type = VLLMTextScorer
            parallel_kwargs = {}
        return scorer_type(
            model_path,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            max_model_len=args.vllm_max_model_len,
            max_tokens=args.vllm_max_tokens,
            temperature=args.vllm_temperature,
            enable_thinking=args.vllm_enable_thinking,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            trust_remote_code=args.vllm_trust_remote_code,
            protocol=args.vllm_protocol,
            rubric=args.vllm_rubric,
            task="fluency",
            aspect="fluency",
            **parallel_kwargs,
        )

    if args.scorer == "unieval":
        from clumsification_code.evals.inference.unieval import load_unieval_fluency_model

        return load_unieval_fluency_model(
            repo_path=args.unieval_repo,
            max_length=args.max_length,
            device=device,
            cache_dir=args.unieval_cache_dir,
        )

    if args.scorer == "unieval-trained":
        from clumsification_code.evals.inference.unieval_trained import load_trained_unieval_model
        if not args.model_dir:
            raise ValueError("--model-dir is required with --scorer unieval-trained")
        return load_trained_unieval_model(args.model_dir, device=device, dtype=dtype)

    if args.scorer == "ppl":
        if not args.hf_model_name_or_path:
            raise ValueError("--hf-model-name-or-path is required with --scorer ppl")
        if args.ppl_data_parallel_size < 1:
            raise ValueError("--ppl-data-parallel-size must be positive")
        scorer_kwargs = dict(
            model_name_or_path=args.hf_model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
        )
        if args.ppl_data_parallel_size > 1:
            from clumsification_code.evals.inference.hf_ppl_parallel import ParallelHFPPLScorer

            return ParallelHFPPLScorer(
                data_parallel_size=args.ppl_data_parallel_size, **scorer_kwargs
            )
        from clumsification_code.evals.inference.hf_ppl import load_hf_ppl_model

        return load_hf_ppl_model(device=device, **scorer_kwargs)

    raise ValueError(f"Unsupported scorer: {args.scorer}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation with a selectable scorer backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_gptscore_args(parser)
    add_metricx_args(parser)
    add_unieval_args(parser)
    add_prometheus_args(parser)

    # G-Eval parser can extend this if needed.
    try:
        from clumsification_code.evals.geval.cli import add_geval_args

        add_geval_args(parser)
    except Exception:
        pass

    return parser.parse_args(argv)


def run_selected_suite(
    args: argparse.Namespace,
    model,
    device: torch.device,
):
    """Run the CLI-selected suite with an already constructed scorer."""
    if args.evaluation_role == "external-dev":
        return run_external_dev_suite(
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            ellipse_path=args.external_dev_ellipse_path,
            cohesentia_path=args.external_dev_cohesentia_path,
            include_story_cloze_diagnostic=args.include_dev_story_cloze_diagnostic,
            max_records_per_dimension=args.max_records_per_dimension,
        )
    elif args.evaluation_role == "formatted-dataset":
        if not args.formatted_dataset_path:
            raise ValueError(
                "--formatted-dataset-path is required with "
                "--evaluation-role formatted-dataset"
            )
        if args.training_method == "regression" and not args.score_name:
            raise ValueError(
                "--score-name is required for formatted regression evaluation"
            )
        return run_formatted_dataset_suite(
            model=model,
            device=device,
            dataset_path=args.formatted_dataset_path,
            split=args.formatted_dataset_split,
            training_method=args.training_method,
            score_name=args.score_name or None,
            pair_policy=args.pair_policy,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_records=args.max_records_per_dimension,
        )
    return run_standard_benchmark_suite(
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        nlg_eval_path=args.nlg_eval_path,
        ellipse_path=args.ellipse_path,
        human_chatgpt_essays_path=args.human_chatgpt_essays_path,
        cohesentia_path=args.cohesentia_path,
        skip_preferences=args.skip_preferences,
        max_records_per_dimension=args.max_records_per_dimension,
        include_multilingual=not args.skip_multilingual,
    )


def _run_geval_batch(
    args: argparse.Namespace,
    device: torch.device,
):
    from clumsification_code.evals.geval.benchmark_batch import (
        BatchGEvalScorer,
        run_benchmark_batch_action,
    )
    from clumsification_code.evals.geval.prompts import GEVAL_QE_PROMPT_VERSION

    scorer = BatchGEvalScorer.from_args(args)
    # Traversal calculates placeholder metrics while collecting requests. Hide
    # those deliberately discarded values from the user-facing output.
    with redirect_stdout(io.StringIO()):
        run_selected_suite(args, scorer, device)
    batch_result = run_benchmark_batch_action(
        scorer=scorer,
        state_root=Path(args.geval_batch_state_root),
        run_id=args.geval_batch_run_id or args.model_name,
        batch_size=args.geval_batch_size,
        action=args.geval_batch_action,
    )
    status = batch_result["status"]
    print(
        f"G-Eval Batch {status}: {batch_result.get('state_path', '')}",
        flush=True,
    )
    if status == "prepared":
        print(
            f"  {batch_result['num_logical_requests']} distinct judgments, "
            f"{batch_result['num_requests']} API requests in "
            f"{batch_result['num_batches']} files.",
            flush=True,
        )
        return scorer, None, batch_result
    if status == "pending":
        print(f"  Batch statuses: {batch_result['batch_statuses']}", flush=True)
        if batch_result.get("queue_limited"):
            print(
                "  The active Batch queue is full. Completed chunks remain saved; "
                "rerun submit after queued jobs finish.",
                flush=True,
            )
        return scorer, None, batch_result
    if status == "completed_with_errors":
        print(
            f"  {batch_result['num_failures']} requests need retry; details: "
            f"{batch_result['failure_path']}",
            flush=True,
        )
        return scorer, None, batch_result
    if batch_result.get("results_written"):
        print(
            f"  Final metrics were already written to {batch_result['results_path']}",
            flush=True,
        )
        return scorer, None, batch_result
    scorer.set_collected_scores(batch_result["scores"])
    results = run_selected_suite(args, scorer, device)
    results.update(
        {
            "geval__processing": "openai_batch",
            "geval__batch_run_id": args.geval_batch_run_id or args.model_name,
            "geval__batch_ids": batch_result["batch_ids"],
            "geval__model": args.geval_model,
            "geval__prompt_version": GEVAL_QE_PROMPT_VERSION,
            "geval__max_output_tokens": args.max_output_tokens,
            "geval__n_samples": args.n_samples,
        }
    )
    return scorer, results, batch_result


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_result = None
    if args.scorer == "geval" and args.geval_processing == "batch":
        model, results, batch_result = _run_geval_batch(args, device)
        if results is None:
            return
    else:
        model = build_scorer(args, device)
        results = run_selected_suite(args, model, device)

    model_dir = (
        args.model_dir
        or getattr(args, "hf_model_name_or_path", "")
        or getattr(args, "metricx_model_name_or_path", "")
        or getattr(args, "vllm_model_name_or_path", "")
        or getattr(args, "unieval_repo", "")
    )

    if args.scorer == "fe":
        if args.evaluation_role in {"external-dev", "formatted-dataset"}:
            model_name = args.model_name
            training_dataset = (
                args.formatted_dataset_path
                if args.evaluation_role == "formatted-dataset"
                else args.training_dataset
            )
            pert_type = args.perturbation_type
            num_layers = args.num_layers
        else:
            model_name, language, pert_type, num_layers, training_ds_name = parse_evaluation_run_name(args.model_name)
            training_dataset = language + "/" + training_ds_name

        metadata = EvalMetadata(
            model_name=model_name,
            model_dir=model_dir,
            scorer=args.scorer,
            training_dataset=training_dataset,
            perturbation_type=pert_type,
            num_layers=num_layers,
            context_length=args.context_length,
            evaluation_tracks=(
                args.evaluation_role
                if args.evaluation_role != "final"
                else ("english" if args.skip_multilingual else "english,multilingual")
            ),
            evaluation_role=args.evaluation_role,
            formatted_dataset_path=(
                args.formatted_dataset_path
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            formatted_dataset_split=(
                args.formatted_dataset_split
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            training_method=(
                args.training_method
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            score_name=(
                args.score_name
                if args.evaluation_role == "formatted-dataset" else ""
            ),
        )

    else:
        protocol = getattr(model, "protocol", "") if args.scorer in {"vllm", "geval"} else ""
        rubric = getattr(model, "rubric", "") if args.scorer in {"vllm", "geval"} else ""
        metadata = EvalMetadata(
            model_name=args.model_name,
            model_dir=model_dir,
            scorer=args.scorer,
            training_dataset="none",
            perturbation_type="none",
            num_layers=0,
            context_length=args.context_length,
            protocol=protocol,
            rubric=rubric,
            evaluation_tracks=(
                args.evaluation_role
                if args.evaluation_role != "final"
                else ("english" if args.skip_multilingual else "english,multilingual")
            ),
            evaluation_role=args.evaluation_role,
            formatted_dataset_path=(
                args.formatted_dataset_path
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            formatted_dataset_split=(
                args.formatted_dataset_split
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            training_method=(
                args.training_method
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            score_name=(
                args.score_name
                if args.evaluation_role == "formatted-dataset" else ""
            ),
            vllm_data_parallel_size=(
                args.vllm_data_parallel_size if args.scorer == "vllm" else 0
            ),
            vllm_tensor_parallel_size=(
                args.vllm_tensor_parallel_size if args.scorer == "vllm" else 0
            ),
            ppl_data_parallel_size=(
                args.ppl_data_parallel_size if args.scorer == "ppl" else 0
            ),
        )

    eval_dir = (
        "data/evals/external_dev"
        if args.evaluation_role == "external-dev"
        else (
            "data/evals/formatted"
            if args.evaluation_role == "formatted-dataset"
            else "data/evals/final"
        )
    )
    results_path = write_results_jsonl(
        metadata=metadata, results=results, eval_dir=Path(eval_dir)
    )
    if batch_result is not None:
        from clumsification_code.evals.geval.benchmark_batch import (
            mark_benchmark_batch_results_written,
        )

        mark_benchmark_batch_results_written(
            Path(batch_result["state_path"]), results_path
        )
    close_model = getattr(model, "close", None)
    if callable(close_model):
        close_model()


if __name__ == "__main__":
    main()
