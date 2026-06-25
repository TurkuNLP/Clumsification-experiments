from __future__ import annotations

import argparse
from typing import Optional

import torch

from clumsification_code.evals.benchmark_runner import run_standard_benchmark_suite
from clumsification_code.evals.inference.ltr import load_ltr_inference_model
from clumsification_code.evals.result_writer import EvalMetadata, write_results_jsonl

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scorer",
        required=True,
        choices=["ltr", "gptscore", "metricx", "geval"],
        help="Evaluation scorer backend.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--training-dataset", default="")
    parser.add_argument("--perturbation-type", default="")
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=list(_DTYPE_MAP),
        help="Torch dtype for local torch-backed scorers.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation for local LTR models.",
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


def build_scorer(args: argparse.Namespace, device: torch.device):
    dtype = _DTYPE_MAP[args.dtype]

    if args.scorer == "ltr":
        if not args.model_dir:
            raise ValueError("--model-dir is required with --scorer ltr")
        return load_ltr_inference_model(
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

    raise ValueError(f"Unsupported scorer: {args.scorer}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation with a selectable scorer backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_gptscore_args(parser)
    add_metricx_args(parser)

    # G-Eval parser can extend this if needed.
    try:
        from clumsification_code.evals.geval.cli import add_geval_args

        add_geval_args(parser)
    except Exception:
        pass

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_scorer(args, device)

    results = run_standard_benchmark_suite(
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    model_dir = (
        args.model_dir
        or getattr(args, "hf_model_name_or_path", "")
        or getattr(args, "metricx_model_name_or_path", "")
    )

    if args.scorer == 'ltr':
        #Parse information from the DS name
        def info_ds_parser(name:str):
            num_layers = 5
            pert_type = "clumsy"

            #Getting model name
            model_name=name[:name.find('_')]
            name=name[name.find('_')+1:]
            #Parsing the language info
            lan=name[:name.find('_')]
            name=name[name.find('_')+1:]
            #Parsing num_layers and pert_type
            training_ds_name=name
            if name[name.rfind('_')+1].isnumeric():
                pert_type = name[:name.rfind('_')]
                num_layers = name[name.rfind('_')+1:]
            else:
                pert_type = name
            return model_name, lan, pert_type, num_layers, training_ds_name
        
        model_name, language, pert_type, num_layers, training_ds_name = info_ds_parser(args.model_name)

        metadata = EvalMetadata(
            model_name=model_name,
            model_dir=model_dir,
            scorer=args.scorer,
            training_dataset=language+"/"+training_ds_name,
            perturbation_type=pert_type,
            num_layers=num_layers,
            context_length=args.context_length,
        )

    else:
        metadata = EvalMetadata(
            model_name=args.model_name,
            model_dir=model_dir,
            scorer=args.scorer,
            training_dataset="none",
            perturbation_type="none",
            num_layers=0,
            context_length=args.context_length,
        )

    write_results_jsonl(metadata=metadata, results=results)


if __name__ == "__main__":
    main()