"""Run a real GPU check for capped thinking followed by a complete JSON answer."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from clumsification_code.perturbations.vllm_runner import VLLMRunner
from clumsification_code.perturbations.parallel_runner import ParallelLLMRunner
from clumsification_code.perturbations.schemas import ChatCompletion
from clumsification_code.data.io import write_json_atomic


def check_result(result):
    metadata = result.metadata if isinstance(result, ChatCompletion) else {}
    boundary = metadata.get("reasoning_tokens_before_end")
    counts = metadata.get("reported_applied_edits", {})
    # A tiny budget plus a long-thinking request should reach the forced boundary.
    # Allow for generation-prefix and forced-token accounting differences.
    passed = (
        isinstance(result, ChatCompletion)
        and not result.failure_reason
        and bool(result.text)
        and boundary is not None
        and 7 <= boundary <= 10
        and type(counts.get("determiner_error")) is int
    )
    return {
        "passed": bool(passed),
        "metadata": metadata,
        "text": getattr(result, "text", None),
        "failure_reason": getattr(result, "failure_reason", None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--device-groups")
    parser.add_argument("--accelerator", choices=["cuda", "rocm"], default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=Path("analysis_outputs/perturbation_pilot/hpc_check.json"),
    )
    args = parser.parse_args()
    config = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": 4096,
        "enable_thinking": True,
        "thinking_token_cap": 8,
        "output_metadata_tokens": 256,
        "answer_reserve_tokens": 512,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "structured_output": True,
    }
    if args.tokenizer:
        config["tokenizer"] = args.tokenizer
    if args.revision:
        config["revision"] = args.revision
    source = "The committee reviewed the proposal carefully before reaching a decision about the new project."
    prompt = [{"role": "user", "content": (
        "Think through at least 100 editing alternatives before answering. "
        "Rewrite this source with one determiner error. Return JSON with text "
        "and applied_edits (determiner_error count). Source: " + source
    )}]
    report = {
        "passed": False, "checks": [], "python": platform.python_version(),
        "configuration": config, "device_groups": args.device_groups, "versions": {},
    }
    for name in ("torch", "vllm", "transformers"):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = None
    runner = None
    try:
        import torch
        report.update(
            hip=torch.version.hip, cuda=torch.version.cuda,
            visible_devices=torch.cuda.device_count(),
            devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
        runner = (
            ParallelLLMRunner(args.device_groups, accelerator=args.accelerator)
            if args.device_groups else VLLMRunner()
        )
        # Exercise every replica, not only the first GPU group.
        count = len(runner.groups) if args.device_groups else 1
        runner.prepare([prompt] * count, [source] * count, config)
        results = runner(
            args.model_path, [prompt] * count, .7, 1024, config=config,
            source_texts=[source] * count,
            request_ids=[f"smoke-{i}" for i in range(count)], attempts=[0] * count,
            requested_edits=[["determiner_error"]] * count,
        )
        report["checks"] = [check_result(result) for result in results]
        report["passed"] = len(results) == count and all(c["passed"] for c in report["checks"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Preserve diagnostics even when imports, model loading, or inference fail.
        write_json_atomic(args.output, report, overwrite=True)
        print(json.dumps(report, indent=2))
        if runner is not None:
            runner.close()
    if not report["passed"]:
        raise SystemExit("HPC check failed: inspect the budget/JSON results before the pilot")


if __name__ == "__main__":
    main()
