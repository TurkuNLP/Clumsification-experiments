#!/usr/bin/env python3
# This script has been co-created, refactored, and cleaned using GPT 5.6.

import argparse
import csv
import json
from itertools import product
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MODEL_NAME = "intfloat/multilingual-e5-large"
MAX_SEQ_LEN = 512
DEFAULT_OBJECTIVE_KEYS = {
    "pairwise": "hpo_dev_pairwise_accuracy",
    "regression": "hpo_dev_spearman",
    "binary": "hpo_dev_binary_accuracy",
}


PAIRWISE_LOSSES = {
    "logistic",
    "pairwise_logistic",
    "hinge",
    "margin",
    "weighted_logistic",
    "logistic_weighted",
    "weighted-logistic",
}
REGRESSION_LOSSES = {"huber", "smooth_l1", "smoothl1", "mse", "mae", "l1"}
HUBER_LOSSES = {"huber", "smooth_l1", "smoothl1"}
DEFAULT_TRIAL_FILES = {
    "pairwise": "configs/hpo/fe_external_dev_initial.json",
    "regression": "configs/hpo/fe_regression_pilot.json",
    "binary": "configs/hpo/fe_binary_pilot.json",
}



def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pick_objective(
    metrics: Optional[Dict[str, Any]],
    objective_key: Optional[str],
) -> Optional[float]:
    """
    Select objective value from hpo_dev_metrics.json.

    Current trainer.evaluate(..., metric_key_prefix="hpo_dev") returns keys like:
      - hpo_dev_pairwise_accuracy
      - hpo_dev_correct_points
      - hpo_dev_strict_correct_pairs
      - hpo_dev_score_tie_rate
      - hpo_dev_total_pairs

    We maximize hpo_dev_pairwise_accuracy by default.
    """
    if not metrics:
        return None

    keys_to_try: List[str] = []
    if objective_key:
        keys_to_try.append(objective_key)

    keys_to_try.extend(
        [
            DEFAULT_OBJECTIVE_KEYS.get("pairwise", "hpo_dev_pairwise_accuracy"),
            "hpo_dev_binary_accuracy",
            "hpo_dev_accuracy",
            "hpo_dev_pairwise_accuracy",
            "hpo_dev_spearman",
            "hpo_dev_acc",
            "hpo_dev_mean_pairwise_accuracy",
        ]
    )

    for key in keys_to_try:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)

    # Conservative fallback: first finite scalar hpo_dev_* metric.
    # This is intentionally last because e.g. total_pairs is scalar but not an
    # optimization target.
    for key, value in metrics.items():
        if (
            key.startswith("hpo_dev_")
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            return float(value)

    return None


def normalize_trial(
    trial: Dict[str, Any], training_method: str = "pairwise"
) -> Dict[str, Any]:
    """
    Make old HPO configs compatible with current training arg choices.

    Old configs may contain:
      - loss="margin_ranking"       -> current CLI does not accept this
      - loss_normalization="batch"  -> current CLI accepts only pairs/items

    This function maps those to current equivalents.
    """
    t = dict(trial)

    loss = str(t.get("loss", "binary" if training_method == "binary" else "logistic"))
    if loss == "margin_ranking":
        loss = "hinge"
    t["loss"] = loss

    valid_losses = (
        REGRESSION_LOSSES if training_method == "regression"
        else {"binary"} if training_method == "binary"
        else PAIRWISE_LOSSES
    )
    if t["loss"] not in valid_losses:
        raise ValueError(
            f"Invalid loss in trial {t.get('trial_id')}: {t['loss']!r}. "
            f"Valid losses for {training_method}: {sorted(valid_losses)}"
        )

    if training_method == "regression" and t["loss"] in HUBER_LOSSES:
        t.setdefault("huber_delta", 1.0)
        delta = float(t["huber_delta"])
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(f"Trial {t.get('trial_id')}: huber_delta must be finite and positive")

    return t


def require_trial_keys(trial: Dict[str, Any], training_method: str = "pairwise") -> None:
    required = {
        "trial_id",
        "trial_name",
        "loss",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    }
    if training_method == "pairwise":
        required.update({"epsilon", "scale"})
    missing = sorted(required - set(trial))
    if missing:
        raise ValueError(
            f"Trial {trial.get('trial_id', '<unknown>')} is missing keys: {missing}"
        )


def build_dataset_args(args: argparse.Namespace) -> List[str]:
    if args.formatted_dataset_path:
        return ["--formatted-dataset-path", args.formatted_dataset_path]
    if args.formatted_dataset_name:
        return ["--formatted-dataset-name", args.formatted_dataset_name]
    raise ValueError(
        "You must provide either --formatted_dataset_name or "
        "--formatted_dataset_path. The current training script expects a "
        "preformatted HF DatasetDict."
    )


def build_trial_command(
    *,
    args: argparse.Namespace,
    trial: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    nproc = len([x for x in args.cuda_visible_devices.split(",") if x.strip()])
    max_seq_len = int(trial.get("max_seq_len", args.max_seq_len))

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        args.train_script,

        # Current fe.args parser uses positional model_name and max_seq_len.
        args.model_name,
        str(max_seq_len),

        *build_dataset_args(args),

        "--training-method",
        args.training_method,

        "--output-dir",
        str(output_dir),

        "--seed",
        str(args.seed),

        "--loss",
        str(trial["loss"]),

        "--learning_rate",
        str(trial["learning_rate"]),

        "--warmup_ratio",
        str(trial["warmup_ratio"]),

        "--weight_decay",
        str(trial["weight_decay"]),

        "--num_train_epochs",
        str(trial["num_train_epochs"]),

        "--per_device_train_batch_size",
        str(trial["per_device_train_batch_size"]),

        "--gradient_accumulation_steps",
        str(trial["gradient_accumulation_steps"]),

        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),

        "--logging_steps",
        str(args.logging_steps),

        "--save_strategy",
        args.save_strategy,

        # HPO does one explicit post-training dev eval via --hpo_mode.
        "--eval_strategy",
        args.eval_strategy,

        "--save_total_limit",
        str(args.save_total_limit),

        "--dataloader_num_workers",
        str(args.dataloader_num_workers),

        # Do not touch held-out test during HPO.
        "--skip_final_test_eval",

        "--attn_implementation",
        args.attn_implementation,

        "--pooling",
        args.pooling,
    ]

    if not args.external_dev_hpo:
        cmd.extend(["--hpo_mode", "--hpo_metric_prefix", "hpo_dev"])

    if args.training_method == "regression":
        if not getattr(args, "score_name", None):
            raise ValueError("--score_name is required for regression HPO")
        cmd.extend(["--score-name", args.score_name])
        if trial["loss"] in HUBER_LOSSES:
            cmd.extend(["--huber_delta", str(trial.get("huber_delta", 1.0))])
    elif args.training_method == "pairwise":
        cmd.extend(["--epsilon", str(trial["epsilon"]), "--scale", str(trial["scale"])])

    cmd.extend(["--parallelism", args.parallelism])
    if args.parallelism == "fsdp":
        cmd.extend(
            [
                "--fsdp-sharding-strategy",
                args.fsdp_sharding_strategy,
                "--fsdp-layer-cls",
                args.fsdp_layer_cls,
            ]
        )

    if args.extra_args:
        cmd.extend(args.extra_args)

    # Append the budget last so extra training arguments cannot override it.
    cmd.extend(["--train-sample-budget", str(args.train_sample_budget),
                "--save_strategy", "no", "--eval_strategy", "no"])
    return cmd


def expand_trial_grid(
    payload: Dict[str, Any], training_method: str = "pairwise"
) -> List[Dict[str, Any]]:
    """Cross LR and batch size with only the parameter used by each loss."""
    loss_defaults = {"epsilon": 0.2, "scale": 5.0} if training_method == "pairwise" else {}
    defaults = {**loss_defaults, "warmup_ratio": 0.03,
                "weight_decay": 0.01, "num_train_epochs": 1,
                "gradient_accumulation_steps": 1, **payload.get("defaults", {})}
    trials = []
    for loss, loss_grid in payload["losses"].items():
        grid = {**payload["grid"], **loss_grid}
        keys = list(grid)
        for values in product(*(grid[key] for key in keys)):
            trial = {**defaults, **dict(zip(keys, values)), "loss": loss}
            trial["trial_id"] = len(trials) + 1
            if loss in HUBER_LOSSES:
                shape = f"delta{trial.get('huber_delta', 1.0):g}_"
            elif loss in REGRESSION_LOSSES or loss == "binary":
                shape = ""
            elif loss in {"hinge", "margin", "margin_ranking"}:
                shape = f"margin{trial['epsilon']:g}_"
            else:
                shape = f"scale{trial['scale']:g}_"
            trial["trial_name"] = (
                f"{trial['trial_id']:03d}_{loss}_lr{trial['learning_rate']:g}_"
                f"{shape}bs{trial['per_device_train_batch_size']}"
            )
            trials.append(trial)
    return trials


def trial_global_batch(args: argparse.Namespace, trial: Dict[str, Any]) -> int:
    return (len([gpu for gpu in args.cuda_visible_devices.split(",") if gpu.strip()])
            * int(trial["per_device_train_batch_size"])
            * int(trial["gradient_accumulation_steps"]))


def selected_trials_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    if args.trials_file is None:
        raise ValueError("--trials_file is required; the old trial table is archived")
    trial_payload = load_json(Path(args.trials_file))
    if isinstance(trial_payload, dict):
        trial_payload = expand_trial_grid(trial_payload, args.training_method)
    if not isinstance(trial_payload, list):
        raise ValueError("--trials_file must contain a JSON trial list or grid object")

    for raw_trial in trial_payload:
        require_trial_keys(raw_trial, args.training_method)
        trial = normalize_trial(raw_trial, args.training_method)

        trial_id = int(trial["trial_id"])

        if args.start_trial_id is not None and trial_id < args.start_trial_id:
            continue

        if args.end_trial_id is not None and trial_id > args.end_trial_id:
            continue

        selected.append(trial)

    return selected


def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
    if path.name == "final":
        return (1, 0)
    return (0, int(path.name.removeprefix("checkpoint-")))


def evaluate_external_dev(
    *,
    args: argparse.Namespace,
    output_root: Path,
    trial_name: str,
    trial_dir: Path,
    max_seq_len: int,
) -> Optional[Path]:
    """Evaluate one trial's saved checkpoints across the allocated GPUs."""
    # The trainer saves final/ after exactly one budgeted pass; no early selection.
    checkpoints = [trial_dir / "final"] if (trial_dir / "final").is_dir() else []
    if not checkpoints:
        print(f"[HPO] No checkpoints found for {trial_name}.", file=sys.stderr)
        return None

    gpu_ids = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
    run_name = f"{output_root.name}__{trial_name}"
    result_dir = Path("data/evals/external_dev")
    result_dir.mkdir(parents=True, exist_ok=True)

    def run_on_gpu(gpu_id: str, assigned: List[Path]) -> bool:
        failed = False
        for checkpoint in assigned:
            model_name = f"{run_name}__{checkpoint.name}"
            result_path = result_dir / f"{model_name}.jsonl"
            log_path = trial_dir / f"eval_dev_{checkpoint.name}.log"
            if args.resume and result_path.exists() and result_path.stat().st_size > 0:
                print(f"[HPO] Skipping existing dev result: {model_name}")
                if not log_path.exists():
                    log_path.write_text(
                        f"Skipped because this result already exists: {result_path}\n",
                        encoding="utf-8",
                    )
                continue

            cmd = [
                sys.executable,
                "-m",
                "clumsification_code.evals.run_benchmark",
                "--evaluation-role",
                "external-dev",
                "--scorer",
                "fe",
                "--model-name",
                model_name,
                "--model-dir",
                str(checkpoint),
                "--batch-size",
                str(args.external_dev_batch_size),
                "--max-length",
                str(max_seq_len),
            ]
            child_env = os.environ.copy()
            child_env.pop("CUDA_VISIBLE_DEVICES", None)
            child_env.pop("HIP_VISIBLE_DEVICES", None)
            child_env["ROCR_VISIBLE_DEVICES"] = gpu_id
            child_env["WANDB_MODE"] = "disabled"
            print(f"[HPO] Evaluating {model_name} on GPU {gpu_id}")
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    cmd,
                    env=child_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if process.returncode != 0:
                failed = True
                print(f"[HPO] Dev evaluation failed: {model_name}. See {log_path}", file=sys.stderr)
        return failed

    worker_count = min(len(gpu_ids), len(checkpoints))
    assignments = [checkpoints[index::worker_count] for index in range(worker_count)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        failures = list(
            executor.map(run_on_gpu, gpu_ids[:worker_count], assignments)
        )

    if any(failures):
        return None

    result_paths = [
        result_dir / f"{run_name}__{checkpoint.name}.jsonl"
        for checkpoint in checkpoints
    ]
    result_paths = [path for path in result_paths if path.exists() and path.stat().st_size > 0]
    if not result_paths:
        return None

    ranking_path = trial_dir / "dev_checkpoint_ranking.csv"
    rank_cmd = [
        sys.executable,
        str(Path(__file__).with_name("rank_external_dev_checkpoints.py")),
        *(str(path) for path in result_paths),
        "--output",
        str(ranking_path),
    ]
    rank_process = subprocess.run(rank_cmd, text=True)
    if rank_process.returncode != 0:
        print(f"[HPO] Could not rank {trial_name} checkpoints.", file=sys.stderr)
        return None
    if any(failures):
        print(f"[HPO] Some {trial_name} checkpoints failed; rerun with --resume to retry them.", file=sys.stderr)
    return ranking_path


def compare_external_dev_trials(
    *,
    output_root: Path,
    ranking_paths: List[Path],
    args: argparse.Namespace,
    trials: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compare budget endpoints, retaining all three dev dataset scores."""
    trial_by_name = {trial["trial_name"]: trial for trial in trials}
    rows = []
    for path in ranking_paths:
        trial = trial_by_name[path.parent.name]
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        scores = {key: float(row[f"{key}_score"])
                  for key in ("ELLIPSE", "JFLEG", "CoheSentia")}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Non-finite dev scores in {path}")
        rows.append({
            "trial_name": trial["trial_name"], "selected_checkpoint": "final",
            "mean_dev_score": sum(scores.values()) / len(scores),
            **{f"{key}_score": value for key, value in scores.items()},
            "training_samples": args.train_sample_budget,
            "global_batch_size": trial_global_batch(args, trial),
            "optimizer_steps": args.train_sample_budget // trial_global_batch(args, trial),
            "loss": trial["loss"], "learning_rate": trial["learning_rate"],
            "epsilon": trial.get("epsilon"), "scale": trial.get("scale"),
            "huber_delta": trial.get("huber_delta"),
        })
    if not rows:
        return None
    rows.sort(key=lambda row: (-row["mean_dev_score"], row["trial_name"]))
    with (output_root / "external_dev_hpo_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {**rows[0],
            "selected_model_dir": str(output_root / rows[0]["trial_name"] / "final"),
            "hparams": trial_by_name[rows[0]["trial_name"]]}
    dump_json(output_root / "best_external_dev_trial.json", best)
    # Keep a winner for each loss so logistic is assessed independently of hinge.
    for loss in {row["loss"] for row in rows}:
        winner = next(row for row in rows if row["loss"] == loss)
        dump_json(output_root / f"best_external_dev_{loss}.json",
                  {**winner, "hparams": trial_by_name[winner["trial_name"]]})
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential HPO runner for train_fe_model.py"
    )

    parser.add_argument(
        "--train_script",
        type=str,
        default="scripts/train_fe_model.py",
        help="Path to train_fe_model.py.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="hpo_runs_multilingual_e5_large_chain5",
    )

    parser.add_argument("--trials_file", type=str, default=None,
                        help="JSON trial list or loss-specific grid; defaults to the training method's config.")
    parser.add_argument(
        "--external_dev_hpo",
        action="store_true",
        help="Select trials and checkpoints with the external human-dev panel.",
    )
    parser.add_argument("--train_sample_budget", type=int, default=4992,
                        help="Training examples per trial, independent of batch size (default: 4992).")
    parser.add_argument("--external_dev_batch_size", type=int, default=32)
    parser.add_argument("--training_method", choices=["pairwise", "regression", "binary"], default="pairwise")
    parser.add_argument("--score_name", type=str, default=None,
                        help="Score field required for regression HPO.")

    parser.add_argument(
        "--formatted_dataset_name",
        type=str,
        default=None,
        help=(
            "Name of a previously-created formatted dataset under "
            "data/hf_datasets/<name>."
        ),
    )

    parser.add_argument(
        "--formatted_dataset_path",
        type=str,
        default=None,
        help="Explicit path to a saved Hugging Face DatasetDict.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Model passed as positional model_name to the training script.",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Max sequence length passed as positional max_seq_len.",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1,2,3",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Use the same seed across trials so train/dev/test splits and "
            "dropout initialization are comparable."
        ),
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--parallelism",
        choices=["ddp", "fsdp"],
        default="ddp",
    )

    parser.add_argument(
        "--fsdp-sharding-strategy",
        choices=["shard_grad_op", "full_shard"],
        default="shard_grad_op",
    )

    parser.add_argument(
        "--fsdp_layer_cls",
        "--fsdp-layer-cls",
        dest="fsdp_layer_cls",
        type=str,
        default=None,
        help="Required only when --parallelism fsdp.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument(
        "--pooling",
        choices=["auto", "mean", "last_token"],
        default="auto",
        help="Pooling strategy; auto selects from the backbone name.",
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        default="no",
        help="For HPO, usually 'no'.",
    )

    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="no",
        help="For HPO, usually 'no' because --hpo_mode evaluates dev once.",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--objective_key",
        type=str,
        default=None,
        help="Metric key from hpo_dev_metrics.json to maximize.",
    )

    parser.add_argument(
        "--start_trial_id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--end_trial_id",
        type=int,
        default=None,
        help="Inclusive.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed training and retry only missing evaluations.",
    )

    parser.add_argument(
        "--overwrite_summary",
        action="store_true",
        help="Delete existing hpo_summary.jsonl before running.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands but do not execute them.",
    )

    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra args passed to the training script. Put this last, e.g. "
            "--extra_args --some_arg value"
        ),
    )

    args = parser.parse_args()

    if args.trials_file is None:
        args.trials_file = DEFAULT_TRIAL_FILES[args.training_method]

    if args.objective_key is None:
        args.objective_key = DEFAULT_OBJECTIVE_KEYS[args.training_method]

    if args.formatted_dataset_name and args.formatted_dataset_path:
        raise ValueError(
            "Use only one of --formatted_dataset_name or --formatted_dataset_path."
        )
    if args.parallelism == "fsdp" and not args.fsdp_layer_cls:
        raise ValueError("--fsdp-layer-cls is required when --parallelism fsdp.")
    if args.training_method == "regression" and not args.score_name:
        raise ValueError("--score_name is required when --training_method regression")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "hpo_summary.jsonl"
    if args.overwrite_summary and summary_path.exists():
        summary_path.unlink()

    env = os.environ.copy()
    if args.external_dev_hpo:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["ROCR_VISIBLE_DEVICES"] = args.cuda_visible_devices
    else:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["WANDB_MODE"] = "disabled"
    env.pop("ACCELERATE_USE_FSDP", None)
    env.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    selected_trials = selected_trials_from_args(args)

    for trial in selected_trials:
        batch = trial_global_batch(args, trial)
        if batch <= 0 or args.train_sample_budget <= 0 or args.train_sample_budget % batch:
            raise ValueError(
                f"Sample budget {args.train_sample_budget} must be divisible by "
                f"global batch {batch} for {trial['trial_name']}"
            )

    if not selected_trials:
        print("[HPO] No trials selected.", file=sys.stderr)
        return

    print(f"[HPO] Selected {len(selected_trials)} trial(s).")
    print(f"[HPO] Model: {args.model_name}")
    print(f"[HPO] Max seq len: {args.max_seq_len}")
    print(f"[HPO] GPUs: {args.cuda_visible_devices}")
    print(
        "[HPO] Objective: mean of the three English dev dataset scores"
        if args.external_dev_hpo
        else f"[HPO] Objective: {args.objective_key}"
    )
    print(f"[HPO] Output root: {output_root}")

    if args.external_dev_hpo:
        ranking_paths: List[Path] = []

        for trial in selected_trials:
            trial_name = str(trial["trial_name"])
            trial_dir = output_root / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            log_path = trial_dir / "run.log"
            command_path = trial_dir / "command.json"
            final_state = trial_dir / "final" / "fe_model_state.pt"
            cmd = build_trial_command(args=args, trial=trial, output_dir=trial_dir)

            dump_json(
                command_path,
                {
                    "cmd": cmd,
                    "cmd_shell": " ".join(shlex.quote(item) for item in cmd),
                    "env_overrides": {
                        "ROCR_VISIBLE_DEVICES": args.cuda_visible_devices,
                        "WANDB_MODE": env["WANDB_MODE"],
                    },
                    "trial": trial,
                },
            )
            print(f"\n[HPO] {trial_name}")
            print(" ".join(shlex.quote(item) for item in cmd))

            if args.dry_run:
                append_jsonl(
                    summary_path,
                    {
                        "trial_id": trial["trial_id"],
                        "trial_name": trial_name,
                        "status": "dry_run",
                        "hparams": trial,
                        "command_path": str(command_path),
                    },
                )
                continue

            training_ok = args.resume and final_state.exists()
            status = "skipped_existing_training" if training_ok else "failed"
            returncode = 0
            if training_ok:
                print(f"[HPO] Training already complete: {trial_name}")
            else:
                with log_path.open("w", encoding="utf-8") as log_file:
                    process = subprocess.run(
                        cmd,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                returncode = process.returncode
                training_ok = returncode == 0
                status = "ok" if training_ok else "failed"

            ranking_path = None
            if training_ok:
                ranking_path = evaluate_external_dev(
                    args=args,
                    output_root=output_root,
                    trial_name=trial_name,
                    trial_dir=trial_dir,
                    max_seq_len=int(trial.get("max_seq_len", args.max_seq_len)),
                )
                if ranking_path is not None:
                    ranking_paths.append(ranking_path)
                else:
                    status = "evaluation_failed"
            else:
                print(f"[HPO] Training failed: {trial_name}. See {log_path}", file=sys.stderr)

            append_jsonl(
                summary_path,
                {
                    "trial_id": trial["trial_id"],
                    "trial_name": trial_name,
                    "status": status,
                    "returncode": returncode,
                    "external_dev_ranking": (
                        str(ranking_path) if ranking_path is not None else None
                    ),
                    "hparams": trial,
                    "output_dir": str(trial_dir),
                    "log_path": str(log_path),
                },
            )

        if args.dry_run:
            print("[HPO] Dry run complete; no training or evaluation was run.")
            return

        best_external = compare_external_dev_trials(
            output_root=output_root,
            ranking_paths=ranking_paths,
            args=args,
            trials=selected_trials,
        )
        if best_external is None:
            print("[HPO] No complete external-dev trial comparison was produced.", file=sys.stderr)
            return
        print(
            f"[HPO] Best external-dev trial: {best_external['trial_name']} "
            f"at {best_external['selected_checkpoint']}"
        )
        return

    best: Optional[Dict[str, Any]] = None

    for trial in selected_trials:
        trial_name = str(trial["trial_name"])
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = trial_dir / "hpo_dev_metrics.json"
        log_path = trial_dir / "run.log"
        command_path = trial_dir / "command.json"

        if args.resume and metrics_path.exists():
            print(f"[HPO] Skipping existing trial: {trial_name}")
            metrics = load_json(metrics_path)
            objective = pick_objective(metrics, args.objective_key)

            record = {
                "trial_id": trial["trial_id"],
                "trial_name": trial_name,
                "status": "skipped_existing",
                "objective": objective,
                "metrics": metrics,
                "hparams": trial,
                "output_dir": str(trial_dir),
                "log_path": str(log_path),
            }

            append_jsonl(summary_path, record)

            if objective is not None and (
                best is None or objective > float(best["objective"])
            ):
                best = record
                dump_json(output_root / "best_trial.json", best)

            continue

        cmd = build_trial_command(
            args=args,
            trial=trial,
            output_dir=trial_dir,
        )

        dump_json(
            command_path,
            {
                "cmd": cmd,
                "cmd_shell": " ".join(shlex.quote(x) for x in cmd),
                "env_overrides": {
                    "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                    "WANDB_MODE": env["WANDB_MODE"],
                    "PARALLELISM": args.parallelism,
                    "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
                },
                "trial": trial,
            },
        )

        print("\n" + "=" * 100)
        print(f"[HPO] Starting {trial_name}")
        print("[HPO] Command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        print("=" * 100 + "\n")

        if args.dry_run:
            record = {
                "trial_id": trial["trial_id"],
                "trial_name": trial_name,
                "status": "dry_run",
                "objective": None,
                "metrics": None,
                "hparams": trial,
                "output_dir": str(trial_dir),
                "log_path": str(log_path),
                "command_path": str(command_path),
            }
            append_jsonl(summary_path, record)
            continue

        with log_path.open("w", encoding="utf-8") as log_f:
            process = subprocess.run(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        metrics = load_json(metrics_path)
        objective = pick_objective(metrics, args.objective_key)

        record = {
            "trial_id": trial["trial_id"],
            "trial_name": trial_name,
            "status": "ok" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "objective": objective,
            "metrics": metrics,
            "hparams": trial,
            "output_dir": str(trial_dir),
            "log_path": str(log_path),
            "command_path": str(command_path),
        }

        append_jsonl(summary_path, record)

        if process.returncode != 0:
            print(
                f"[HPO] Trial failed: {trial_name}. See {log_path}",
                file=sys.stderr,
            )
            continue

        if objective is None:
            print(
                f"[HPO] Trial finished but objective could not be read: "
                f"{trial_name}. Check {metrics_path}.",
                file=sys.stderr,
            )
            if metrics is not None:
                print(f"[HPO] Available metric keys: {sorted(metrics.keys())}")
            continue

        print(f"[HPO] Finished {trial_name}. objective={objective}")

        if best is None or objective > float(best["objective"]):
            best = record
            best_path = output_root / "best_trial.json"
            dump_json(best_path, best)

            print(f"[HPO] New best trial: {trial_name}, objective={objective}")
            print(f"[HPO] Saved best trial to {best_path}")

    print("\n" + "=" * 100)
    print("[HPO] Done.")

    if best is not None:
        print(f"[HPO] Best trial: {best['trial_name']}")
        print(f"[HPO] Best objective: {best['objective']}")
        print(f"[HPO] Best hparams: {json.dumps(best['hparams'], indent=2)}")
    else:
        print("[HPO] No successful trial with a readable objective.")

    print("=" * 100)


if __name__ == "__main__":
    main()
