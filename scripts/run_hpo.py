#!/usr/bin/env python3
# This script has been co-created, refactored, and cleaned using GPT 5.6.

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MODEL_NAME = "intfloat/multilingual-e5-large"
MAX_SEQ_LEN = 512
DEFAULT_OBJECTIVE_KEYS = {
    "pairwise": "hpo_dev_pairwise_accuracy",
    "regression": "hpo_dev_spearman",
}


VALID_LOSSES = {
    "logistic",
    "pairwise_logistic",
    "hinge",
    "margin",
    "weighted_logistic",
    "logistic_weighted",
    "weighted-logistic",
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


def normalize_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make old HPO configs compatible with current training arg choices.

    Old configs may contain:
      - loss="margin_ranking"       -> current CLI does not accept this
      - loss_normalization="batch"  -> current CLI accepts only pairs/items

    This function maps those to current equivalents.
    """
    t = dict(trial)

    loss = str(t.get("loss", "logistic"))
    if loss == "margin_ranking":
        loss = "hinge"
    t["loss"] = loss

    if t["loss"] not in VALID_LOSSES:
        raise ValueError(
            f"Invalid loss in trial {t.get('trial_id')}: {t['loss']!r}. "
            f"Valid losses: {sorted(VALID_LOSSES)}"
        )

    return t


def require_trial_keys(trial: Dict[str, Any]) -> None:
    required = {
        "trial_id",
        "trial_name",
        "loss",
        "epsilon",
        "scale",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    }
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

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        args.train_script,

        # Current fe.args parser uses positional model_name and max_seq_len.
        args.model_name,
        str(args.max_seq_len),

        *build_dataset_args(args),

        "--training-method",
        args.training_method,

        "--output-dir",
        str(output_dir),

        "--seed",
        str(args.seed),

        "--loss",
        str(trial["loss"]),

        "--epsilon",
        str(trial["epsilon"]),

        "--scale",
        str(trial["scale"]),

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

        "--hpo_mode",

        # Do not touch held-out test during HPO.
        "--skip_final_test_eval",

        "--hpo_metric_prefix",
        "hpo_dev",

        "--attn_implementation",
        args.attn_implementation,
    ]

    if args.fsdp_layer_cls:
        cmd.extend(["--fsdp_layer_cls", args.fsdp_layer_cls])

    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def selected_trials_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    if args.trials_file is None:
        raise ValueError("--trials_file is required; the old trial table is archived")
    trial_payload = load_json(Path(args.trials_file))
    if not isinstance(trial_payload, list):
        raise ValueError("--trials_file must contain a JSON list of trial objects")

    for raw_trial in trial_payload:
        require_trial_keys(raw_trial)
        trial = normalize_trial(raw_trial)

        trial_id = int(trial["trial_id"])

        if args.start_trial_id is not None and trial_id < args.start_trial_id:
            continue

        if args.end_trial_id is not None and trial_id > args.end_trial_id:
            continue

        selected.append(trial)

    return selected


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

    parser.add_argument("--trials_file", type=str, required=True,
                        help="JSON list of HPO trial objects.")
    parser.add_argument("--training_method", choices=["pairwise", "regression"], default="pairwise")

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
        "--fsdp_layer_cls",
        type=str,
        default="XLMRobertaLayer",
        help=(
            "For intfloat/multilingual-e5-large this is usually "
            "XLMRobertaLayer."
        ),
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
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
        default=DEFAULT_OBJECTIVE_KEYS["pairwise"],
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
        help="Skip trials that already have hpo_dev_metrics.json.",
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

    if args.formatted_dataset_name and args.formatted_dataset_path:
        raise ValueError(
            "Use only one of --formatted_dataset_name or --formatted_dataset_path."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "hpo_summary.jsonl"
    if args.overwrite_summary and summary_path.exists():
        summary_path.unlink()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["WANDB_MODE"] = "disabled"
    env["ACCELERATE_USE_FSDP"] = "true"
    env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    selected_trials = selected_trials_from_args(args)

    if not selected_trials:
        print("[HPO] No trials selected.", file=sys.stderr)
        return

    print(f"[HPO] Selected {len(selected_trials)} trial(s).")
    print(f"[HPO] Model: {args.model_name}")
    print(f"[HPO] Max seq len: {args.max_seq_len}")
    print(f"[HPO] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    print(f"[HPO] Objective: {args.objective_key}")
    print(f"[HPO] Output root: {output_root}")

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
                    "ACCELERATE_USE_FSDP": env["ACCELERATE_USE_FSDP"],
                    "FSDP_CPU_RAM_EFFICIENT_LOADING": env[
                        "FSDP_CPU_RAM_EFFICIENT_LOADING"
                    ],
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
