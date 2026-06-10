import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from hps_to_test import HPS_TO_TEST


def str_bool_flag(name: str, value: bool):
    return [name] if value else []


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_objective(metrics: Dict[str, Any], objective_key: Optional[str]) -> Optional[float]:
    """
    Select objective value from hpo_dev_metrics.json.

    If objective_key is provided, use that exact key.
    Otherwise, try common names. You should ideally pass --objective_key explicitly
    once you know what evaluate_win_rate_distributed returns.
    """
    if metrics is None:
        return None

    if objective_key:
        value = metrics.get(objective_key)
        return float(value) if isinstance(value, (int, float)) else None

    preferred_keys = [
        "hpo_dev_win_rate",
        "hpo_dev_accuracy",
        "hpo_dev_pair_accuracy",
        "hpo_dev_acc",
        "hpo_dev_mean_win_rate",
    ]

    for key in preferred_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    # Fallback: use first scalar hpo_dev_* metric.
    for key, value in metrics.items():
        if key.startswith("hpo_dev_") and isinstance(value, (int, float)):
            return float(value)

    return None


def build_trial_command(
    *,
    train_script: str,
    trial: Dict[str, Any],
    output_dir: Path,
    custom_datasets,
    seed: int,
    per_device_eval_batch_size: int,
    logging_steps: int,
    dataloader_num_workers: int,
    fsdp_layer_cls: str,
    extra_args,
):
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "4",
        train_script,

        "--model_name",
        "intfloat/multilingual-e5-large",

        "--output_dir",
        str(output_dir),

        "--max_seq_len",
        "512",

        "--seed",
        str(seed),

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

        "--loss_normalization",
        str(trial["loss_normalization"]),

        "--num_train_epochs",
        str(trial["num_train_epochs"]),

        "--per_device_train_batch_size",
        str(trial["per_device_train_batch_size"]),

        "--gradient_accumulation_steps",
        str(trial["gradient_accumulation_steps"]),

        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),

        "--logging_steps",
        str(logging_steps),

        "--save_strategy",
        "no",

        # We explicitly do a post-training dev eval via --hpo_mode.
        # Avoid repeated eval during training unless you want learning curves.
        "--eval_strategy",
        "no",

        "--save_total_limit",
        "1",

        "--dataloader_num_workers",
        str(dataloader_num_workers),

        "--hpo_mode",
        "--skip_final_test_eval",
        "--hpo_metric_prefix",
        "hpo_dev",
    ]

    if custom_datasets:
        cmd.append("--custom_datasets")
        cmd.extend(custom_datasets)

    if fsdp_layer_cls:
        cmd.extend(["--fsdp_layer_cls", fsdp_layer_cls])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_script",
        type=str,
        required=True,
        help="Path to your main training script, e.g. train_ltr.py",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="hpo_runs_multilingual_e5_large",
    )

    parser.add_argument(
        "--custom_datasets",
        nargs="+",
        required=True,
        help="Datasets passed to --custom_datasets in your training script.",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1,2,3",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Use the same seed across trials so train/dev/test splits are identical.",
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
            "For intfloat/multilingual-e5-large this is usually XLMRobertaLayer. "
            "If your auto-wrap code expects a fully qualified name, change this."
        ),
    )

    parser.add_argument(
        "--objective_key",
        type=str,
        default=None,
        help=(
            "Exact metric key from hpo_dev_metrics.json to maximize, e.g. "
            "hpo_dev_win_rate. If omitted, the runner guesses."
        ),
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
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed through to the training script. Put after --extra_args.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "hpo_summary.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["WANDB_MODE"] = "disabled"
    env["ACCELERATE_USE_FSDP"] = "true"
    env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    selected_trials = []
    for trial in HPS_TO_TEST:
        trial_id = int(trial["trial_id"])

        if args.start_trial_id is not None and trial_id < args.start_trial_id:
            continue
        if args.end_trial_id is not None and trial_id > args.end_trial_id:
            continue

        selected_trials.append(trial)

    best = None

    for trial in selected_trials:
        trial_name = trial["trial_name"]
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = trial_dir / "hpo_dev_metrics.json"

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
            }

            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if objective is not None and (best is None or objective > best["objective"]):
                best = record

            continue

        cmd = build_trial_command(
            train_script=args.train_script,
            trial=trial,
            output_dir=trial_dir,
            custom_datasets=args.custom_datasets,
            seed=args.seed,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=args.logging_steps,
            dataloader_num_workers=args.dataloader_num_workers,
            fsdp_layer_cls=args.fsdp_layer_cls,
            extra_args=args.extra_args,
        )

        print("\n" + "=" * 100)
        print(f"[HPO] Starting {trial_name}")
        print("[HPO] Command:")
        print(" ".join(cmd))
        print("=" * 100 + "\n")

        log_path = trial_dir / "run.log"

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
        }

        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if process.returncode != 0:
            print(f"[HPO] Trial failed: {trial_name}. See {log_path}", file=sys.stderr)
            continue

        if objective is None:
            print(
                f"[HPO] Trial finished but objective could not be read: {trial_name}. "
                f"Check {metrics_path}.",
                file=sys.stderr,
            )
            if metrics is not None:
                print(f"[HPO] Available metric keys: {sorted(metrics.keys())}")
            continue

        print(f"[HPO] Finished {trial_name}. objective={objective}")

        if best is None or objective > best["objective"]:
            best = record
            best_path = output_root / "best_trial.json"
            with best_path.open("w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)

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