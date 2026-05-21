import os
import json
import math
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import gc
import csv
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Custom dataset module
import dataset_functions as d_f

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fsdp_ltr")

os.environ["WANDB_MODE"] = "disabled"
os.environ["ACCELERATE_USE_FSDP"] = "true"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"


# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str)
    parser.add_argument("max_seq_len", type=int)
    parser.add_argument("custom_dataset", type=str)
    parser.add_argument("output_dir", type=str)

    parser.add_argument("--downsample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="items",
        choices=["pairs", "items"],
    )

    parser.add_argument("--length_diagnostics", action="store_true")
    parser.add_argument("--length_plot_num_bins", type=int, default=10)
    parser.add_argument("--length_plot_max_pairs", type=int, default=200000)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--fsdp_layer_cls", type=str, default=None)

    return parser.parse_args()


# Plotting functions

def _diagnostic_dir(output_dir: Optional[str]) -> str:
    root = output_dir or "."
    diag_dir = os.path.join(root, "length_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    return diag_dir


def _safe_float_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
    
def _safe_int_metric(metrics: Dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _write_winrate_history_plot(
    output_dir: Optional[str],
    step: Optional[int],
    epoch: Optional[float],
    metrics: Dict[str, Any],
) -> None:
    """
    Appends one evaluation record and updates a persistent training-history plot.

    Produces:
        length_diagnostics/winrate_history.jsonl
        length_diagnostics/winrate_history.png
    """
    diag_dir = _diagnostic_dir(output_dir)

    history_path = os.path.join(diag_dir, "winrate_history.jsonl")
    plot_path = os.path.join(diag_dir, "winrate_history.png")

    record = {
        "step": int(step) if step is not None else None,
        "epoch": float(epoch) if epoch is not None else None,

        "win_rate": _safe_float_metric(metrics, "win_rate"),
        "win_rate_when_shorter_is_better": _safe_float_metric(
            metrics, "win_rate_when_shorter_is_better"
        ),
        "win_rate_when_longer_is_better": _safe_float_metric(
            metrics, "win_rate_when_longer_is_better"
        ),

        # With tie-as-half-credit, this can be fractional.
        "correct_points": _safe_float_metric(metrics, "correct_points"),

        # Kept for backward compatibility with older logs.
        "correct_pairs": _safe_float_metric(metrics, "correct_pairs"),

        "strict_correct_pairs": _safe_int_metric(metrics, "strict_correct_pairs"),
        "score_tie_pairs": _safe_int_metric(metrics, "score_tie_pairs"),
        "total_pairs": _safe_int_metric(metrics, "total_pairs"),
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    # Use global step on the x-axis when available.
    if all(r.get("step") is not None for r in records):
        x = [r["step"] for r in records]
        xlabel = "Global step"
    else:
        x = list(range(1, len(records) + 1))
        xlabel = "Evaluation number"

    series = [
        ("win_rate", "Overall win rate"),
        ("win_rate_when_shorter_is_better", "Win rate when shorter is better"),
        ("win_rate_when_longer_is_better", "Win rate when longer is better"),
    ]

    plt.figure(figsize=(9, 5))

    for key, label in series:
        y = [r.get(key) for r in records]
        valid = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]

        if not valid:
            continue

        xx, yy = zip(*valid)
        plt.plot(xx, yy, marker="o", linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Win rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def _build_relative_length_quantile_rows(
    diagnostic_pairs: List[tuple],
    num_bins: int,
) -> List[Dict[str, Any]]:
    """
    Builds data-derived quantile buckets.

    diagnostic_pairs contains:
        (relative_length_difference, point_value)

    point_value is:
        1.0 if the model strictly ranked the pair correctly.
        0.5 if the model assigned exactly tied scores.
        0.0 if the model strictly ranked the pair incorrectly.

    Returns rows with:
        bin_id, rel_len_diff_min, rel_len_diff_max, rel_len_diff_mean,
        win_rate, total_pairs, correct_points
    """
    if not diagnostic_pairs:
        return []

    diagnostic_pairs = sorted(diagnostic_pairs, key=lambda x: x[0])

    n = len(diagnostic_pairs)
    k = max(1, min(int(num_bins), n))

    rows = []

    for b in range(k):
        start = b * n // k
        end = (b + 1) * n // k

        chunk = diagnostic_pairs[start:end]

        if not chunk:
            continue

        rels = [float(x[0]) for x in chunk]
        points = [float(x[1]) for x in chunk]

        total = len(chunk)
        correct_points = sum(points)

        rows.append(
            {
                "bin_id": b,
                "rel_len_diff_min": min(rels),
                "rel_len_diff_max": max(rels),
                "rel_len_diff_mean": sum(rels) / total,
                "win_rate": correct_points / total if total > 0 else 0.0,
                "total_pairs": total,

                # New, semantically correct name.
                "correct_points": correct_points,

                # Backward-compatible alias.
                "correct_pairs": correct_points,
            }
        )

    return rows


def _write_relative_length_winrate_plot(
    output_dir: Optional[str],
    step: Optional[int],
    epoch: Optional[float],
    diagnostic_pairs: List[tuple],
    num_bins: int = 10,
) -> None:
    """
    Writes a plot of win rate versus relative length difference.

    Relative length difference:
        abs(len_i - len_j) / max(len_i, len_j)

    Buckets are equal-frequency quantile buckets inferred from the diagnostic data.

    Produces:
        length_diagnostics/relative_length_winrate_step_<STEP>.png
        length_diagnostics/relative_length_winrate_latest.png
        length_diagnostics/relative_length_winrate_step_<STEP>.csv
    """
    if not diagnostic_pairs:
        return

    diag_dir = _diagnostic_dir(output_dir)

    step_str = str(step) if step is not None else "unknown"

    rows = _build_relative_length_quantile_rows(
        diagnostic_pairs=diagnostic_pairs,
        num_bins=num_bins,
    )

    if not rows:
        return

    csv_path = os.path.join(
        diag_dir,
        f"relative_length_winrate_step_{step_str}.csv",
    )

    png_path = os.path.join(
        diag_dir,
        f"relative_length_winrate_step_{step_str}.png",
    )

    latest_png_path = os.path.join(
        diag_dir,
        "relative_length_winrate_latest.png",
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_id",
                "rel_len_diff_min",
                "rel_len_diff_max",
                "rel_len_diff_mean",
                "win_rate",
                "total_pairs",
                "correct_points",
                "correct_pairs",  # backward-compatible alias
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    x = [row["rel_len_diff_mean"] for row in rows]
    y = [row["win_rate"] for row in rows]
    counts = [row["total_pairs"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=2)

    for xx, yy, count in zip(x, y, counts):
        plt.annotate(
            str(count),
            xy=(xx, yy),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.75,
        )

    title = "Win rate by relative length difference"
    if epoch is not None:
        title += f" | epoch={float(epoch):.3f}"
    if step is not None:
        title += f" | step={step}"

    plt.title(title)
    plt.xlabel("Relative length difference: abs(len_i - len_j) / max(len_i, len_j)")
    plt.ylabel("Win rate")
    plt.ylim(0.0, 1.0)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    shutil.copyfile(png_path, latest_png_path)

# ----------------------------
# Model
# ----------------------------
class ScoringHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = 256, dropout: float = 0.1):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _pad_group_scores(flat_scores: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    sizes = group_sizes.detach().cpu().tolist()
    max_k = max(sizes)

    padded = flat_scores.new_zeros((len(sizes), max_k))

    start = 0
    for b, n in enumerate(sizes):
        padded[b, :n] = flat_scores[start:start + n]
        start += n

    return padded


class LTRModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        self.encoder.config.use_cache = False

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        emb_dim = self.encoder.config.hidden_size

        self.scorer = ScoringHead(
            emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.scorer.to(dtype=next(self.encoder.parameters()).dtype)

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def score_flat(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.encode(input_ids, attention_mask)
        return self.scorer(emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_sizes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epsilon: float = 0.2,
        loss_normalization: str = "items",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        flat_scores = self.score_flat(input_ids, attention_mask)
        scores = _pad_group_scores(flat_scores, group_sizes)

        output = {
            "flat_scores": flat_scores,
            "scores": scores,
            "logits": scores,
        }

        if labels is not None:
            output["loss"] = pairwise_margin_ranking_loss_flat(
                flat_scores=flat_scores,
                labels=labels,
                group_sizes=group_sizes,
                epsilon=epsilon,
                normalization=loss_normalization,
            )

        return output



# ----------------------------
# Loss (same principle)
# ----------------------------

def pairwise_margin_ranking_loss_flat(
    flat_scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    epsilon: float = 0.2,
    normalization: str = "items",
) -> torch.Tensor:
    """
    Pairwise hinge ranking loss over variable-size groups.

    Lower label value means better item.
    If label_i < label_j, score_i should be greater than score_j.

    normalization:
        "pairs": divide by number of valid unequal-label pairs.
        "items": divide by number of forwarded items.
    """
    if normalization not in {"pairs", "items"}:
        raise ValueError(f"Unknown normalization: {normalization}")

    scores = flat_scores.float()
    device = scores.device
    dtype = scores.dtype

    total_loss = scores.new_zeros(())
    denom = scores.new_zeros(())

    start = 0
    sizes = group_sizes.detach().cpu().tolist()

    for b, n in enumerate(sizes):
        group_scores = scores[start:start + n]
        group_labels = labels[b, :n].to(device=device)

        if normalization == "items":
            denom = denom + float(n)

        if n >= 2:
            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)

            si = group_scores[idx_i]
            sj = group_scores[idx_j]

            li = group_labels[idx_i]
            lj = group_labels[idx_j]

            valid = (li != -100) & (lj != -100) & (li != lj)

            if valid.any():
                sign = torch.sign((lj - li).to(dtype))
                diff = si - sj

                losses = torch.relu(epsilon - sign * diff)
                total_loss = total_loss + losses[valid].sum()

                if normalization == "pairs":
                    denom = denom + valid.to(dtype).sum()

        start += n

    return total_loss / denom.clamp_min(1.0)

# ----------------------------
# Data collator
# ----------------------------
# Group level collator (using all of the training data)
@dataclass
class GroupAllPairsCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        flat_texts = []
        group_sizes = []

        max_group_size = max(len(f["texts"]) for f in features)
        padded_labels = []

        for f in features:
            texts = f["texts"]
            labels = f["labels"]

            group_sizes.append(len(texts))
            flat_texts.extend(texts)

            padded = list(labels) + [-100] * (max_group_size - len(labels))
            padded_labels.append(padded)

        tok = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.float32),
        }

# ----------------------------
# Trainer
# ----------------------------

class PairwiseLTRTrainer(Trainer):
    def __init__(
        self,
        epsilon: float = 0.2,
        loss_normalization: str = "items",
        win_rate_tokenizer=None,
        win_rate_max_length: Optional[int] = None,
        length_diagnostics: bool = False,
        length_plot_num_bins: int = 10,
        length_plot_max_pairs: int = 200000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon
        self.loss_normalization = loss_normalization
        self.win_rate_tokenizer = win_rate_tokenizer
        self.win_rate_max_length = win_rate_max_length
        self.length_diagnostics = length_diagnostics
        self.length_plot_num_bins = length_plot_num_bins
        self.length_plot_max_pairs = length_plot_max_pairs

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            group_sizes=inputs["group_sizes"],
            labels=inputs["labels"],
            epsilon=self.epsilon,
            loss_normalization=self.loss_normalization,
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if eval_dataset is None:
            return {}

        if self.win_rate_tokenizer is None or self.win_rate_max_length is None:
            raise ValueError(
                "win_rate_tokenizer and win_rate_max_length are required for evaluation."
            )

        metrics = evaluate_win_rate_distributed(
            model=self.model,
            dataset=eval_dataset,
            tokenizer=self.win_rate_tokenizer,
            max_length=self.win_rate_max_length,
            batch_size=max(1, self.args.per_device_eval_batch_size),
            collect_length_diagnostics=self.length_diagnostics,
            length_plot_num_bins=self.length_plot_num_bins,
            length_plot_max_pairs=self.length_plot_max_pairs,
            length_diag_output_dir=self.args.output_dir,
            length_diag_step=self.state.global_step,
            length_diag_epoch=self.state.epoch,
            length_diag_seed=getattr(self.args, "seed", 0),
        )

        metrics = {
            f"{metric_key_prefix}_{k}": v
            for k, v in metrics.items()
        }

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )

        return metrics

# Custom evaluation function
@torch.no_grad()
def evaluate_win_rate_distributed(
    model: nn.Module,
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batch_size: int = 4,
    collect_length_diagnostics: bool = False,
    length_plot_num_bins: int = 10,
    length_plot_max_pairs: int = 200000,
    length_diag_output_dir: Optional[str] = None,
    length_diag_step: Optional[int] = None,
    length_diag_epoch: Optional[float] = None,
    length_diag_seed: int = 0,
) -> Dict[str, float]:
    """
    Distributed evaluation.

    Main metric:
        win_rate over all unequal-label pairs.

    Optional length diagnostics:
    - How often the model prefers the shorter text.
    - Accuracy when the shorter text is truly better.
    - Accuracy when the longer text is truly better.
    - Training plot of:
        overall win rate,
        win rate when shorter text is better,
        win rate when longer text is better.
    - Per-evaluation plot of win rate versus relative length difference.

    Relative length difference is:
        abs(len_i - len_j) / max(len_i, len_j)

    Character length is computed with Python len(text), i.e. Unicode code points.
    """
    model.eval()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = next(model.parameters()).device
    dataset_len = len(dataset)

    if dataset_len == 0:
        return {
            "win_rate": 0.0,
            "correct_pairs": 0,
            "total_pairs": 0,
        }

    # -------------------------
    # Length diagnostic setup
    # -------------------------
    if collect_length_diagnostics:
        # [
        #   model_prefers_shorter_count,
        #   unequal_length_non_tie_score_count,
        #   true_shorter_better_count,
        #   unequal_length_count,
        #   correct_points_when_shorter_better,
        #   shorter_better_count,
        #   correct_points_when_longer_better,
        #   longer_better_count,
        # ]
        local_len_counts = torch.zeros(8, dtype=torch.float64, device=device)

        # For the relative-length plot.
        # Each row is:
        #     (relative_length_difference, correct_int)
        local_plot_pairs = []
        seen_plot_pairs = 0

        rng = random.Random(
            int(length_diag_seed)
            + 1_000_003 * int(rank)
            + 9_176 * int(length_diag_step or 0)
        )

        if length_plot_max_pairs < 0:
            local_plot_cap = None
        elif length_plot_max_pairs == 0:
            local_plot_cap = 0
        else:
            local_plot_cap = max(
                1,
                math.ceil(length_plot_max_pairs / max(world_size, 1)),
            )

        def maybe_add_plot_pair(row: tuple):
            """
            Reservoir-sample diagnostic pairs so very large eval sets do not explode RAM.

            row:
                (relative_length_difference, correct_int)
            """
            nonlocal seen_plot_pairs, local_plot_pairs

            if local_plot_cap == 0:
                return

            seen_plot_pairs += 1

            if local_plot_cap is None:
                local_plot_pairs.append(row)
                return

            if len(local_plot_pairs) < local_plot_cap:
                local_plot_pairs.append(row)
            else:
                replace_idx = rng.randrange(seen_plot_pairs)
                if replace_idx < local_plot_cap:
                    local_plot_pairs[replace_idx] = row

    # -------------------------
    # Distributed shard setup
    # -------------------------
    local_indices = list(range(rank, dataset_len, world_size))
    local_len = len(local_indices)

    local_len_tensor = torch.tensor([local_len], dtype=torch.long, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_len_tensor, op=dist.ReduceOp.MAX)

    max_local_len = int(local_len_tensor.item())

    padded_indices = local_indices + [-1] * (max_local_len - local_len)

    local_correct_points = 0.0
    local_total = 0

    # Diagnostics for score ties.
    local_score_tie_pairs = 0
    local_strict_correct_pairs = 0

    use_cuda_amp = torch.cuda.is_available() and device.type == "cuda"
    use_bf16_amp = use_cuda_amp and torch.cuda.is_bf16_supported()

    if dist.is_available() and dist.is_initialized():
        logger.info(
            f"[rank {rank}] eval dataset_len={dataset_len}, "
            f"local_len={local_len}, max_local_len={max_local_len}, "
            f"batch_size={batch_size}, "
            f"num_eval_batches={math.ceil(max_local_len / batch_size)}"
        )

    # -------------------------
    # Main eval loop
    # -------------------------
    for start in range(0, max_local_len, batch_size):
        batch_indices = padded_indices[start:start + batch_size]

        real_mask = [idx >= 0 for idx in batch_indices]
        fetch_indices = [idx if idx >= 0 else 0 for idx in batch_indices]

        rows = dataset[fetch_indices]

        texts_list = rows["texts"]
        labels_list = rows["labels"]

        group_sizes = [len(x) for x in texts_list]
        flat_texts = [t for group in texts_list for t in group]

        tok = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)

        gs = torch.tensor(group_sizes, dtype=torch.long, device=device)

        if use_cuda_amp:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16_amp else torch.float16,
                enabled=True,
            ):
                out = model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    group_sizes=gs,
                    labels=None,
                )
        else:
            out = model(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                group_sizes=gs,
                labels=None,
            )

        scores = out["scores"]

        for b, labels in enumerate(labels_list):
            if not real_mask[b]:
                continue

            texts = texts_list[b]

            n = len(labels)

            if n < 2:
                continue

            s = scores[b, :n]
            l = torch.tensor(labels, device=device)

            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)

            li = l[idx_i]
            lj = l[idx_j]
            si = s[idx_i]
            sj = s[idx_j]

            valid = li != lj

            if not valid.any():
                continue

            # -------------------------------------------------
            # Tie-aware pair scoring
            # -------------------------------------------------
            #
            # Lower label means better item.
            #
            # If scores are strictly ordered:
            #   1.0 point if the ordering is correct.
            #   0.0 points if the ordering is incorrect.
            #
            # If scores are exactly tied:
            #   0.5 points.
            #
            # This avoids the previous bug where si == sj implied
            # pred_i_better=False, which effectively awarded the win
            # to item j because of pair ordering.
            pred_i_better = si > sj
            true_i_better = li < lj

            score_tie = torch.isclose(
                si.float(),
                sj.float(),
                rtol=0.0,
                atol=1e-6,
            )

            strict_pair_correct = pred_i_better == true_i_better

            pair_points = strict_pair_correct.to(torch.float32)
            pair_points = torch.where(
                score_tie,
                torch.full_like(pair_points, 0.5),
                pair_points,
            )

            valid_pair_points = pair_points[valid]

            local_correct_points += float(valid_pair_points.sum().item())
            local_total += int(valid.sum().item())

            local_score_tie_pairs += int(score_tie[valid].sum().item())
            local_strict_correct_pairs += int(
                ((~score_tie) & strict_pair_correct & valid).sum().item()
            )

            # -------------------------
            # Optional length diagnostics
            # -------------------------
            if collect_length_diagnostics:
                text_lengths = torch.tensor(
                    [len(x) for x in texts],
                    dtype=torch.long,
                    device=device,
                )

                len_i = text_lengths[idx_i]
                len_j = text_lengths[idx_j]

                valid_positions = torch.nonzero(valid, as_tuple=False).flatten()

                unequal_length = valid & (len_i != len_j)

                # If scores tie exactly, do not include in "model prefers shorter"
                # denominator, because there is no strict score preference.
                unequal_length_non_tie_score = unequal_length & (~score_tie)

                pred_prefers_shorter = (
                    (pred_i_better & (len_i < len_j))
                    | ((~pred_i_better) & (len_j < len_i))
                )

                true_shorter_better = (
                    ((li < lj) & (len_i < len_j))
                    | ((lj < li) & (len_j < len_i))
                )

                shorter_better_mask = unequal_length & true_shorter_better
                longer_better_mask = unequal_length & (~true_shorter_better)

                local_len_counts[0] += float(
                    pred_prefers_shorter[unequal_length_non_tie_score].sum().item()
                )
                local_len_counts[1] += float(
                    unequal_length_non_tie_score.sum().item()
                )

                local_len_counts[2] += float(
                    true_shorter_better[unequal_length].sum().item()
                )
                local_len_counts[3] += float(unequal_length.sum().item())

                # These are now point sums, not integer counts.
                # Score ties contribute 0.5 points.
                local_len_counts[4] += float(
                    pair_points[shorter_better_mask].sum().item()
                )
                local_len_counts[5] += float(shorter_better_mask.sum().item())

                local_len_counts[6] += float(
                    pair_points[longer_better_mask].sum().item()
                )
                local_len_counts[7] += float(longer_better_mask.sum().item())

                # Relative length-difference plot data.
                #
                # rel_len_diff = abs(len_i - len_j) / max(len_i, len_j)
                #
                # This is bounded in [0, 1] and is less sensitive to the absolute
                # scale of text length.
                if local_plot_cap != 0:
                    for pos_t in valid_positions:
                        pos = int(pos_t.item())

                        len_i_val = int(len_i[pos].item())
                        len_j_val = int(len_j[pos].item())

                        denom = max(len_i_val, len_j_val, 1)
                        rel_len_diff = abs(len_i_val - len_j_val) / float(denom)

                        # This can be 0.0, 0.5, or 1.0.
                        point_value = float(pair_points[pos].item())

                        maybe_add_plot_pair(
                            (
                                float(rel_len_diff),
                                point_value,
                            )
                        )

    # -------------------------
    # Reduce main counts
    # -------------------------
    counts = torch.tensor(
        [
            local_correct_points,
            float(local_total),
            float(local_score_tie_pairs),
            float(local_strict_correct_pairs),
        ],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    correct_points = float(counts[0].item())
    total = int(round(float(counts[1].item())))
    score_tie_pairs = int(round(float(counts[2].item())))
    strict_correct_pairs = int(round(float(counts[3].item())))

    win_rate = correct_points / total if total > 0 else 0.0

    metrics = {
        "win_rate": win_rate,

        # Semantically correct name under tie-as-half-credit.
        "correct_points": correct_points,

        # Backward-compatible alias. This may now be fractional.
        "correct_pairs": correct_points,

        "strict_correct_pairs": strict_correct_pairs,
        "score_tie_pairs": score_tie_pairs,
        "score_tie_rate": score_tie_pairs / total if total > 0 else 0.0,

        "total_pairs": total,
    }

    # -------------------------
    # Reduce length diagnostics and write plots
    # -------------------------
    if collect_length_diagnostics:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_len_counts, op=dist.ReduceOp.SUM)

        len_counts = local_len_counts.detach().cpu().tolist()

        model_prefers_shorter = float(len_counts[0])
        unequal_len_non_tie = float(len_counts[1])

        true_shorter_better_count = float(len_counts[2])
        unequal_len = float(len_counts[3])

        correct_points_when_shorter_better = float(len_counts[4])
        total_shorter_better = float(len_counts[5])

        correct_points_when_longer_better = float(len_counts[6])
        total_longer_better = float(len_counts[7])

        metrics["model_prefers_shorter_rate"] = (
            model_prefers_shorter / unequal_len_non_tie
            if unequal_len_non_tie > 0
            else 0.0
        )

        metrics["true_shorter_better_rate"] = (
            true_shorter_better_count / unequal_len
            if unequal_len > 0
            else 0.0
        )

        metrics["win_rate_when_shorter_is_better"] = (
            correct_points_when_shorter_better / total_shorter_better
            if total_shorter_better > 0
            else 0.0
        )

        metrics["win_rate_when_longer_is_better"] = (
            correct_points_when_longer_better / total_longer_better
            if total_longer_better > 0
            else 0.0
        )

        metrics["correct_points_when_shorter_is_better"] = (
            correct_points_when_shorter_better
        )
        metrics["correct_points_when_longer_is_better"] = (
            correct_points_when_longer_better
        )

        metrics["pairs_where_shorter_is_better"] = int(round(total_shorter_better))
        metrics["pairs_where_longer_is_better"] = int(round(total_longer_better))

        # Gather plot pairs on rank 0.
        if length_plot_max_pairs != 0:
            if dist.is_available() and dist.is_initialized():
                gathered_plot_pairs = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_plot_pairs, local_plot_pairs)
            else:
                gathered_plot_pairs = [local_plot_pairs]

            if rank == 0:
                all_plot_pairs = []

                for rows_for_rank in gathered_plot_pairs:
                    if rows_for_rank:
                        all_plot_pairs.extend(rows_for_rank)

                # Deterministic global truncation after local reservoir sampling.
                if (
                    length_plot_max_pairs > 0
                    and len(all_plot_pairs) > length_plot_max_pairs
                ):
                    rng_global = random.Random(
                        int(length_diag_seed)
                        + 53_111 * int(length_diag_step or 0)
                    )
                    all_plot_pairs = rng_global.sample(
                        all_plot_pairs,
                        length_plot_max_pairs,
                    )

                _write_winrate_history_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    metrics=metrics,
                )

                _write_relative_length_winrate_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    diagnostic_pairs=all_plot_pairs,
                    num_bins=length_plot_num_bins,
                )

                logger.info(
                    f"Wrote length diagnostic plots with "
                    f"{len(all_plot_pairs)} diagnostic pairs."
                )
        else:
            if rank == 0:
                _write_winrate_history_plot(
                    output_dir=length_diag_output_dir,
                    step=length_diag_step,
                    epoch=length_diag_epoch,
                    metrics=metrics,
                )

    return metrics

def baseline_winrates(dataset: Dataset) -> Dict[str, float]:
    """
    Computes simple baselines over valid unequal-label pairs.

    Lower label means better item.

    Baselines:
        random_baseline:
            Expected accuracy of random choice between two items.

        first_item_baseline:
            Accuracy if we always predict item i is better than item j.

        shorter_text_baseline:
            Accuracy if we predict the shorter text is better.
            If lengths are equal, count expected random accuracy of 0.5.
    """
    total = 0
    first_correct = 0
    shorter_correct = 0.0

    for texts, labels in zip(dataset["texts"], dataset["labels"]):
        n = len(labels)

        for i in range(n - 1):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue

                total += 1

                i_better = labels[i] < labels[j]

                if i_better:
                    first_correct += 1

                len_i = len(texts[i])
                len_j = len(texts[j])

                if len_i < len_j:
                    shorter_correct += float(i_better)
                elif len_j < len_i:
                    shorter_correct += float(not i_better)
                else:
                    shorter_correct += 0.5

    if total == 0:
        return {
            "random_baseline": 0.0,
            "first_item_baseline": 0.0,
            "shorter_text_baseline": 0.0,
            "total_valid_pairs": 0,
        }

    return {
        "random_baseline": 0.5,
        "first_item_baseline": first_correct / total,
        "shorter_text_baseline": shorter_correct / total,
        "total_valid_pairs": total,
    }

def load_ltr_model(
    final_dir: str,
    attn_implementation: str = "sdpa",
) -> LTRModel:
    head_path = os.path.join(final_dir, "ltr_head.pt")
    head_state = torch.load(head_path, map_location="cpu")

    model = LTRModel(
        model_name=final_dir,
        hidden_dim=head_state["hidden_dim"],
        dropout=head_state["dropout"],
        attn_implementation=attn_implementation,
    )

    model.scorer.load_state_dict(head_state["scorer"])
    return model


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        "_orig_mod.",
        "module.",
        "_fsdp_wrapped_module.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def main():
    args = parse_args()
    set_seed(args.seed)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Critical for large model FSDP loading.
    # This tells Transformers/Accelerate to only load the full checkpoint on rank 0
    # and initialize empty weights on the other ranks before FSDP syncs/shards.
    if world_size > 1:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    if rank == 0:
        logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")

    # Data
    ds = d_f.format_custom_dataset(args.custom_dataset)
    ds = d_f.shuffle_and_transform_formatted_dataset(ds, seed=args.seed)

    if args.downsample_size is not None:
        ds = ds.select(range(min(args.downsample_size, len(ds))))

    split = ds.train_test_split(test_size=0.3, seed=args.seed)
    train_dataset = split["train"].shuffle(seed=args.seed)
    dev_test = split["test"].train_test_split(test_size=0.5, seed=args.seed)
    dev_dataset = dev_test["train"].shuffle(seed=args.seed)
    test_dataset = dev_test["test"].shuffle(seed=args.seed)

    if rank == 0:
        logger.info(train_dataset)

    if rank == 0:
        logger.info(f"train={len(train_dataset)} dev={len(dev_dataset)} test={len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LTRModel(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation,
    )

    if model.encoder.config.pad_token_id is None:
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    data_collator = GroupAllPairsCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )



    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    fsdp_layer_cls = args.fsdp_layer_cls

    if rank == 0:
        logger.info(f"use_cuda={use_cuda}")
        logger.info(f"use_bf16={use_bf16}")
        logger.info(f"FSDP transformer layer class={fsdp_layer_cls}")

    training_kwargs = dict(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            logging_strategy="steps",
            logging_first_step=True,
            save_strategy=args.save_strategy,
            eval_strategy=args.eval_strategy,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=use_cuda,
            bf16=use_bf16,
            fp16=False,
            report_to=[],
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            disable_tqdm=False,
    )

    use_fsdp = use_cuda
    if use_fsdp:
        fsdp_config = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
            "limit_all_gathers": True,
            "activation_checkpointing": False,
            "sync_module_states": world_size > 1,
            "cpu_ram_efficient_loading": world_size > 1,
            "cpu_offload": False,
        }

        if args.fsdp_layer_cls is not None:
            training_kwargs["fsdp"] = "full_shard auto_wrap"
            fsdp_config["transformer_layer_cls_to_wrap"] = args.fsdp_layer_cls
        else:
            training_kwargs["fsdp"] = "full_shard"
            logger.warning(
                "No fsdp_layer_cls provided. Using fsdp='full_shard' without auto_wrap."
            )

        training_kwargs["fsdp_config"] = fsdp_config
    else:
        logger.warning("FSDP disabled because CUDA is not available.")

    training_args = TrainingArguments(**training_kwargs)

    trainer = PairwiseLTRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        epsilon=args.epsilon,
        loss_normalization=args.loss_normalization,
        win_rate_tokenizer=tokenizer,
        win_rate_max_length=args.max_seq_len,
        length_diagnostics=args.length_diagnostics,
        length_plot_num_bins=args.length_plot_num_bins,
        length_plot_max_pairs=args.length_plot_max_pairs,
    )
    # Code that check whether all datatypes are the same, so that there are no issues furing training
    if rank == 0:
        dtype_counts = {}
        for name, param in model.named_parameters():
            dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + param.numel()
            if param.dtype == torch.float32:
                logger.info(f"FP32 parameter: {name}, shape={tuple(param.shape)}")

        logger.info(f"Parameter dtype counts: {dtype_counts}")

    trainer.train()

    # ------------------------------------------------------------
    # Save final model first
    # ------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")

    if rank == 0:
        os.makedirs(final_dir, exist_ok=True)

    trainer.accelerator.wait_for_everyone()

    # Drop optimizer/scheduler references before final full save.
    # Optimizer states can be huge and are not needed for inference.
    trainer.optimizer = None
    trainer.lr_scheduler = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    trainer.accelerator.wait_for_everyone()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    # Collect full model state dict from FSDP.
    state_dict = get_model_state_dict(
        trainer.model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    if rank == 0:
        cleaned_state_dict = {
            _strip_known_prefixes(k): v
            for k, v in state_dict.items()
        }

        encoder_state_dict = {
            k.removeprefix("encoder."): v
            for k, v in cleaned_state_dict.items()
            if k.startswith("encoder.")
        }

        scorer_state_dict = {
            k.removeprefix("scorer."): v
            for k, v in cleaned_state_dict.items()
            if k.startswith("scorer.")
        }

        if not encoder_state_dict:
            raise RuntimeError(
                "encoder_state_dict is empty. State dict keys were not parsed correctly. "
                f"Example keys: {list(cleaned_state_dict.keys())[:20]}"
            )

        if not scorer_state_dict:
            raise RuntimeError(
                "scorer_state_dict is empty. State dict keys were not parsed correctly. "
                f"Example keys: {list(cleaned_state_dict.keys())[:20]}"
            )

        # Save encoder using HF save_pretrained.
        unwrapped.encoder.save_pretrained(
            final_dir,
            state_dict=encoder_state_dict,
            safe_serialization=True,
            max_shard_size="2GB",
        )

        tokenizer.save_pretrained(final_dir)

        # Save custom ranking head separately.
        torch.save(
            {
                "scorer": scorer_state_dict,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            os.path.join(final_dir, "ltr_head.pt"),
        )

        logger.info(f"Saved final model to {final_dir}")

        del cleaned_state_dict
        del encoder_state_dict
        del scorer_state_dict

    del state_dict

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    trainer.accelerator.wait_for_everyone()

    # ------------------------------------------------------------
    # Test after saving
    # ------------------------------------------------------------
    metrics_test = evaluate_win_rate_distributed(
        model=trainer.model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        batch_size=max(1, args.per_device_eval_batch_size),
        collect_length_diagnostics=args.length_diagnostics,
        length_plot_num_bins=args.length_plot_num_bins,
        length_plot_max_pairs=args.length_plot_max_pairs,
        length_diag_output_dir=args.output_dir,
        length_diag_step=trainer.state.global_step,
        length_diag_epoch=trainer.state.epoch,
        length_diag_seed=args.seed,
    )

    trainer.accelerator.wait_for_everyone()

    if rank == 0:
        metrics_path = os.path.join(final_dir, "metrics.json")
        baselines = baseline_winrates(test_dataset)

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "test": metrics_test,
                    "baselines": baselines,
                },
                f,
                indent=2,
            )

        logger.info(f"Test metrics: {metrics_test}")
        logger.info(f"Baselines: {baselines}")
        logger.info(f"Saved metrics to {metrics_path}")

    trainer.accelerator.wait_for_everyone()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()