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
import torch.nn.functional as F

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
    TrainerCallback,
    set_seed,
)
from tqdm.auto import tqdm
import gc

# Your custom dataset module
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

    parser.add_argument("--parallel_mode", choices=["fsdp"], default="fsdp")
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
    parser.add_argument("--pairs_per_group", type=int, default=1)

    parser.add_argument(
        "--use_all_pairs",
        action="store_true",
        help="If set, train on all valid pairs within each group instead of sampled pairs.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help=(
            "Attention implementation. Use sdpa for broad compatibility. "
            "flash_attention_2 is faster but not supported by every model/environment."
        ),
    )

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--use_torch_compile", action="store_true")

    parser.add_argument("--fsdp_layer_cls", type=str, default=None)

    return parser.parse_args()


# ----------------------------
# Model
# ----------------------------
class ScoringHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
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
        # x: [N, H]
        return self.net(x).squeeze(-1)  # [N]


# Helper function
def _pad_group_scores(
    flat_scores: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Converts flat scores [sum(group_sizes)] into padded [B, Kmax].
    """
    sizes = group_sizes.detach().cpu().tolist()
    batch_size = len(sizes)
    max_k = max(sizes)

    padded = flat_scores.new_zeros((batch_size, max_k))

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
        attn_implementation: str = "flash_attention_2",
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

        encoder_dtype = next(self.encoder.parameters()).dtype
        self.scorer.to(dtype=encoder_dtype)

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
        group_sizes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pair_targets: Optional[torch.Tensor] = None,
        pair_weights: Optional[torch.Tensor] = None,
        epsilon: float = 0.2,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        flat_scores = self.score_flat(input_ids, attention_mask)

        output = {
            "flat_scores": flat_scores,
            "logits": flat_scores,
        }

        # -------------------------
        # Sampled-pair mode
        # -------------------------
        if pair_targets is not None:
            if flat_scores.numel() != pair_targets.numel() * 2:
                raise ValueError(
                    f"Expected exactly 2 scores per pair_target. "
                    f"Got {flat_scores.numel()} scores and "
                    f"{pair_targets.numel()} pair targets."
                )

            s1 = flat_scores[0::2]
            s2 = flat_scores[1::2]

            pair_targets = pair_targets.to(
                device=flat_scores.device,
                dtype=flat_scores.dtype,
            )

            pair_loss = F.relu(epsilon - pair_targets * (s1 - s2)).float()

            if pair_weights is not None:
                pair_weights = pair_weights.to(
                    device=flat_scores.device,
                    dtype=pair_loss.dtype,
                )
                loss = (pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
            else:
                loss = pair_loss.mean()

            output["loss"] = loss
            return output

        # -------------------------
        # All-pairs group mode
        # -------------------------
        if group_sizes is not None:
            scores = _pad_group_scores(flat_scores, group_sizes)

            output["scores"] = scores
            output["logits"] = scores

            if labels is not None:
                labels = labels.to(device=scores.device)
                loss = pairwise_margin_ranking_loss(
                    scores=scores,
                    labels=labels,
                    epsilon=epsilon,
                )
                output["loss"] = loss

            return output

        return output



# ----------------------------
# Loss (same principle)
# ----------------------------
def pairwise_margin_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Lower label value means better item.
    For pair i, j:
        if labels[i] < labels[j], score[i] should be greater than score[j].
    """
    scores = scores.float()
    labels = labels.to(device=scores.device)

    B, K = scores.shape
    device = scores.device

    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=device)

    si = scores[:, idx_i]
    sj = scores[:, idx_j]

    li = labels[:, idx_i]
    lj = labels[:, idx_j]

    valid = (li != -100) & (lj != -100) & (li != lj)

    sign = torch.sign((lj - li).float())
    diff = si - sj

    pair_loss = torch.relu(epsilon - sign * diff)
    pair_loss = pair_loss * valid.float()

    denom = valid.float().sum().clamp_min(1.0)
    return pair_loss.sum() / denom


# ----------------------------
# Data collator
# ----------------------------
@dataclass
class PairSamplingCollator:
    tokenizer: Any
    max_length: int
    pairs_per_group: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        flat_texts = []
        pair_targets = []
        pair_weights = []

        for f in features:
            texts = f["texts"]
            labels = f["labels"]

            valid_pairs = [
                (i, j)
                for i in range(len(texts))
                for j in range(i + 1, len(texts))
                if labels[i] != labels[j]
            ]

            if not valid_pairs:
                continue

            for _ in range(self.pairs_per_group):
                i, j = random.choice(valid_pairs)

                if random.random() < 0.5:
                    i, j = j, i

                target = 1.0 if labels[i] < labels[j] else -1.0

                flat_texts.append(texts[i])
                flat_texts.append(texts[j])
                pair_targets.append(target)
                pair_weights.append(1.0)

        if len(pair_targets) == 0:
            # Dummy zero-weight pair. This prevents crashing but contributes no loss.
            texts = features[0]["texts"]
            if len(texts) == 1:
                flat_texts = [texts[0], texts[0]]
            else:
                flat_texts = [texts[0], texts[1]]

            pair_targets = [1.0]
            pair_weights = [0.0]

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
            "pair_targets": torch.tensor(pair_targets, dtype=torch.float32),
            "pair_weights": torch.tensor(pair_weights, dtype=torch.float32),
        }
    
# Group level collator (using all of the training data)

@dataclass
class GroupAllPairsCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of groups.

        Each feature is expected to contain:
            {
                "texts": List[str],
                "labels": List[int or float]
            }

        Returns:
            input_ids: [sum(group_sizes), seq_len]
            attention_mask: [sum(group_sizes), seq_len]
            group_sizes: [B]
            labels: [B, Kmax], padded with -100
        """
        flat_texts = []
        group_sizes = []

        max_group_size = max(len(f["texts"]) for f in features)
        padded_labels = []

        for f in features:
            texts = f["texts"]
            labels = f["labels"]

            group_size = len(texts)
            group_sizes.append(group_size)
            flat_texts.extend(texts)

            padded = list(labels) + [-100] * (max_group_size - group_size)
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
class TqdmProgressCallback(TrainerCallback):
    """
    Explicit tqdm progress bar for Trainer.

    Shows optimizer-step progress, ETA, loss, LR, epoch, and eval metrics.
    Only renders on global rank 0.
    """

    def __init__(self):
        self.training_bar = None
        self.last_global_step = 0
        self.postfix = {}

    def _is_main_process(self, args, state) -> bool:
        if hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0

        return True

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_main_process(args, state):
            return control

        total = state.max_steps if state.max_steps and state.max_steps > 0 else None

        self.training_bar = tqdm(
            total=total,
            initial=state.global_step,
            desc="Training",
            dynamic_ncols=True,
            leave=True,
        )
        self.last_global_step = state.global_step
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.training_bar is None:
            return control

        delta = state.global_step - self.last_global_step
        if delta > 0:
            self.training_bar.update(delta)
            self.last_global_step = state.global_step

        optimizer = kwargs.get("optimizer", None)
        if optimizer is not None and len(optimizer.param_groups) > 0:
            self.postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"

        if state.epoch is not None:
            self.postfix["epoch"] = f"{state.epoch:.2f}"

        self.training_bar.set_postfix(self.postfix)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.training_bar is None or logs is None:
            return control

        for key in [
            "loss",
            "learning_rate",
            "eval_win_rate",
            "eval_loss",
            "train_loss",
        ]:
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    if key == "learning_rate":
                        self.postfix["lr"] = f"{value:.2e}"
                    else:
                        self.postfix[key] = f"{value:.4f}"
                else:
                    self.postfix[key] = value

        self.training_bar.set_postfix(self.postfix)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            if state.global_step > self.last_global_step:
                self.training_bar.update(state.global_step - self.last_global_step)

            self.training_bar.close()
            self.training_bar = None

        return control

class PairwiseLTRTrainer(Trainer):
    def __init__(
        self,
        epsilon: float = 0.2,
        win_rate_tokenizer=None,
        win_rate_max_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.win_rate_tokenizer = win_rate_tokenizer
        self.win_rate_max_length = win_rate_max_length

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = dict(inputs)

        # Sampled-pair mode.
        if "pair_targets" in inputs:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pair_targets=inputs["pair_targets"],
                pair_weights=inputs.get("pair_weights"),
                epsilon=self.epsilon,
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        # All-pairs group mode.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            group_sizes=inputs["group_sizes"],
            labels=inputs["labels"],
            epsilon=self.epsilon,
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
                "win_rate_tokenizer and win_rate_max_length must be provided "
                "to PairwiseLTRTrainer in order to use custom win-rate evaluation."
            )

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        logger.info(f"[rank {rank}] Entering custom evaluation")

        with self.accelerator.autocast():
            metrics = evaluate_win_rate_distributed(
                model=self.model,
                dataset=eval_dataset,
                tokenizer=self.win_rate_tokenizer,
                max_length=self.win_rate_max_length,
                batch_size=max(1, self.args.per_device_eval_batch_size),
            )

        logger.info(f"[rank {rank}] Finished custom evaluation: {metrics}")

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
) -> Dict[str, float]:
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

    # Real local shard for this rank.
    local_indices = list(range(rank, dataset_len, world_size))
    local_len = len(local_indices)

    # Find max local shard length across ranks.
    local_len_tensor = torch.tensor([local_len], dtype=torch.long, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_len_tensor, op=dist.ReduceOp.MAX)

    max_local_len = int(local_len_tensor.item())

    # Pad with dummy index -1 so all ranks have same number of eval items.
    padded_indices = local_indices + [-1] * (max_local_len - local_len)

    local_correct = 0
    local_total = 0

    use_cuda_amp = torch.cuda.is_available() and device.type == "cuda"
    use_bf16_amp = use_cuda_amp and torch.cuda.is_bf16_supported()

    if dist.is_available() and dist.is_initialized():
        logger.info(
            f"[rank {rank}] eval dataset_len={dataset_len}, "
            f"local_len={local_len}, max_local_len={max_local_len}, "
            f"batch_size={batch_size}, "
            f"num_eval_batches={math.ceil(max_local_len / batch_size)}"
        )

    # All ranks now execute exactly the same number of loop iterations.
    for start in range(0, max_local_len, batch_size):
        batch_indices = padded_indices[start:start + batch_size]

        # Mark which groups are real and which are padding.
        real_mask = [idx >= 0 for idx in batch_indices]

        # Replace dummy indices with a valid sample so tokenizer/model forward works.
        # These dummy groups will be ignored for metrics.
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

        # Keep eval dtype consistent with training where possible.
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
            # Ignore padded dummy examples.
            if not real_mask[b]:
                continue

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

            if valid.any():
                pred_i_better = si > sj
                true_i_better = li < lj

                local_correct += int(
                    (pred_i_better == true_i_better)[valid].sum().item()
                )
                local_total += int(valid.sum().item())

    counts = torch.tensor(
        [local_correct, local_total],
        dtype=torch.long,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    correct = int(counts[0].item())
    total = int(counts[1].item())

    win_rate = correct / total if total > 0 else 0.0

    return {
        "win_rate": win_rate,
        "correct_pairs": correct,
        "total_pairs": total,
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

    if args.use_torch_compile:
        model = torch.compile(model)

    # Choosing the data collator depedning on if we are sampling pairs or using all possible pairs
    if args.use_all_pairs:
        if rank == 0:
            logger.info("Using all-pairs group collator.")

        data_collator = GroupAllPairsCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
    else:
        if rank == 0:
            logger.info("Using sampled-pair collator.")

        data_collator = PairSamplingCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            pairs_per_group=args.pairs_per_group,
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
        prediction_loss_only=True,

        # Disable HF's default progress callback because we add our own below.
        disable_tqdm=True,
    )

    use_fsdp = args.parallel_mode == "fsdp" and use_cuda

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

        if fsdp_layer_cls is not None:
            training_kwargs["fsdp"] = "full_shard auto_wrap"
            fsdp_config["transformer_layer_cls_to_wrap"] = fsdp_layer_cls
        else:
            logger.warning(
                "Could not infer transformer_layer_cls_to_wrap. "
                "Falling back to fsdp='full_shard' without auto_wrap. "
                "This is more compatible but may use more memory for large models."
            )
            training_kwargs["fsdp"] = "full_shard"

        training_kwargs["fsdp_config"] = fsdp_config
    else:
        if rank == 0:
            logger.warning(
                "FSDP is disabled because CUDA is not available or parallel_mode is not fsdp."
            )

    training_args = TrainingArguments(**training_kwargs)

    trainer = PairwiseLTRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        epsilon=args.epsilon,
        win_rate_tokenizer=tokenizer,
        win_rate_max_length=args.max_seq_len,
        callbacks=[TqdmProgressCallback()],
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
            k.removeprefix("_orig_mod."): v
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
    )

    trainer.accelerator.wait_for_everyone()

    if rank == 0:
        metrics_path = os.path.join(final_dir, "metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "test": metrics_test,
                },
                f,
                indent=2,
            )

        logger.info(f"Test metrics: {metrics_test}")
        logger.info(f"Saved metrics to {metrics_path}")

    trainer.accelerator.wait_for_everyone()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()