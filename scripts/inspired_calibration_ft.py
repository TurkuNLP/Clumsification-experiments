# This script has been co-created, refactored, and cleaned using GPT 5.6.
# vibe coded

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ.setdefault("WANDB_MODE", "disabled")

class EvaluationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
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


def load_auto_model_with_dtype(
    model_dir: str,
    attn_implementation: str,
    dtype: torch.dtype,
):
    """
    Compatible with both newer Transformers `dtype=...` and older
    `torch_dtype=...`.
    """
    kwargs = dict(
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    try:
        return AutoModel.from_pretrained(
            model_dir,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            **kwargs,
        )


class CoheSentiaFEModel(nn.Module):
    """
    Loads the encoder + evaluation head saved by your FE training code.

    Expected model_dir contents:
      - config/tokenizer/model files from save_pretrained()
      - fe_head.pt with:
          {
            "hidden_dim": ...,
            "dropout": ...,
            "evaluation_head": state_dict
          }
    """

    def __init__(
        self,
        model_dir: str,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.model_dir = model_dir
        self.dtype = dtype

        head_path = os.path.join(model_dir, "fe_head.pt")
        if not os.path.exists(head_path):
            from clumsification_code.compat.fe_checkpoints import find_legacy_head

            head_path = find_legacy_head(model_dir)

        self.encoder = load_auto_model_with_dtype(
            model_dir=model_dir,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )

        self.encoder.config.use_cache = False

        if hasattr(self.encoder, "gradient_checkpointing_enable") and not freeze_encoder:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        try:
            head_state = torch.load(head_path, map_location="cpu", weights_only=False)
        except TypeError:
            head_state = torch.load(head_path, map_location="cpu")

        if "evaluation_head" not in head_state:
            from clumsification_code.compat.fe_checkpoints import normalize_legacy_head_state

            head_state = normalize_legacy_head_state(head_state)

        hidden_dim = head_state["hidden_dim"]
        dropout = head_state["dropout"]
        evaluation_head_state = head_state["evaluation_head"]

        # Compatible with either "net.0.weight" or "evaluation_head.net.0.weight" keys.
        evaluation_head_state = {
            k[len("evaluation_head."):] if k.startswith("evaluation_head.") else k: v
            for k, v in evaluation_head_state.items()
        }

        emb_dim = self.encoder.config.hidden_size

        self.evaluation_head = EvaluationHead(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.evaluation_head.load_state_dict(evaluation_head_state, strict=True)

        # Keep evaluation_head in same dtype as encoder during fine-tuning.
        self.evaluation_head.to(dtype=dtype)

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        emb = self.mean_pool(out.last_hidden_state, attention_mask)

        evaluation_head_dtype = next(self.evaluation_head.parameters()).dtype
        emb = emb.to(dtype=evaluation_head_dtype)

        scores = self.evaluation_head(emb)

        return {
            "logits": scores,
            "scores": scores,
        }


# ---------------------------------------------------------------------
# CoheSentia loading
# ---------------------------------------------------------------------


def iter_json_entries(data):
    if isinstance(data, dict):
        return data.values()
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON top-level type: {type(data)}")


def load_cohesentia_json(path: str) -> Tuple[List[str], List[float]]:
    """
    Supports the format used in your eval code:

        entry["Text"]
        entry["HolisticData"]["consensus_score"]

    Also accepts a few fallback key names for convenience.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    labels: List[float] = []

    for entry in iter_json_entries(data):
        if not isinstance(entry, dict):
            continue

        text = entry.get("Text", None)
        if text is None:
            text = entry.get("text", None)

        score = None

        holistic = entry.get("HolisticData", None)
        if isinstance(holistic, dict) and "consensus_score" in holistic:
            score = holistic["consensus_score"]

        if score is None:
            for key in ["consensus_score", "score", "label", "labels"]:
                if key in entry:
                    score = entry[key]
                    break

        if text is None or score is None:
            continue

        text = str(text).strip()

        try:
            score = float(score)
        except Exception:
            continue

        if not text or not math.isfinite(score):
            continue

        texts.append(text)
        labels.append(score)

    if len(texts) == 0:
        raise ValueError(f"No valid CoheSentia examples found in {path}")

    return texts, labels


def build_tokenized_dataset(
    json_path: str,
    tokenizer,
    max_length: int,
) -> Dataset:
    texts, labels = load_cohesentia_json(json_path)

    ds = Dataset.from_dict(
        {
            "text": texts,
            "labels": labels,
        }
    )

    def tokenize_batch(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        tok["labels"] = batch["labels"]
        return tok

    ds = ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {json_path}",
    )

    return ds


class ScalarTextCollator:
    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = torch.tensor(
            [float(f.pop("labels")) for f in features],
            dtype=torch.float32,
        )

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------
# Pairwise scalar fine-tuning loss
# ---------------------------------------------------------------------


def scalar_pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    scale: float = 1.0,
    min_label_delta: float = 0.0,
    weight_by_label_delta: bool = True,
) -> torch.Tensor:
    """
    In-batch pairwise ranking loss.

    For every pair i, j where label_i > label_j + min_label_delta,
    encourages score_i > score_j.

    Loss:
        softplus(-scale * (score_i - score_j))

    If weight_by_label_delta=True, pairs with larger human-score gaps get
    larger weight.
    """
    scores = scores.float().view(-1)
    labels = labels.float().view(-1)

    label_delta = labels[:, None] - labels[None, :]
    score_delta = scores[:, None] - scores[None, :]

    pair_mask = label_delta > min_label_delta

    if not pair_mask.any():
        # Zero loss with gradient connection.
        return scores.sum() * 0.0

    pair_losses = F.softplus(-scale * score_delta[pair_mask])

    if weight_by_label_delta:
        weights = label_delta[pair_mask].abs().float()
        weights = weights / torch.clamp(weights.mean(), min=1e-6)
        pair_losses = pair_losses * weights

    return pair_losses.mean()


class CoheSentiaFineTuneTrainer(Trainer):
    def __init__(
        self,
        pairwise_scale: float = 1.0,
        min_label_delta: float = 0.0,
        weight_by_label_delta: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pairwise_scale = pairwise_scale
        self.min_label_delta = min_label_delta
        self.weight_by_label_delta = weight_by_label_delta

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs["labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

        scores = outputs["scores"]

        loss = scalar_pairwise_ranking_loss(
            scores=scores,
            labels=labels,
            scale=self.pairwise_scale,
            min_label_delta=self.min_label_delta,
            weight_by_label_delta=self.weight_by_label_delta,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite CoheSentia fine-tuning loss: {loss.item()}"
            )

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def compute_scalar_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds).reshape(-1).astype(np.float64)
    labels = np.asarray(labels).reshape(-1).astype(np.float64)

    mask = np.isfinite(preds) & np.isfinite(labels)
    preds = preds[mask]
    labels = labels[mask]

    metrics = {
        "n": int(len(labels)),
    }

    if len(labels) < 2:
        metrics.update(
            {
                "spearman": float("nan"),
                "pearson": float("nan"),
                "pairwise_acc": float("nan"),
            }
        )
        return metrics

    if np.all(preds == preds[0]):
        spearman = float("nan")
    else:
        spearman = float(spearmanr(labels, preds).correlation)

    if np.std(preds) == 0 or np.std(labels) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(labels, preds)[0, 1])

    # Full pairwise tie-aware accuracy on eval set.
    label_delta = labels[:, None] - labels[None, :]
    pred_delta = preds[:, None] - preds[None, :]

    mask_pairs = label_delta > 0

    if mask_pairs.any():
        pairwise_acc = float(np.mean(pred_delta[mask_pairs] > 0))
    else:
        pairwise_acc = float("nan")

    metrics.update(
        {
            "spearman": spearman,
            "pearson": pearson,
            "pairwise_acc": pairwise_acc,
        }
    )

    return metrics


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------


def save_fe_model(
    trainer: Trainer,
    tokenizer,
    output_dir: str,
):
    """
    Saves in the same style expected by your benchmark code:

        output_dir/
          config.json
          model.safetensors / pytorch_model.bin
          tokenizer files
          fe_head.pt
    """
    os.makedirs(output_dir, exist_ok=True)

    model = trainer.model

    try:
        model = trainer.accelerator.unwrap_model(model)
    except Exception:
        pass

    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    evaluation_head_state = {
        k: v.detach().cpu()
        for k, v in model.evaluation_head.state_dict().items()
    }

    head_payload = {
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        "evaluation_head": evaluation_head_state,
    }

    torch.save(head_payload, os.path.join(output_dir, "fe_head.pt"))

    print(f"Saved CoheSentia-fine-tuned model to: {output_dir}")


# ---------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------


def parse_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()

    if dtype_name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16

    if dtype_name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_training_args(args) -> TrainingArguments:
    """
    Handles Transformers versions that use either `eval_strategy`
    or `evaluation_strategy`.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kwargs = dict(
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
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[],
        bf16=args.dtype in {"auto", "bf16", "bfloat16"} and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.dtype in {"fp16", "float16", "half"} and torch.cuda.is_available(),
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=True,
        disable_tqdm=False,
    )

    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[strategy_key] = args.eval_strategy

    return TrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing FE model on CoheSentia scalar ratings."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Existing trained final model dir containing encoder/tokenizer files and fe_head.pt.",
    )

    parser.add_argument(
        "--train_json",
        type=str,
        default="data/benchmarks/CohesentiaTrainData.json",
        help="CoheSentia training JSON.",
    )

    parser.add_argument(
        "--eval_json",
        type=str,
        default=None,
        help="Optional CoheSentia eval JSON. Prefer test-only if using this.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the CoheSentia-fine-tuned model will be saved.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Use 'flash_attention_2' if your model/GPU supports it.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Only train the evaluation head. Useful as a safer final calibration step.",
    )

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--eval_strategy", type=str, default="no")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument(
        "--pairwise_scale",
        type=float,
        default=1.0,
        help="Scale inside softplus(-scale * score_delta).",
    )

    parser.add_argument(
        "--min_label_delta",
        type=float,
        default=0.0,
        help="Ignore pairs with human-score difference <= this value.",
    )

    parser.add_argument(
        "--no_weight_by_label_delta",
        action="store_true",
        help="Disable weighting pair losses by absolute human-score difference.",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dtype = parse_dtype(args.dtype)

    print("===== CoheSentia FE fine-tuning =====")
    print(f"Base model dir: {args.model_dir}")
    print(f"Train JSON:     {args.train_json}")
    print(f"Eval JSON:      {args.eval_json}")
    print(f"Output dir:     {args.output_dir}")
    print(f"dtype:          {dtype}")
    print(f"freeze_encoder: {args.freeze_encoder}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CoheSentiaFEModel(
        model_dir=args.model_dir,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
        freeze_encoder=args.freeze_encoder,
    )

    if model.encoder.config.pad_token_id is None:
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = build_tokenized_dataset(
        json_path=args.train_json,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    eval_dataset = None
    if args.eval_json is not None:
        eval_dataset = build_tokenized_dataset(
            json_path=args.eval_json,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval examples:  {len(eval_dataset)}")

    collator = ScalarTextCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = build_training_args(args)

    trainer = CoheSentiaFineTuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_scalar_metrics if eval_dataset is not None else None,
        pairwise_scale=args.pairwise_scale,
        min_label_delta=args.min_label_delta,
        weight_by_label_delta=not args.no_weight_by_label_delta,
    )

    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print("Final CoheSentia eval metrics:")
        print(json.dumps(metrics, indent=2))

    # Save only on main process.
    if trainer.is_world_process_zero():
        save_fe_model(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
        )

    try:
        trainer.accelerator.wait_for_everyone()
    except Exception:
        pass


if __name__ == "__main__":
    main()
