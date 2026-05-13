# Claude 4.6 has been used to create part of this code

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import destroy_process_group
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
)

import sentence_transformers.sentence_transformer.modules as models

from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
import logging
from typing import Iterable, Dict
import sys
import os
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
import dataset_functions as d_f

os.environ["WANDB_MODE"] = "disabled"

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("max_seq_len", type=int)
    parser.add_argument("custom_dataset")
    parser.add_argument("output_dir")
    parser.add_argument("downsample_size", nargs="?", type=int, default=None)

    parser.add_argument(
        "--parallel_mode",
        choices=["ddp", "model_parallel"],
        default="ddp",
        help="ddp = one full model per GPU; model_parallel = shard model with device_map=auto (use nproc-per-node=1).",
    )
    return parser.parse_args()

# Trying out the scoring head as a module instead of a separate object
class ScoringHead(models.Module):
    """
    Sentence-Transformers compatible scoring head.
    Takes `features["sentence_embedding"]` and writes `features["score"]`.
    """

    def __init__(self, input_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        x = features["sentence_embedding"]          # [bs, dim]
        score = self.net(x).squeeze(-1)             # [bs]
        features["score"] = score
        return features

    def get_sentence_embedding_dimension(self) -> int:
        # This module doesn't change embedding dimensionality;
        # the pooled embedding remains available in `features`.
        return self.input_dim

    def get_config_dict(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def save(self, output_path: str, *args, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)

        # Save config
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f)

        # Save weights
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str):
        # Load config
        with open(os.path.join(input_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        module = ScoringHead(**config)

        # Load weights
        state_dict = torch.load(
            os.path.join(input_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        module.load_state_dict(state_dict)
        return module

# ──────────────────────────────────────────────
# 1.  Custom Pairwise Ranking Loss
# ──────────────────────────────────────────────
class PairwiseRankingLoss(nn.Module):
    """
    Pairwise margin ranking loss over n candidates per example.
    Expects the SentenceTransformer model to output:
      - "sentence_embedding"
      - "score"  (produced by ScoringHead module)
    """
    def __init__(self, model: SentenceTransformer, epsilon: float = 1.0):
        super().__init__()
        self.model = model
        self.epsilon = epsilon

    @property
    def citation(self) -> str:
        return ""

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        n = len(sentence_features)

        if labels.dim() == 1:
            raise ValueError(
                f"Expected labels of shape (batch_size, {n}), got {labels.shape}."
            )
        if labels.shape[1] != n:
            raise ValueError(
                f"Label width {labels.shape[1]} != number of sentence columns {n}."
            )

        # Run each sentence column through shared model once
        scores = []
        for sf in sentence_features:
            out = self.model(sf)
            if "score" not in out:
                raise KeyError(
                    "Model output missing 'score'. Ensure ScoringHead is included in SentenceTransformer modules."
                )
            scores.append(out["score"])  # shape [bs]

        scores = torch.stack(scores, dim=1)  # [bs, n]
        labels = labels.float()

        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=scores.device)

        scores_i = scores[:, idx_i]
        scores_j = scores[:, idx_j]
        labels_i = labels[:, idx_i]
        labels_j = labels[:, idx_j]

        sign = torch.sign(labels_j - labels_i)   # +1 if i better than j, -1 otherwise
        diff = scores_j - scores_i
        pair_loss = torch.relu(sign * diff + self.epsilon)

        mask = (labels_i != labels_j).float()
        pair_loss = pair_loss * mask

        valid_pairs = mask.sum()
        if valid_pairs.item() > 0:
            return pair_loss.sum() / valid_pairs
        return (scores * 0).sum()  # graph-connected zero



# ──────────────────────────────────────────────
# 2.  Win-Rate Evaluator
# ──────────────────────────────────────────────
class PairwiseWinRateEvaluator(SentenceEvaluator):
    def __init__(
        self,
        texts_list: list[list[str]],
        labels_list: list[list[int]],
        name: str = "pairwise_win_rate",
        batch_size: int = 32,
    ):
        super().__init__()
        self.texts_list = texts_list
        self.labels_list = labels_list
        self.name = name
        self.batch_size = batch_size
        self.primary_metric = "win_rate"

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        model.eval()

        flat_texts = []
        boundaries = [0]
        for texts in self.texts_list:
            flat_texts.extend(texts)
            boundaries.append(boundaries[-1] + len(texts))

        # We need embeddings first, then run ScoringHead module
        embeddings = model.encode(
            flat_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Apply only the final ScoringHead module on embeddings
        # Assumes ScoringHead is last module
        scoring_module = model._modules[list(model._modules.keys())[-1]]
        if not isinstance(scoring_module, ScoringHead):
            raise TypeError("Expected last module to be ScoringHead.")

        scoring_module = scoring_module.to(embeddings.device)
        with torch.no_grad():
            scored = scoring_module({"sentence_embedding": embeddings})
            flat_scores = scored["score"]  # [total_texts]

        correct = 0
        total = 0

        for idx, labels in enumerate(self.labels_list):
            start, end = boundaries[idx], boundaries[idx + 1]
            scores = flat_scores[start:end]  # [n]
            n = len(labels)

            labels_t = torch.tensor(labels, device=scores.device).float()
            idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=scores.device)

            li, lj = labels_t[idx_i], labels_t[idx_j]
            si, sj = scores[idx_i], scores[idx_j]

            valid = li != lj
            total += int(valid.sum().item())

            correct_mask = ((li < lj) & (si > sj)) | ((lj < li) & (sj > si))
            correct += int((correct_mask & valid).sum().item())

        win_rate = correct / total if total > 0 else 0.0
        metrics = {
            f"{self.name}_win_rate": win_rate,
            f"{self.name}_correct_pairs": correct,
            f"{self.name}_total_pairs": total,
        }

        logger.info(
            "[%s] epoch=%d steps=%d win_rate=%.4f (%d / %d pairs)",
            self.name, epoch, steps, win_rate, correct, total
        )

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"{self.name}_results.txt"), "a") as f:
                f.write(
                    f"epoch={epoch} steps={steps} win_rate={win_rate:.6f} "
                    f"correct={correct} total={total}\n"
                )

        model.train()
        return metrics


# ──────────────────────────────────────────────
# 3.  Dataset helper
# ──────────────────────────────────────────────
def expand_dataset_for_trainer(ds: Dataset) -> Dataset:
    """
    Convert a dataset with columns ``texts`` (list[str]) and ``labels``
    (list[int]) into the flat ``sentence_0, sentence_1, …, label``
    format that the SentenceTransformerTrainer data collator expects.

    Every row in ``ds`` must have the **same** number of texts.  The
    function verifies this at startup and raises early if violated.

    The ``label`` column is kept as a list[int] (one int per sentence);
    the data collator will stack these into a (batch_size, n) tensor.
    """
    all_texts = ds["texts"]    # list of list[str]
    all_labels = ds["labels"]  # list of list[int]

    n_per_row = len(all_texts[0])

    # Validate
    for row_idx, (texts, labels) in enumerate(zip(all_texts, all_labels)):
        if len(texts) != n_per_row:
            raise ValueError(
                f"Row {row_idx} has {len(texts)} texts, expected {n_per_row}."
            )
        if len(labels) != n_per_row:
            raise ValueError(
                f"Row {row_idx} has {len(labels)} labels, expected {n_per_row}."
            )

    new_data = {}
    for k in range(n_per_row):
        new_data[f"sentence_{k}"] = [row[k] for row in all_texts]
    new_data["label"] = all_labels

    return Dataset.from_dict(new_data)


# ──────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────
def main(args):

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    

    MODEL_NAME = args.model_name
    MAX_SEQ_LEN = args.max_seq_len
    CUSTOM_DATASET = args.custom_dataset
    OUTPUT_DIR = args.output_dir
    #If not downsampling, then use a larger batch size
    BS = 16
    if args.parallel_mode != "ddp":
        BS = 8
    #Optional downsampling for testing purposes
    DOWNSAMPLE_SIZE = args.downsample_size  # define once
    if DOWNSAMPLE_SIZE:
        if DOWNSAMPLE_SIZE == 50:
            BS = 2

    SPLIT_SEED = 42

    # ──────────────────────────────────────────────
    # Create training arguments EARLY — this triggers
    # HuggingFace's internal dist.init_process_group()
    # with the correct backend and configuration that
    # the Trainer's dataloader setup depends on.
    # ──────────────────────────────────────────────
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=BS,
        learning_rate=1e-5,
        warmup_steps=10,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        eval_strategy="epoch",
        fp16=False,
        bf16=True,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0
    )


    if args.parallel_mode == "ddp":
        assert world_size > 1, "DDP mode expects torchrun with nproc-per-node>1"
        assert dist.is_initialized(), "DDP mode requires initialized process group"
    else:
        # model_parallel mode
        if world_size != 1 and rank == 0:
            logger.warning("model_parallel mode should be launched with nproc-per-node=1; got WORLD_SIZE=%d", world_size)

    if rank == 0:
        logger.info(
            "dist.is_initialized() = %s, world_size = %s",
            dist.is_initialized(),
            dist.get_world_size() if dist.is_initialized() else "N/A",
        )

    # ──────────────────────────────────────────────
    # Build the SentenceTransformer model
    # ──────────────────────────────────────────────

    if args.parallel_mode == "model_parallel":
        word_embedding_model = models.Transformer(
            args.model_name,
            max_seq_length=args.max_seq_len,
            model_kwargs={
                "device_map": "auto",
                "dtype": torch.bfloat16,
                "max_memory": {
                    0: "36GiB",
                    1: "36GiB",
                    2: "36GiB",
                    3: "36GiB",
                    "cpu": "80GiB"
                },
                "low_cpu_mem_usage": True,
            },
        )
    else:
        word_embedding_model = models.Transformer(
            args.model_name,
            max_seq_length=args.max_seq_len,
            model_kwargs={
                "dtype": torch.bfloat16,
            },
        )

    pooling_model = models.Pooling(
        word_embedding_model.get_embedding_dimension(),
        pooling_mode="mean"
    )
    scoring_head = ScoringHead(input_dim=pooling_model.get_embedding_dimension(), hidden_dim=256, dropout=0.1)

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, scoring_head],
    )
    if rank == 0:
        logger.info(
            "Embedding dimension: %d",
            model.get_embedding_dimension(),
        )

    # ──────────────────────────────────────────────
    # Prepare datasets
    # FIX #1: deterministic, identical splits on every rank
    # ──────────────────────────────────────────────
    t = d_f.format_custom_dataset(CUSTOM_DATASET)

    # Pass a fixed seed so the shuffle is identical on every rank.
    # If shuffle_and_transform_formatted_dataset accepts a seed
    # parameter, pass it here. Otherwise ensure it is deterministic
    # or do the shuffle only on rank 0 and broadcast the indices.
    ds = d_f.shuffle_and_transform_formatted_dataset(
        t, seed=SPLIT_SEED
    )
    if DOWNSAMPLE_SIZE:
        ds=ds.select(range(DOWNSAMPLE_SIZE)) 
    # FIX #1 (continued): explicit seed ensures identical splits
    # ds is a HuggingFace Dataset, with the rows "id", "texts", and "labels"
    # id is an integer and purely informational
    # texts contains lists of texts of different quality levels
    # labels contains lists of the quality levels of the texts as integers
    # labels and texts are always in corresponding order
    ds = ds.train_test_split(0.3, seed=SPLIT_SEED)
    train_dataset = ds["train"].shuffle(seed=SPLIT_SEED)

    dev_test = ds["test"].train_test_split(0.5, seed=SPLIT_SEED)
    dev_dataset = dev_test["train"].shuffle(seed=SPLIT_SEED)
    test_dataset = dev_test["test"].shuffle(seed=SPLIT_SEED)

    if rank == 0:
        logger.info("Training dataset:\n%s", train_dataset)
        logger.info("Dev dataset:\n%s", dev_dataset)
        logger.info("Test dataset:\n%s", test_dataset)

    if dist.is_initialized():
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        local_len = torch.tensor([len(train_dataset), len(dev_dataset), len(test_dataset)], device=device)
        gathered = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_len)
        if rank == 0:
            for r, g in enumerate(gathered):
                assert torch.equal(g, local_len), (
                    f"Rank {r} has different split sizes {g.tolist()} "
                    f"vs rank 0 {local_len.tolist()} — data leak!"
                )
            logger.info(
                "✓ All ranks have identical split sizes: "
                "train=%d, dev=%d, test=%d",
                local_len[0].item(),
                local_len[1].item(),
                local_len[2].item(),
            )

    train_dataset_flat = expand_dataset_for_trainer(train_dataset)
    eval_dataset_flat = expand_dataset_for_trainer(dev_dataset)
    if rank == 0:
        logger.info("Flattened training dataset:\n%s", train_dataset_flat)
        logger.info("Flattened eval dataset:\n%s", eval_dataset_flat)

    # ──────────────────────────────────────────────
    # Instantiate loss
    # ──────────────────────────────────────────────
    train_loss = PairwiseRankingLoss(model=model, epsilon=0.2)

    # ──────────────────────────────────────────────
    # Build the win-rate evaluator on dev data
    # ──────────────────────────────────────────────
    dev_evaluator = PairwiseWinRateEvaluator(
        texts_list=dev_dataset["texts"],
        labels_list=dev_dataset["labels"],
        name="dev",
        batch_size=32,
    )

    # ──────────────────────────────────────────────
    # Create trainer & train
    # ──────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_flat,
        eval_dataset=eval_dataset_flat,
        loss=train_loss,
        evaluator=dev_evaluator,
    )

    # ── Diagnostic: what did the trainer do to our dataset? ──
    if rank == 0:
        logger.info(f"trainer.train_dataset type: {type(trainer.train_dataset)}")
        logger.info(f"trainer.train_dataset length: {getattr(trainer.train_dataset, '__len__', 'NO __len__')}")
        try:
            dl = trainer.get_train_dataloader()
            logger.info(f"Dataloader type: {type(dl)}")
            logger.info(f"Dataloader length: {len(dl)}")
        except TypeError as e:
            logger.error(f"Dataloader has no __len__: {e}")
        except Exception as e:
            logger.error(f"Dataloader error: {e}")

    trainer.train()

    if rank == 0:
        save_dir = os.path.join(OUTPUT_DIR, "final")
        model.save_pretrained(save_dir)
        logger.info("Model and scoring head saved to %s.", save_dir)

        # Final evaluation on test set
        test_evaluator = PairwiseWinRateEvaluator(
            texts_list=test_dataset["texts"],
            labels_list=test_dataset["labels"],
            name="test",
            batch_size=32,
        )
        test_metrics = test_evaluator(
            model, output_path=save_dir,
        )
        logger.info("=== Final test metrics ===")
        for k, v in test_metrics.items():
            logger.info("  %s = %s", k, v)

        # Sanity check
        # Baseline: how often does positional order match label order?
        baseline_correct = 0
        baseline_total = 0
        for test_ls in test_dataset['labels']:
            n = len(test_ls)
            for i in range(n):
                for j in range(i + 1, n):
                    if test_ls[i] == test_ls[j]:
                        continue
                    baseline_total += 1
                    # "Always predict earlier position has higher score"
                    # i.e., score(i) > score(j) for all i < j
                    if test_ls[i] < test_ls[j]:  # lower label = better
                        baseline_correct += 1
        if baseline_total > 0:
            logger.info(
                "Pairwise win-rate if always predicting positional order: %.4f  (%d / %d)",
                baseline_correct / baseline_total, baseline_correct, baseline_total,
            )
        
    # FIX #5: barrier so non-zero ranks don't tear down
    # the process group while rank 0 is still evaluating
    if dist.is_initialized():
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)