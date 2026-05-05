# Claude 4.6 has been used to create part of this code

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import destroy_process_group
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    models,
)
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator
import logging
from typing import Iterable, Dict
import sys
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
import dataset_functions as d_f

os.environ["WANDB_MODE"] = "disabled"

# ──────────────────────────────────────────────
# 1.  Custom Pairwise Ranking Loss
# ──────────────────────────────────────────────
class PairwiseRankingLoss(nn.Module):
    """
    Listwise pairwise margin-based ranking loss.

    Given n samples per training instance, each annotated with an ordinal
    quality label (0 = best, 1 = second-best, …), the loss enumerates all
    n(n−1)/2 ordered pairs and sums the original margin loss over them:

        L_ij = max(0, sign_ij · (f(xj) − f(xi)) + ε)

    where sign_ij = +1 when xi should rank higher than xj (label_i < label_j),
          sign_ij = -1 when xj should rank higher than xi (label_j < label_i).

    Pairs that share the same label are masked out (no preference to enforce).

    ──────────────────────────────────────────────────────
    EFFICIENT SIAMESE DESIGN
    ──────────────────────────────────────────────────────
    •  Each of the n texts is encoded through the shared BERT encoder
       exactly ONCE.
    •  A single shared scoring head maps every embedding to a scalar.
    •  All pairwise losses are computed from the n cached scalars,
       so gradients from every pair flow back through the encoder
       in one backward pass — no redundant forward passes.
    ──────────────────────────────────────────────────────

    Original two-sample formulation (preserved as the inner kernel):

        L = max(0, (1 − 2·l_12) · (f(x2) − f(x1)) + ε)

        l_12 = 0  ⟹  x1 ranks higher  ⟹  max(0, f(x2) − f(x1) + ε)
        l_12 = 1  ⟹  x2 ranks higher  ⟹  max(0, f(x1) − f(x2) + ε)
    """
    def __init__(self, model, epsilon=1.0, scoring_hidden=None):
        super().__init__()
        self.model = model
        self.epsilon = epsilon

        emb_dim = model.get_sentence_embedding_dimension()

        if scoring_hidden is not None:
            self.scoring_head = nn.Sequential(
                nn.Linear(emb_dim, scoring_hidden),
                nn.ReLU(),
                nn.Linear(scoring_hidden, 1),
            )
        else:
            self.scoring_head = nn.Linear(emb_dim, 1)
        self._head_moved = False

    @property
    def citation(self) -> str:
        return ""

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence_features : list[dict], length n
            Each element is a tokenised text column produced by the data
            collator (keys: input_ids, attention_mask, …).
            All n texts are encoded through the shared BERT model exactly
            once.

        labels : Tensor, shape (batch_size, n)
            Now each row contains n ordinal quality ranks:
                0 = highest quality, 1 = second-highest, …
            Pairs whose labels are equal contribute zero loss.

        Returns
        -------
        loss : scalar Tensor
            Mean over all valid (unequal-label) pairs across the batch.
        """
        # Ensure scoring_head is on the same device as the model
        if not self._head_moved:
            device = next(self.model.parameters()).device
            self.scoring_head.to(device)
            self._head_moved = True

        n = len(sentence_features)

        # Defensive check: labels must be (batch_size, n)
        if labels.dim() == 1:
            raise ValueError(
                f"Expected labels of shape (batch_size, {n}), "
                f"but got 1-D tensor of shape {labels.shape}. "
                f"The data collator may be squeezing list labels."
            )
        assert labels.shape[1] == n, (
            f"Label width {labels.shape[1]} != number of sentence columns {n}"
        )

        scores = []
        for sf in sentence_features:
            emb = self.model(sf)["sentence_embedding"]
            score = self.scoring_head(emb).squeeze(-1)
            scores.append(score)
        # ==================================================================
        # Each text passes through the shared BERT encoder exactly once;
        # the resulting embeddings stay in the computation graph so that
        # every pairwise loss term can back-propagate through the encoder.
        # ==================================================================

        # Stack into a single tensor for convenient indexing.
        # Shape: (batch_size, n)
        scores = torch.stack(scores, dim=1)  # (batch_size, n)

        labels = labels.float()

        # Build all (i, j) pair indices with i < j
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)  # each shape: (n_pairs,)

        # Gather scores and labels for all pairs at once
        # scores_i, scores_j: (batch_size, n_pairs)
        scores_i = scores[:, idx_i]
        scores_j = scores[:, idx_j]
        labels_i = labels[:, idx_i]
        labels_j = labels[:, idx_j]

        sign = torch.sign(labels_j - labels_i)      # (batch_size, n_pairs)
        diff = scores_j - scores_i                    # (batch_size, n_pairs)
        pair_loss = torch.relu(sign * diff + self.epsilon)

        mask = (labels_i != labels_j).float()         # (batch_size, n_pairs)
        pair_loss = pair_loss * mask

        valid_pairs = mask.sum()
        if valid_pairs > 0:
            return pair_loss.sum() / valid_pairs
        else:
            # Return 0 that's connected to the graph (scores sum * 0)
            return (scores * 0).sum()



# ──────────────────────────────────────────────
# 2.  Win-Rate Evaluator
# ──────────────────────────────────────────────
class PairwiseWinRateEvaluator(SentenceEvaluator):
    """
    Evaluates the model on a held-out set using **pairwise win rate**.

    For every evaluation example the evaluator:
      1. Encodes all n texts through the shared encoder.
      2. Scores each embedding with the scoring head.
      3. For every ordered pair (i, j) where label_i ≠ label_j,
         checks whether the model assigns a higher score to the
         text with the *lower* (= better) ordinal label.

    The overall win-rate is:

        win_rate = correct_pairs / total_pairs

    A random baseline would score ~50 %; a perfect ranker scores 100 %.
    """

    def __init__(
        self,
        texts_list: list[list[str]],
        labels_list: list[list[int]],
        scoring_head: nn.Module,
        name: str = "pairwise_win_rate",
        batch_size: int = 32,
    ):
        """
        Parameters
        ----------
        texts_list  : outer list = evaluation examples;
                      inner list = the n texts per example.
        labels_list : matching ordinal labels (0 = best).
        scoring_head: the nn.Module that maps embeddings → scalars.
        """
        super().__init__()
        self.texts_list = texts_list
        self.labels_list = labels_list
        self.scoring_head = scoring_head
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

        self.scoring_head.eval()

        flat_texts: list[str] = []
        boundaries: list[int] = [0]
        for texts in self.texts_list:
            flat_texts.extend(texts)
            boundaries.append(boundaries[-1] + len(texts))

        embeddings = model.encode(
            flat_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Ensure scoring head is on the same device as embeddings
        self.scoring_head.to(embeddings.device)

        correct = 0
        total = 0

        with torch.no_grad():
            with torch.no_grad():
                for idx, labels in enumerate(self.labels_list):
                    start, end = boundaries[idx], boundaries[idx + 1]
                    embs = embeddings[start:end]
                    scores = self.scoring_head(embs).squeeze(-1)

                    n = len(labels)
                    labels_t = torch.tensor(labels, device=scores.device).float()

                    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
                    li = labels_t[idx_i]
                    lj = labels_t[idx_j]
                    si = scores[idx_i]
                    sj = scores[idx_j]

                    valid = li != lj
                    total += valid.sum().item()

                    # "correct" when lower label has higher score
                    correct_mask = ((li < lj) & (si > sj)) | ((lj < li) & (sj > si))
                    correct += (correct_mask & valid).sum().item()

        win_rate = correct / total if total > 0 else 0.0

        metrics = {
            f"{self.name}_win_rate": win_rate,
            f"{self.name}_correct_pairs": correct,
            f"{self.name}_total_pairs": total,
        }

        logger.info(
            "[%s] epoch=%d  steps=%d  win_rate=%.4f  (%d / %d pairs)",
            self.name, epoch, steps, win_rate, correct, total,
        )

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            with open(
                os.path.join(output_path, f"{self.name}_results.txt"), "a"
            ) as f:
                f.write(
                    f"epoch={epoch}  steps={steps}  "
                    f"win_rate={win_rate:.6f}  "
                    f"correct={correct}  total={total}\n"
                )

        self.scoring_head.train()
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
def main(cmd_args):

    # ──────────────────────────────────────────────
    # FIX #7: use a single local `rank` variable
    # ──────────────────────────────────────────────
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    MODEL_NAME = cmd_args[0]
    MAX_SEQ_LEN = int(cmd_args[1])
    CUSTOM_DATASET = cmd_args[2]
    OUTPUT_DIR = cmd_args[3]
    #If note downsampling, then use a larger batch size
    BS = 16
    #Optional downsampling for testing purposes
    DOWNSAMPLE_SIZE = None
    if len(cmd_args) > 4:
        DOWNSAMPLE_SIZE = int(cmd_args[4])
        if DOWNSAMPLE_SIZE == 50:
            BS = 2

    # ──────────────────────────────────────────────
    # FIX #1: Fixed seed for ALL random data operations
    #         so that every rank produces identical splits
    # ──────────────────────────────────────────────
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
    )

    # Make sure this dist works...
    assert dist.is_initialized()

    if rank == 0:
        logger.info(
            "dist.is_initialized() = %s, world_size = %s",
            dist.is_initialized(),
            dist.get_world_size() if dist.is_initialized() else "N/A",
        )

    # ──────────────────────────────────────────────
    # Build the SentenceTransformer model
    # ──────────────────────────────────────────────

    word_embedding_model = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        model_args={
            "dtype": torch.bfloat16,
        },
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
    )
    if rank == 0:
        logger.info(
            "Embedding dimension: %d",
            model.get_sentence_embedding_dimension(),
        )

    # ──────────────────────────────────────────────
    # Prepare datasets
    # FIX #1: deterministic, identical splits on every rank
    # ──────────────────────────────────────────────
    t = d_f.format_custom_dataset(CUSTOM_DATASET)
    if rank == 0:
        print(t[0])

    # Pass a fixed seed so the shuffle is identical on every rank.
    # If shuffle_and_transform_formatted_dataset accepts a seed
    # parameter, pass it here. Otherwise ensure it is deterministic
    # or do the shuffle only on rank 0 and broadcast the indices.
    ds = d_f.shuffle_and_transform_formatted_dataset(
        t, seed=SPLIT_SEED          # ← you may need to add this
    )
    if DOWNSAMPLE_SIZE:
        ds=ds.select(range(DOWNSAMPLE_SIZE))                                #   parameter in dataset_functions
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

    # ──────────────────────────────────────────────
    # FIX #1 verification: sanity-check that all ranks
    # agree on the exact same split (cheap fingerprint)
    # ──────────────────────────────────────────────
    if dist.is_initialized():
        local_len = torch.tensor(
            [len(train_dataset), len(dev_dataset), len(test_dataset)],
            device=local_rank,
        )
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
    train_loss = PairwiseRankingLoss(
        model=model,
        epsilon=0.2,
        scoring_hidden=None,
    )
    # ──────────────────────────────────────────────
    # Ensure scoring_head is on the correct GPU
    # before any distributed operations
    # ──────────────────────────────────────────────
    if dist.is_initialized() and dist.get_world_size() > 1:
        device = torch.device(f"cuda:{local_rank}")
        train_loss.scoring_head.to(device)

        # Now broadcast rank 0's weights to all other ranks
        for param in train_loss.scoring_head.parameters():
            dist.broadcast(param.data, src=0)
        if rank == 0:
            logger.info("Broadcast scoring_head weights from rank 0 to all ranks.")

        # Register gradient-sync hooks
        def _make_sync_hook():
            world_size = dist.get_world_size()
            def hook(grad: torch.Tensor) -> torch.Tensor:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                return grad / world_size
            return hook

        for param in train_loss.scoring_head.parameters():
            if param.requires_grad:
                param.register_hook(_make_sync_hook())

        if rank == 0:
            logger.info(
                "Registered gradient-sync hooks on %d scoring_head parameters.",
                sum(1 for p in train_loss.scoring_head.parameters()
                    if p.requires_grad),
            )

    # ──────────────────────────────────────────────
    # Build the win-rate evaluator on dev data
    # ──────────────────────────────────────────────
    dev_evaluator = PairwiseWinRateEvaluator(
        texts_list=dev_dataset["texts"],
        labels_list=dev_dataset["labels"],
        scoring_head=train_loss.scoring_head,
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

    # ──────────────────────────────────────────────
    # FIX #4 & #5: Only rank 0 saves and evaluates;
    # other ranks wait at a barrier, then all exit
    # together.
    # ──────────────────────────────────────────────
    if rank == 0:
        save_dir = os.path.join(OUTPUT_DIR, "final")
        model.save_pretrained(save_dir)
        torch.save(
            train_loss.scoring_head.state_dict(),
            os.path.join(save_dir, "scoring_head.pt"),
        )
        logger.info("Model and scoring head saved to %s.", save_dir)

        # Final evaluation on test set
        test_evaluator = PairwiseWinRateEvaluator(
            texts_list=test_dataset["texts"],
            labels_list=test_dataset["labels"],
            scoring_head=train_loss.scoring_head,
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
        # Also log random baseline
        logger.info("Random baseline: 0.5000")
        
    # FIX #5: barrier so non-zero ranks don't tear down
    # the process group while rank 0 is still evaluating
    if dist.is_initialized():
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])