# Claude 4.6 has been used to create part of this code

import torch
import torch.nn as nn
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

    def __init__(
        self,
        model: SentenceTransformer,
        epsilon: float = 1.0,
        scoring_hidden: int | None = None,
    ):
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

        device = next(model.parameters()).device
        self.scoring_head.to(device)

    # ---- required by SentenceTransformerTrainer ----
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

        n = len(sentence_features)  # number of samples per instance

        # ==================================================================
        # Each text passes through the shared BERT encoder exactly once;
        # the resulting embeddings stay in the computation graph so that
        # every pairwise loss term can back-propagate through the encoder.
        # ==================================================================
        scores = []  # will hold n tensors, each (batch_size,)
        for sf in sentence_features:
            emb = self.model(sf)["sentence_embedding"]  # (batch_size, emb_dim)
            score = self.scoring_head(emb).squeeze(-1)  # (batch_size,)
            scores.append(score)

        # Stack into a single tensor for convenient indexing.
        # Shape: (batch_size, n)
        scores = torch.stack(scores, dim=1)

        # ==================================================================
        # For every pair (i, j) with i < j we determine which sample
        # should score higher from the ordinal labels, then apply the
        # same margin-loss kernel that the original two-sample version
        # used.  Pairs with equal labels are masked out.
        # ==================================================================
        labels = labels.float()  # (batch_size, n)

        total_loss = torch.tensor(0.0, device=scores.device)
        valid_pairs = torch.tensor(0.0, device=scores.device)

        for i in range(n):
            for j in range(i + 1, n):
                label_i = labels[:, i]  # (batch_size,)
                label_j = labels[:, j]  # (batch_size,)

                sign = torch.sign(label_j - label_i)  # (batch_size,)
                diff = scores[:, j] - scores[:, i]  # (batch_size,)

                pair_loss = torch.relu(sign * diff + self.epsilon)

                mask = (label_i != label_j).float()  # (batch_size,)
                pair_loss = pair_loss * mask

                total_loss = total_loss + pair_loss.sum()
                valid_pairs = valid_pairs + mask.sum()

        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return total_loss


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
        """Return a dict whose keys start with ``self.name + '_'``."""

        self.scoring_head.eval()

        # Flatten every text for one big encode call, then un-flatten.
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

        correct = 0
        total = 0

        with torch.no_grad():
            for idx, labels in enumerate(self.labels_list):
                start, end = boundaries[idx], boundaries[idx + 1]
                embs = embeddings[start:end]  # (n, emb_dim)
                scores = self.scoring_head(embs).squeeze(-1)  # (n,)

                n = len(labels)
                for i in range(n):
                    for j in range(i + 1, n):
                        li, lj = labels[i], labels[j]
                        if li == lj:
                            continue
                        total += 1
                        # The text with the *lower* label should have a
                        # *higher* score.
                        if li < lj and scores[i] > scores[j]:
                            correct += 1
                        elif lj < li and scores[j] > scores[i]:
                            correct += 1
                        # Ties in score count as incorrect.

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
    n_per_row = len(ds[0]["texts"])
    # Sanity check: all rows must have the same width.
    for row_idx in range(len(ds)):
        if len(ds[row_idx]["texts"]) != n_per_row:
            raise ValueError(
                f"Row {row_idx} has {len(ds[row_idx]['texts'])} texts, "
                f"expected {n_per_row}."
            )
        if len(ds[row_idx]["labels"]) != n_per_row:
            raise ValueError(
                f"Row {row_idx} has {len(ds[row_idx]['labels'])} labels, "
                f"expected {n_per_row}."
            )

    # Build new column-oriented data.
    new_data: dict[str, list] = {
        f"sentence_{k}": [] for k in range(n_per_row)
    }
    new_data["label"] = []

    for row in ds:
        for k in range(n_per_row):
            new_data[f"sentence_{k}"].append(row["texts"][k])
        new_data["label"].append(row["labels"])

    return Dataset.from_dict(new_data)


# ──────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────
def main(cmd_args):

    # ──────────────────────────────────────────────
    # Build the SentenceTransformer model
    # ──────────────────────────────────────────────
    MODEL_NAME = cmd_args[0]
    MAX_SEQ_LEN = int(cmd_args[1])

    word_embedding_model = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
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

    logger.info(
        "Embedding dimension: %d", model.get_sentence_embedding_dimension()
    )

    # ──────────────────────────────────────────────
    # Prepare datasets
    # ──────────────────────────────────────────────
    ### NEW ADDED CODE ###
    t = d_f.format_custom_dataset("yle_2019")
    # ds is a HuggingFace Dataset, with the rows "id", "texts", and "labels"
    # id is an integer and purely informational
    # texts contains lists of texts of different quality levels
    # labels contains lists of the quality levels of the texts as integers
    # labels and texts are always in corresponding order
    ds = d_f.shuffle_and_transform_formatted_dataset(t).train_test_split(0.7)
    train_dataset = ds["train"]
    dev_test = ds["test"].train_test_split(0.5)
    dev_dataset = dev_test["train"]
    test_dataset = dev_test["test"]

    logger.info("Training dataset:\n%s", train_dataset)
    logger.info("Dev dataset:\n%s", dev_dataset)
    logger.info("Test dataset:\n%s", test_dataset)

    ### ADJUSTED CODE BELOW ###

    # Convert list-of-texts format → sentence_0 … sentence_{n-1} + label
    train_dataset_flat = expand_dataset_for_trainer(train_dataset)
    eval_dataset_flat = expand_dataset_for_trainer(dev_dataset)

    logger.info("Flattened training dataset:\n%s", train_dataset_flat)
    logger.info("Flattened eval dataset:\n%s", eval_dataset_flat)

    # ──────────────────────────────────────────────
    # Instantiate loss
    # ──────────────────────────────────────────────
    train_loss = PairwiseRankingLoss(
        model=model,
        epsilon=1.0,
        scoring_hidden=None,
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
    # Training arguments
    # ──────────────────────────────────────────────
    training_args = SentenceTransformerTrainingArguments(
        # --- output ---
        output_dir="output/finnish-pairwise-ranker",
        # --- epochs & batching ---
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # --- optimizer / scheduler ---
        learning_rate=2e-5,
        warmup_steps=10,
        # --- logging ---
        logging_strategy="steps",
        logging_steps=1,
        # --- saving ---
        save_strategy="epoch",
        save_total_limit=2,
        # --- reproducibility ---
        seed=42,
        # --- evaluation (win-rate on dev) ---
        eval_strategy="epoch",
        # --- performance ---
        fp16=False,
        bf16=False,
        dataloader_drop_last=False,
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

    trainer.train()

    # Save final model (encoder only)
    model.save_pretrained("output/finnish-pairwise-ranker/final")

    # Save scoring head separately
    torch.save(
        train_loss.scoring_head.state_dict(),
        "output/finnish-pairwise-ranker/final/scoring_head.pt",
    )
    logger.info("Model and scoring head saved.")

    # ──────────────────────────────────────────────
    # Final evaluation on test set
    # ──────────────────────────────────────────────
    test_evaluator = PairwiseWinRateEvaluator(
        texts_list=test_dataset["texts"],
        labels_list=test_dataset["labels"],
        scoring_head=train_loss.scoring_head,
        name="test",
        batch_size=32,
    )
    test_metrics = test_evaluator(
        model, output_path="output/finnish-pairwise-ranker/final"
    )
    logger.info("=== Final test metrics ===")
    for k, v in test_metrics.items():
        logger.info("  %s = %s", k, v)


    """

    # ──────────────────────────────────────────────
    # Inference helper
    # ──────────────────────────────────────────────
    def predict_scores(
        sentences: list[str],
        st_model: SentenceTransformer,
        scoring_head: nn.Module,
    ) -> torch.Tensor:
        ""Return scalar relevance scores for a list of sentences.""
        embeddings = st_model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        with torch.no_grad():
            scores = scoring_head(embeddings).squeeze(-1)
        return scores

    test_sentences = [
        "Helsinki on Suomen pääkaupunki.",
        "Epärelevantti dokumentti.",
        "Turku on vanha kaupunki Suomessa.",
    ]

    scores = predict_scores(test_sentences, model, train_loss.scoring_head)

    logger.info("=== Inference scores ===")
    for sent, score in zip(test_sentences, scores):
        logger.info("  %.4f  │  %s", score.item(), sent)

    # ──────────────────────────────────────────────
    # Loading saved model for later use
    # ──────────────────────────────────────────────
    def load_model_and_head(
        model_path: str,
        head_path: str,
        emb_dim: int = 768,
    ):
        ""Reload the encoder + scoring head from disk.""
        loaded_model = SentenceTransformer(model_path)

        loaded_head = nn.Linear(emb_dim, 1)
        loaded_head.load_state_dict(torch.load(head_path))
        loaded_head.eval()

        return loaded_model, loaded_head

    # Example reload:
    # reloaded_model, reloaded_head = load_model_and_head(
    #     "output/finnish-pairwise-ranker/final",
    #     "output/finnish-pairwise-ranker/final/scoring_head.pt",
    # )
    # scores = predict_scores(test_sentences, reloaded_model, reloaded_head)

    """


if __name__ == "__main__":
    main(sys.argv[1:])