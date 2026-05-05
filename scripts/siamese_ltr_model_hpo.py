#!/usr/bin/env python
"""
hpo.py — Single-GPU Optuna hyperparameter optimisation for
         PairwiseRankingLoss + SentenceTransformerTrainer.

Usage
─────
    CUDA_VISIBLE_DEVICES=0 python hpo.py MODEL MAX_SEQ_LEN DATASET OUTPUT_DIR [DOWNSAMPLE] [N_TRIALS]

•  MUST be launched with plain `python`, NOT `torchrun` / `accelerate launch`.
•  Set CUDA_VISIBLE_DEVICES to pin the script to exactly one GPU.

After the study finishes the best hyper-parameters are written to
    <OUTPUT_DIR>/hpo_results.json
so they can be transferred to the multi-GPU training script.
"""

import gc
import json
import logging
import os
import shutil
import sys

import optuna
import torch
import torch.nn as nn
from transformers import TrainerCallback

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    models,
)
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
)

# ── Imports from the existing training code-base ──────────────
# Adjust the module name to match whatever you called the
# original multi-GPU training script (e.g. "train_multigpu").
from siamese_ltr_model_training import (
    PairwiseRankingLoss,
    PairwiseWinRateEvaluator,
    expand_dataset_for_trainer,
)
import dataset_functions as d_f

# ──────────────────────────────────────────────────────────────

os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SPLIT_SEED = 42


# ──────────────────────────────────────────────
# 1.  Optuna Pruning Callback for HF Trainer
# ──────────────────────────────────────────────
class OptunaPruningCallback(TrainerCallback):
    """
    Reports the dev win-rate to Optuna after every evaluation step
    so that unpromising trials can be pruned early.

    The evaluator returns metrics keyed as ``dev_win_rate``.
    Depending on the sentence-transformers version this may appear
    in the Trainer metrics dict as ``dev_win_rate`` or
    ``eval_dev_win_rate``; the callback checks for both.
    """

    def __init__(self, trial: optuna.Trial, metric_key: str = "dev_win_rate"):
        super().__init__()
        self.trial = trial
        self.metric_key = metric_key
        self._eval_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        value = None
        for candidate_key in [self.metric_key, f"eval_{self.metric_key}"]:
            if candidate_key in metrics:
                value = metrics[candidate_key]
                break

        if value is not None:
            self.trial.report(value, step=self._eval_count)
            self._eval_count += 1
            if self.trial.should_prune():
                raise optuna.TrialPruned()


# ──────────────────────────────────────────────
# 2.  Main
# ──────────────────────────────────────────────
def main(cmd_args):
    # ──────────────────────────────────────────
    # Safety: refuse to run under distributed
    # ──────────────────────────────────────────
    if (
        int(os.environ.get("WORLD_SIZE", 1)) > 1
        or int(os.environ.get("LOCAL_WORLD_SIZE", 1)) > 1
        or torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "This HPO script MUST run on a single GPU.\n"
            "Launch with:  CUDA_VISIBLE_DEVICES=0 python hpo.py ...\n"
            "Do NOT use torchrun / torch.distributed.launch / accelerate launch."
        )

    # ──────────────────────────────────────────
    # Parse CLI arguments
    # ──────────────────────────────────────────
    MODEL_NAME = cmd_args[0]
    MAX_SEQ_LEN = int(cmd_args[1])
    CUSTOM_DATASET = cmd_args[2]
    OUTPUT_DIR = cmd_args[3]
    DOWNSAMPLE_SIZE = int(cmd_args[4]) if len(cmd_args) > 4 else None
    N_TRIALS = int(cmd_args[5]) if len(cmd_args) > 5 else 20

    logger.info("HPO configuration:")
    logger.info("  model          = %s", MODEL_NAME)
    logger.info("  max_seq_len    = %d", MAX_SEQ_LEN)
    logger.info("  dataset        = %s", CUSTOM_DATASET)
    logger.info("  output_dir     = %s", OUTPUT_DIR)
    logger.info("  downsample     = %s", DOWNSAMPLE_SIZE)
    logger.info("  n_trials       = %d", N_TRIALS)

    # ──────────────────────────────────────────
    # Prepare datasets (identical split logic
    # to the multi-GPU training script)
    # ──────────────────────────────────────────
    t = d_f.format_custom_dataset(CUSTOM_DATASET)
    ds = d_f.shuffle_and_transform_formatted_dataset(t, seed=SPLIT_SEED)

    if DOWNSAMPLE_SIZE:
        ds = ds.select(range(DOWNSAMPLE_SIZE))

    ds = ds.train_test_split(0.3, seed=SPLIT_SEED)
    train_dataset = ds["train"].shuffle(seed=SPLIT_SEED)

    dev_test = ds["test"].train_test_split(0.5, seed=SPLIT_SEED)
    dev_dataset = dev_test["train"].shuffle(seed=SPLIT_SEED)
    test_dataset = dev_test["test"].shuffle(seed=SPLIT_SEED)

    logger.info(
        "Split sizes — train: %d | dev: %d | test: %d",
        len(train_dataset),
        len(dev_dataset),
        len(test_dataset),
    )

    train_dataset_flat = expand_dataset_for_trainer(train_dataset)
    eval_dataset_flat = expand_dataset_for_trainer(dev_dataset)

    dev_texts = dev_dataset["texts"]
    dev_labels = dev_dataset["labels"]

    # ──────────────────────────────────────────
    # Optuna objective
    # ──────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        """
        A single HPO trial:
          1. Sample hyper-parameters from the search space.
          2. Build a fresh model, loss, and evaluator.
          3. Train.
          4. Return the dev win-rate for Optuna to maximise.
        """

        # ── Suggest hyper-parameters ──────────
        # Training-argument HPs
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32],
        )
        epochs = trial.suggest_int("num_train_epochs", 1, 5)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

        # Loss HPs
        epsilon = trial.suggest_float("epsilon", 0.05, 2.0, log=True)
        use_scoring_hidden = trial.suggest_categorical(
            "use_scoring_hidden", [True, False],
        )
        scoring_hidden = (
            trial.suggest_int("scoring_hidden", 64, 512, log=True)
            if use_scoring_hidden
            else None
        )

        trial_output_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number}")

        # ── Fresh model ───────────────────────
        word_embedding_model = models.Transformer(
            MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            model_args={"dtype": torch.bfloat16},
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

        # ── Fresh loss ────────────────────────
        loss = PairwiseRankingLoss(
            model=model,
            epsilon=epsilon,
            scoring_hidden=scoring_hidden,
        )

        # ── Evaluator (references this trial's scoring head) ──
        evaluator = PairwiseWinRateEvaluator(
            texts_list=dev_texts,
            labels_list=dev_labels,
            scoring_head=loss.scoring_head,
            name="dev",
            batch_size=32,
        )

        # ── Training arguments ────────────────
        training_args = SentenceTransformerTrainingArguments(
            output_dir=trial_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=1,
            eval_strategy="epoch",
            seed=42,
            fp16=False,
            bf16=True,
            dataloader_drop_last=False,
        )

        # ── Trainer ───────────────────────────
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_flat,
            eval_dataset=eval_dataset_flat,
            loss=loss,
            evaluator=evaluator,
            callbacks=[OptunaPruningCallback(trial, metric_key="dev_win_rate")],
        )

        # ── Train & evaluate ──────────────────
        try:
            trainer.train()

            # Final evaluation with the evaluator
            metrics = evaluator(model)
            win_rate = metrics["dev_win_rate"]

            logger.info(
                "Trial %d finished — win_rate=%.4f | lr=%.2e  bs=%d  "
                "epochs=%d  warmup=%.2f  wd=%.4f  eps=%.3f  hidden=%s",
                trial.number,
                win_rate,
                lr,
                batch_size,
                epochs,
                warmup_ratio,
                weight_decay,
                epsilon,
                scoring_hidden,
            )

            return win_rate

        finally:
            # ── Cleanup: free GPU memory & disk ──
            del trainer, model, loss, evaluator
            gc.collect()
            torch.cuda.empty_cache()

            if os.path.exists(trial_output_dir):
                shutil.rmtree(trial_output_dir, ignore_errors=True)

    # ──────────────────────────────────────────
    # Create study & run optimisation
    # ──────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        study_name="pairwise_ranking_hpo",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # run at least 5 trials to completion before pruning
            n_warmup_steps=1,     # within a trial, allow at least 1 eval before pruning
        ),
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # ──────────────────────────────────────────
    # Report & persist results
    # ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HPO COMPLETE — %d trials", len(study.trials))
    logger.info("-" * 60)
    logger.info("Best trial        : %d", study.best_trial.number)
    logger.info("Best dev win-rate : %.4f", study.best_trial.value)
    logger.info("Best hyper-parameters:")
    for k, v in study.best_trial.params.items():
        logger.info("  %-35s = %s", k, v)
    logger.info("=" * 60)

    # Serialise all trials so nothing is lost
    results = {
        "best_trial_number": study.best_trial.number,
        "best_win_rate": study.best_trial.value,
        "best_hyperparameters": study.best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "hpo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main(sys.argv[1:])