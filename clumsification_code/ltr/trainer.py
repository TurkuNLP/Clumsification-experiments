import json
from typing import Optional

import torch
from transformers import Trainer

from .evaluation import evaluate_win_rate_distributed
from .utils import tensor_debug_summary
from .losses import canonicalize_loss_name


class PairwiseLTRTrainer(Trainer):
    def __init__(
        self,
        epsilon: float = 0.2,
        scale: float = 5.0,
        loss: str = "logistic",
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
        self.scale = scale
        self.loss = canonicalize_loss_name(loss)
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
            scale=self.scale,
            loss=self.loss,
            loss_normalization=self.loss_normalization,
        )

        loss = outputs["loss"]

        if not torch.isfinite(loss):
            debug = {}

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    debug[f"input.{k}"] = tensor_debug_summary(v)

            for k, v in outputs.items():
                if torch.is_tensor(v):
                    debug[f"output.{k}"] = tensor_debug_summary(v)

            bad_params = []
            for name, p in model.named_parameters():
                if p is not None and torch.is_tensor(p):
                    if not torch.isfinite(p).all():
                        bad_params.append(
                            {
                                "name": name,
                                **tensor_debug_summary(p),
                            }
                        )
                        if len(bad_params) >= 20:
                            break

            debug["bad_params"] = bad_params

            raise FloatingPointError(
                f"Non-finite training loss detected at step {self.state.global_step}. "
                f"Debug summary: {json.dumps(debug, indent=2)}"
            )

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