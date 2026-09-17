# This script has been co-created, refactored, and cleaned using GPT 5.6.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import inspect
import os
import time

# Avoid the tokenizer library's multi-process advisory on every DDP worker.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

from clumsification_code.data.hf_dataset import load_formatted_dataset_dict
from clumsification_code.data.io import default_formatted_dataset_path
from clumsification_code.data.flattening import flatten_dataset_dict
from clumsification_code.fe.args import parse_train_args
from clumsification_code.fe.checkpointing import (
    save_final_model,
    write_trainer_checkpoint_metadata,
)
from clumsification_code.fe.early_stopping import CheckpointEarlyStoppingCallback
from clumsification_code.fe.collators import (
    BinaryCollator,
    PairwiseCollator,
    RegressionCollator,
)
from clumsification_code.fe.modeling import FEModel
from clumsification_code.fe.metrics import binary_metrics, pairwise_metrics, regression_metrics
from clumsification_code.fe.regression_data import build_regression_dataset_dict
from clumsification_code.fe.scheduling import ExampleSchedule, resolve_example_schedule
from clumsification_code.fe.utils import (
    configure_logging,
    get_preferred_param_dtype,
    logger,
)


os.environ["WANDB_MODE"] = "disabled"


class FETrainerCheckpointCallback(TrainerCallback):
    """Make standard Trainer checkpoints loadable by FE evaluation clients."""

    def __init__(self, *, model_name: str, pooling: str, max_seq_len: int, objective: str):
        self.model_name = model_name
        self.pooling = pooling
        self.max_seq_len = max_seq_len
        self.objective = objective

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if state.is_world_process_zero:
            write_trainer_checkpoint_metadata(
                checkpoint_dir=os.path.join(args.output_dir, f"checkpoint-{state.global_step}"),
                model_name=self.model_name,
                pooling=self.pooling,
                max_seq_len=self.max_seq_len,
                objective=self.objective,
            )
        # Trainer has completed the atomic checkpoint save on every rank before
        # it invokes on_save.  Keep ranks aligned until the FE metadata exists,
        # so every retained checkpoint is immediately evaluation-loadable.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return control


def _compat_kwargs(callable_obj, kwargs: dict) -> dict:
    """Keep the entrypoint usable across supported Transformers releases."""
    parameters = inspect.signature(callable_obj).parameters
    accepted = {name for name, parameter in parameters.items()
                if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
    filtered = {name: value for name, value in kwargs.items() if name in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        logger.warning("Ignoring unsupported Transformers arguments: %s", dropped)
    return filtered


def build_training_arguments(
    args,
    use_cuda: bool,
    use_bf16: bool,
    use_fp16: bool,
    world_size: int,
    rank: int = 0,
    example_schedule: ExampleSchedule | None = None,
):
    save_strategy = "steps" if example_schedule is not None else args.save_strategy
    eval_strategy = "steps" if example_schedule is not None else args.eval_strategy
    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        save_total_limit=args.save_total_limit,
        save_only_model=getattr(args, "save_only_model", False),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        # Trainer's progress callback is process-local; make global rank zero
        # the only writer even when a job spans multiple nodes.
        disable_tqdm=rank != 0,
    )
    if example_schedule is not None:
        training_kwargs.update(
            {
                "save_steps": example_schedule.save_steps,
                "eval_steps": example_schedule.eval_steps,
            }
        )

    # Loading the best model requires evaluation and checkpoint saving to use
    # compatible strategies. Pairwise runs may deliberately evaluate during
    # training without saving intermediate checkpoints: save_final_model()
    # persists the final model after trainer.train() in that workflow.
    load_best_model_at_end = save_strategy != "no"
    training_kwargs["load_best_model_at_end"] = load_best_model_at_end

    if load_best_model_at_end:
        training_kwargs.update(
            {
                "metric_for_best_model": (
                    "spearman"
                    if args.training_method == "regression"
                    else ("binary_accuracy" if args.training_method == "binary" else "pairwise_accuracy")
                ),
                "greater_is_better": True,
            }
        )
    elif eval_strategy != "no" and save_strategy == "no":
        logger.info(
            "Intermediate checkpoint saving is disabled; validation will run "
            "with eval_strategy=%s, and the final model will be saved without "
            "best-checkpoint reloading.",
            eval_strategy,
        )

    if args.parallelism == "fsdp":
        if not use_cuda or world_size <= 1:
            raise ValueError("FSDP requires CUDA and WORLD_SIZE greater than one.")
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

        training_kwargs["fsdp"] = f"{args.fsdp_sharding_strategy} auto_wrap"
        fsdp_config["transformer_layer_cls_to_wrap"] = args.fsdp_layer_cls
        training_kwargs["fsdp_config"] = fsdp_config

    # `eval_strategy` replaced the older `evaluation_strategy` name.  The
    # cluster's Transformers build is old enough that it may also lack newer
    # fields such as `warmup_ratio`; filter against the actual constructor.
    training_parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" not in training_parameters and "evaluation_strategy" in training_parameters:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
    return TrainingArguments(**_compat_kwargs(TrainingArguments, training_kwargs))


def log_dtype_counts(model) -> None:
    dtype_counts = {}

    for name, parameter in model.named_parameters():
        dtype_counts[str(parameter.dtype)] = (
            dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
        )

        if parameter.dtype == torch.float32:
            logger.info(f"FP32 parameter: {name}, shape={tuple(parameter.shape)}")

    logger.info(f"Parameter dtype counts: {dtype_counts}")


def resolve_formatted_dataset_path(args) -> str:
    if args.formatted_dataset_path is not None:
        return args.formatted_dataset_path

    return default_formatted_dataset_path(args.formatted_dataset_name)


def main():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    configure_logging(rank=rank)

    args = parse_train_args()
    set_seed(args.seed)

    eval_only = getattr(args, "eval_only", False)
    if args.train_sample_budget is not None:
        batch = (world_size * args.per_device_train_batch_size
                 * args.gradient_accumulation_steps)
        if args.train_sample_budget <= 0 or batch <= 0 or args.train_sample_budget % batch:
            raise ValueError("--train-sample-budget must be positive and divisible by global batch size")
        args.max_steps = args.train_sample_budget // batch
        args.num_train_epochs = 1
        # A fixed-budget trial always evaluates the endpoint, without early stopping.
        args.save_every_examples = args.eval_every_examples = None
        args.early_stopping_checkpoints = None
        args.save_strategy = args.eval_strategy = "no"
    example_schedule = resolve_example_schedule(
        save_every_examples=args.save_every_examples,
        eval_every_examples=args.eval_every_examples,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=world_size,
    )
    if args.early_stopping_checkpoints is not None and example_schedule is None:
        raise ValueError(
            "--early-stopping-checkpoints requires --save-every-examples and "
            "--eval-every-examples"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if args.parallelism == "fsdp":
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    else:
        os.environ.pop("ACCELERATE_USE_FSDP", None)
        os.environ.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)

    dataset_path = resolve_formatted_dataset_path(args)

    if rank == 0:
        logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        logger.info(f"Loading formatted dataset from: {dataset_path}")
        logger.info(f"training_method={args.training_method}")
        logger.info(f"pair_policy={args.pair_policy}")
        if example_schedule is not None:
            logger.info(
                "Example schedule: global_batch=%d, eval_steps=%d "
                "(~%d examples), save_steps=%d (~%d examples)",
                example_schedule.global_batch_size,
                example_schedule.eval_steps,
                example_schedule.effective_eval_examples,
                example_schedule.save_steps,
                example_schedule.effective_save_examples,
            )

        if eval_only:
            logger.info(
                "Running in --eval_only mode: training will be skipped. "
                f"Evaluating the model supplied via --model_name ({args.model_name})."
            )

    dataset_dict = load_formatted_dataset_dict(dataset_path)
    regression_metadata = None

    if args.training_method == "regression":
        dataset_dict, regression_metadata = build_regression_dataset_dict(
            grouped_dataset_dict=dataset_dict,
            score_name=args.score_name,
            exclude_layer_zero=args.exclude_layer_zero,
            exclude_layer_zero_training=args.exclude_layer_zero_training,
        )
    elif args.training_method == "pairwise":
        # Factorial pairwise datasets are already flat. Historical FE inputs
        # remain grouped chains and still need source-safe flattening.
        if "chosen_text" not in dataset_dict["train"].column_names:
            dataset_dict = flatten_dataset_dict(
                dataset_dict,
                training_method="pairwise",
                pair_policy=args.pair_policy,
            )
    else:
        required = {"text", "label"}
        missing = required - set(dataset_dict["train"].column_names)
        if missing:
            raise ValueError(f"Binary datasets require columns {sorted(required)}; missing {sorted(missing)}")

    train_dataset = dataset_dict["train"]
    if args.train_sample_budget is not None and not eval_only:
        if len(train_dataset) < args.train_sample_budget:
            raise ValueError(f"Need {args.train_sample_budget} training rows; got {len(train_dataset)}")
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.train_sample_budget))
        logger.info("Training budget: %d rows, global batch %d, %d optimizer steps",
                    args.train_sample_budget, batch, args.max_steps)
    dev_dataset = dataset_dict["dev"]
    test_dataset = dataset_dict["test"]

    if rank == 0:
        logger.info(train_dataset)
        logger.info(
            f"train={len(train_dataset)} "
            f"dev={len(dev_dataset)} "
            f"test={len(test_dataset)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_dtype = get_preferred_param_dtype()

    if rank == 0:
        logger.info(f"Using parameter dtype: {param_dtype}")

    if eval_only:
        # Evaluation-only runs must use the trained encoder and scalar head,
        # rather than silently constructing a fresh random head.
        model = load_fe_model(
            final_dir=args.model_name,
            attn_implementation=args.attn_implementation,
            param_dtype=param_dtype,
            map_location="cpu",
        )
        model.to(dtype=param_dtype)
    else:
        model = FEModel(
            model_name=args.model_name,
            attn_implementation=args.attn_implementation,
            param_dtype=param_dtype,
            pooling=args.pooling,
        )
    model.training_objective = args.training_method
    model.ranking_epsilon = args.epsilon
    model.ranking_scale = args.scale
    model.ranking_loss = args.loss
    model.regression_loss_name = args.loss if args.training_method == "regression" else "huber"
    model.regression_huber_delta = args.huber_delta

    if model.encoder.config.pad_token_id is None:
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    if args.training_method == "regression":
        data_collator = RegressionCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            text_prefix=args.text_prefix,
        )
    elif args.training_method == "pairwise":
        data_collator = PairwiseCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            text_prefix=args.text_prefix,
        )
    else:
        data_collator = BinaryCollator(
            tokenizer=tokenizer, max_length=args.max_seq_len,
            text_prefix=args.text_prefix,
        )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and param_dtype == torch.bfloat16
    use_fp16 = use_cuda and param_dtype == torch.float16

    if rank == 0:
        logger.info(f"use_cuda={use_cuda}")
        logger.info(f"use_bf16={use_bf16}")

        if args.training_method == "pairwise":
            logger.info(f"loss={args.loss}")
        elif args.training_method == "regression":
            logger.info(f"score_name={args.score_name}")
            logger.info(f"text_prefix={args.text_prefix!r}")
            logger.info(
                f"target_scaling={regression_metadata['target_scaling']}"
            )
        else:
            logger.info("loss=binary")
            logger.info(f"text_prefix={args.text_prefix!r}")

        logger.info(
            "parallelism=%s fsdp_sharding_strategy=%s fsdp_layer_cls=%s",
            args.parallelism,
            args.fsdp_sharding_strategy if args.parallelism == "fsdp" else None,
            args.fsdp_layer_cls if args.parallelism == "fsdp" else None,
        )
        logger.info("pooling=%s", model.pooling)

    training_args = build_training_arguments(
        args=args,
        use_cuda=use_cuda,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        world_size=world_size,
        rank=rank,
        example_schedule=example_schedule,
    )

    trainer_common = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "processing_class": tokenizer,
    }
    if args.training_method == "regression":
        trainer_common["compute_metrics"] = regression_metrics
    elif args.training_method == "pairwise":
        trainer_common["compute_metrics"] = pairwise_metrics
    else:
        trainer_common["compute_metrics"] = binary_metrics
    callbacks = [
        FETrainerCheckpointCallback(
            model_name=args.model_name,
            pooling=model.pooling,
            max_seq_len=args.max_seq_len,
            objective=args.training_method,
        )
    ]
    if args.early_stopping_checkpoints is not None:
        metric_name = {
            "regression": "eval_spearman",
            "pairwise": "eval_pairwise_accuracy",
            "binary": "eval_binary_accuracy",
        }[args.training_method]
        callbacks.append(
            CheckpointEarlyStoppingCallback(
                metric_name=metric_name,
                patience=args.early_stopping_checkpoints,
                threshold=args.early_stopping_threshold,
                min_examples=args.early_stopping_min_examples,
                global_batch_size=example_schedule.global_batch_size,
            )
        )
    trainer_common["callbacks"] = callbacks
    if "processing_class" not in inspect.signature(Trainer).parameters:
        trainer_common["tokenizer"] = trainer_common.pop("processing_class")
    trainer = Trainer(**_compat_kwargs(Trainer, trainer_common))

    if rank == 0:
        log_dtype_counts(model)

    final_dir = args.output_dir
    hpo_dev_metrics = None

    if not eval_only:
        trainer.train()

        final_dir = save_final_model(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            rank=rank,
            parallelism=args.parallelism,
            metadata={
                "objective": args.training_method,
                "pair_policy": args.pair_policy if args.training_method == "pairwise" else None,
                "loss": args.loss,
                "epsilon": args.epsilon if args.training_method == "pairwise" else None,
                "scale": args.scale if args.training_method == "pairwise" else None,
                "huber_delta": args.huber_delta if args.training_method == "regression" else None,
                "target_transformation": regression_metadata["target_scaling"] if regression_metadata else None,
                "tokenizer": {
                    "class": tokenizer.__class__.__name__,
                    "name_or_path": getattr(tokenizer, "name_or_path", None),
                    "max_length": args.max_seq_len,
                },
                "distributed": {
                    "parallelism": args.parallelism,
                    "fsdp_sharding_strategy": (
                        args.fsdp_sharding_strategy if args.parallelism == "fsdp" else None
                    ),
                    "world_size": world_size,
                    "parameter_dtype": str(param_dtype),
                },
                "pooling": model.pooling,
            },
        )

        if args.training_method == "regression":
            trainer.accelerator.wait_for_everyone()

            if rank == 0:
                metadata_path = os.path.join(
                    final_dir,
                    "regression_metadata.json",
                )

                with open(metadata_path, "w", encoding="utf-8") as output_file:
                    json.dump(regression_metadata, output_file, indent=2)

                logger.info(f"Saved regression metadata to {metadata_path}")

        if getattr(args, "hpo_mode", False):
            hpo_dev_metrics = trainer.evaluate(
                eval_dataset=dev_dataset,
                metric_key_prefix=getattr(args, "hpo_metric_prefix", "hpo_dev"),
            )

            trainer.accelerator.wait_for_everyone()

            if rank == 0:
                hpo_metrics_path = os.path.join(
                    args.output_dir,
                    "hpo_dev_metrics.json",
                )

                with open(hpo_metrics_path, "w", encoding="utf-8") as output_file:
                    json.dump(hpo_dev_metrics, output_file, indent=2)

                logger.info(f"HPO dev metrics: {hpo_dev_metrics}")
                logger.info(f"Saved HPO dev metrics to {hpo_metrics_path}")

    else:
        if rank == 0:
            logger.info("Skipping training and model saving (--eval_only mode).")
            os.makedirs(final_dir, exist_ok=True)

    trainer.accelerator.wait_for_everyone()

    if getattr(args, "skip_final_test_eval", False):
        if rank == 0:
            logger.info(
                "Skipping final test evaluation because "
                "--skip_final_test_eval was set."
            )

        trainer.accelerator.wait_for_everyone()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        return

    evaluation_started_at = time.monotonic()
    if rank == 0:
        logger.info(
            "Final evaluation: starting on %d test examples",
            len(test_dataset),
        )

    metrics_test = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    baselines = None

    if rank == 0:
        logger.info(
            "Final evaluation: inference completed in %.1f seconds; "
            "synchronizing ranks",
            time.monotonic() - evaluation_started_at,
        )

    trainer.accelerator.wait_for_everyone()

    if rank == 0:
        metrics_path = os.path.join(final_dir, "metrics.json")

        with open(metrics_path, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "test": metrics_test,
                    "baselines": baselines,
                    "hpo_dev": hpo_dev_metrics,
                    "regression": regression_metadata,
                },
                output_file,
                indent=2,
            )

        logger.info(f"Test metrics: {metrics_test}")

        if baselines is not None:
            logger.info(f"Baselines: {baselines}")

        logger.info(f"Saved metrics to {metrics_path}")

    trainer.accelerator.wait_for_everyone()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info(
            "Final evaluation completed in %.1f seconds",
            time.monotonic() - evaluation_started_at,
        )


if __name__ == "__main__":
    main()
