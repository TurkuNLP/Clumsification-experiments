# This script has been co-created, refactored, and cleaned using GPT 5.6.
import json
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainingArguments, set_seed

from clumsification_code.data.hf_dataset import load_formatted_dataset_dict
from clumsification_code.data.io import default_formatted_dataset_path
from clumsification_code.fe.args import parse_train_args
from clumsification_code.fe.checkpointing import save_final_model
from clumsification_code.fe.collators import (
    GroupedRankingCollator,
    RegressionCollator,
)
from clumsification_code.fe.evaluation import (
    baseline_pairwise_accuracies,
    evaluate_pairwise_accuracy_distributed,
)
from clumsification_code.fe.modeling import FEModel
from clumsification_code.fe.regression import (
    RegressionFETrainer,
    build_regression_dataset_dict,
    evaluate_regression_distributed,
)
from clumsification_code.fe.trainer import PairwiseFETrainer
from clumsification_code.fe.utils import (
    configure_logging,
    get_preferred_param_dtype,
    logger,
)


os.environ["WANDB_MODE"] = "disabled"
os.environ["ACCELERATE_USE_FSDP"] = "true"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"


def build_training_arguments(
    args,
    use_cuda: bool,
    use_bf16: bool,
    use_fp16: bool,
    world_size: int,
):
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
        fp16=use_fp16,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # Regression selects the best checkpoint using validation Spearman
    # correlation. args.py validates matching save/evaluation strategies.
    if args.training_method == "regression":
        training_kwargs.update(
            {
                "load_best_model_at_end": True,
                "metric_for_best_model": "spearman",
                "greater_is_better": True,
            }
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
                "No fsdp_layer_cls provided. Using fsdp='full_shard' "
                "without auto_wrap."
            )

        training_kwargs["fsdp_config"] = fsdp_config
    else:
        logger.warning("FSDP disabled because CUDA is not available.")

    return TrainingArguments(**training_kwargs)


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
    configure_logging()

    args = parse_train_args()
    set_seed(args.seed)

    eval_only = getattr(args, "eval_only", False)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    dataset_path = resolve_formatted_dataset_path(args)

    if rank == 0:
        logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        logger.info(f"Loading formatted dataset from: {dataset_path}")
        logger.info(f"training_method={args.training_method}")

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
        )

    train_dataset = dataset_dict["train"]
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

    model = FEModel(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation,
        param_dtype=param_dtype,
    )

    if model.encoder.config.pad_token_id is None:
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    if args.training_method == "regression":
        data_collator = RegressionCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            text_prefix=args.text_prefix,
        )
    else:
        data_collator = GroupedRankingCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and param_dtype == torch.bfloat16
    use_fp16 = use_cuda and param_dtype == torch.float16

    if rank == 0:
        logger.info(f"use_cuda={use_cuda}")
        logger.info(f"use_bf16={use_bf16}")

        if args.training_method == "pairwise":
            logger.info(f"loss={args.loss}")
            logger.info(f"loss_normalization={args.loss_normalization}")
        else:
            logger.info(f"score_name={args.score_name}")
            logger.info(f"text_prefix={args.text_prefix!r}")
            logger.info(
                f"target_scaling={regression_metadata['target_scaling']}"
            )

        logger.info(f"FSDP transformer layer class={args.fsdp_layer_cls}")

    training_args = build_training_arguments(
        args=args,
        use_cuda=use_cuda,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        world_size=world_size,
    )

    if args.training_method == "regression":
        scaling = regression_metadata["target_scaling"]

        trainer = RegressionFETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            regression_tokenizer=tokenizer,
            regression_max_length=args.max_seq_len,
            text_prefix=args.text_prefix,
            train_min=scaling["train_min"],
            train_max=scaling["train_max"],
        )
    else:
        trainer = PairwiseFETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            epsilon=args.epsilon,
            scale=args.scale,
            loss=args.loss,
            loss_normalization=args.loss_normalization,
            pairwise_eval_tokenizer=tokenizer,
            pairwise_eval_max_length=args.max_seq_len,
            length_diagnostics=args.length_diagnostics,
            length_plot_num_bins=args.length_plot_num_bins,
            length_plot_max_pairs=args.length_plot_max_pairs,
        )

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
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            rank=rank,
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

    if args.training_method == "regression":
        scaling = regression_metadata["target_scaling"]

        metrics_test = evaluate_regression_distributed(
            model=trainer.model,
            dataset=test_dataset,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            batch_size=max(1, args.per_device_eval_batch_size),
            text_prefix=args.text_prefix,
            train_min=scaling["train_min"],
            train_max=scaling["train_max"],
        )

        baselines = None

    else:
        metrics_test = evaluate_pairwise_accuracy_distributed(
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

        baselines = baseline_pairwise_accuracies(test_dataset)

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


if __name__ == "__main__":
    main()
