import json
import os

import torch
import torch.distributed as dist
from datasets import concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments, set_seed

import dataset_functions as d_f

from ltr.args import parse_args
from ltr.checkpointing import save_final_model
from ltr.collators import GroupAllPairsCollator
from ltr.evaluation import baseline_winrates, evaluate_win_rate_distributed
from ltr.modeling import LTRModel
from ltr.trainer import PairwiseLTRTrainer
from ltr.utils import configure_logging, get_preferred_param_dtype, logger


os.environ["WANDB_MODE"] = "disabled"
os.environ["ACCELERATE_USE_FSDP"] = "true"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"


def build_training_arguments(args, use_cuda: bool, use_bf16: bool, use_fp16: bool, world_size: int):
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

    return TrainingArguments(**training_kwargs)


def log_dtype_counts(model) -> None:
    dtype_counts = {}

    for name, param in model.named_parameters():
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + param.numel()
        if param.dtype == torch.float32:
            logger.info(f"FP32 parameter: {name}, shape={tuple(param.shape)}")

    logger.info(f"Parameter dtype counts: {dtype_counts}")


def prepare_datasets(args):
    dataset_names = args.custom_datasets
    max_layers = args.max_layers

    all_datasets = []
    for ds_name in dataset_names:
        logger.info(f"Loading dataset: {ds_name} (layer_type={args.layer_type})")
        ds = d_f.format_custom_dataset(
            ds_name,
            max_layers,
            layer_type=args.layer_type,
            seed=args.seed,
        )
        ds = d_f.shuffle_and_transform_formatted_dataset(ds, seed=args.seed)
        logger.info(f"  → {ds_name}: {len(ds)} examples")
        all_datasets.append(ds)

    if len(all_datasets) == 1:
        merged = all_datasets[0]
    else:
        merged = concatenate_datasets(all_datasets)
        logger.info(
            f"Merged {len(all_datasets)} datasets → {len(merged)} total examples"
        )

    # Shuffle the merged dataset so examples from different sources are interleaved
    merged = merged.shuffle(seed=args.seed)

    if args.downsample_size is not None:
        merged = merged.select(range(min(args.downsample_size, len(merged))))

    split = merged.train_test_split(test_size=0.3, seed=args.seed)

    train_dataset = split["train"].shuffle(seed=args.seed)

    dev_test = split["test"].train_test_split(test_size=0.5, seed=args.seed)
    dev_dataset = dev_test["train"].shuffle(seed=args.seed)
    test_dataset = dev_test["test"].shuffle(seed=args.seed)

    return train_dataset, dev_dataset, test_dataset


def main():
    configure_logging()

    args = parse_args()
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

    if rank == 0:
        logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        logger.info(f"Datasets to load: {args.custom_datasets}")
        if eval_only:
            logger.info(
                "Running in --eval_only mode: training will be skipped. "
                f"Evaluating the model supplied via --model_name ({args.model_name})."
            )

    train_dataset, dev_dataset, test_dataset = prepare_datasets(args)

    if rank == 0:
        logger.info(train_dataset)
        logger.info(
            f"train={len(train_dataset)} dev={len(dev_dataset)} test={len(test_dataset)}"
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

    model = LTRModel(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation,
        param_dtype=param_dtype,
    )

    if model.encoder.config.pad_token_id is None:
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    data_collator = GroupAllPairsCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and param_dtype == torch.bfloat16
    use_fp16 = use_cuda and param_dtype == torch.float16

    if rank == 0:
        logger.info(f"use_cuda={use_cuda}")
        logger.info(f"use_bf16={use_bf16}")
        logger.info(f"loss={args.loss}")
        logger.info(f"loss_normalization={args.loss_normalization}")
        logger.info(f"FSDP transformer layer class={args.fsdp_layer_cls}")

    training_args = build_training_arguments(
        args=args,
        use_cuda=use_cuda,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        world_size=world_size,
    )

    trainer = PairwiseLTRTrainer(
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
        win_rate_tokenizer=tokenizer,
        win_rate_max_length=args.max_seq_len,
        length_diagnostics=args.length_diagnostics,
        length_plot_num_bins=args.length_plot_num_bins,
        length_plot_max_pairs=args.length_plot_max_pairs,
    )

    if rank == 0:
        log_dtype_counts(model)

    # -----------------------------------------------------------------
    # Training (skipped entirely when --eval_only is set)
    # -----------------------------------------------------------------
    final_dir = args.output_dir          # default used in eval-only mode
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

        # -----------------------------------------------------------------
        # HPO-safe dev evaluation.
        # This evaluates ONLY the internal dev split, not external benchmarks.
        # Use these metrics for hyperparameter selection.
        # -----------------------------------------------------------------
        if getattr(args, "hpo_mode", False):
            hpo_dev_metrics = trainer.evaluate(
                eval_dataset=dev_dataset,
                metric_key_prefix=getattr(args, "hpo_metric_prefix", "hpo_dev"),
            )

            trainer.accelerator.wait_for_everyone()

            if rank == 0:
                hpo_metrics_path = os.path.join(args.output_dir, "hpo_dev_metrics.json")
                with open(hpo_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(hpo_dev_metrics, f, indent=2)

                logger.info(f"HPO dev metrics: {hpo_dev_metrics}")
                logger.info(f"Saved HPO dev metrics to {hpo_metrics_path}")

    else:
        if rank == 0:
            logger.info("Skipping training and model saving (--eval_only mode).")
            os.makedirs(final_dir, exist_ok=True)

    trainer.accelerator.wait_for_everyone()

    # ---------------------------------------------------------------------
    # Optional final test evaluation.
    # During HPO, skip this. After choosing the best HPs, run the best model
    # once on your real benchmarks / final test sets.
    # ---------------------------------------------------------------------
    if getattr(args, "skip_final_test_eval", False):
        if rank == 0:
            logger.info("Skipping final test evaluation because --skip_final_test_eval was set.")

        trainer.accelerator.wait_for_everyone()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        return

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

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test": metrics_test,
                    "baselines": baselines,
                    "hpo_dev": hpo_dev_metrics,
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