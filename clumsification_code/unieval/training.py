# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Hugging Face Trainer construction for UniEval Phase 2 models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

from .dataset import EncoderUniEvalCollator, GenerativeUniEvalCollator, UniEvalDataset
from .modeling import UniEvalEncoderClassifier


def _metrics(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    scores = torch.sigmoid(torch.tensor(logits.reshape(-1))).numpy()
    labels = labels.reshape(-1)
    return {"accuracy": float(((scores >= 0.5) == labels).mean())}


def train_unieval(*, model_name: str, data_file: str, output_dir: str,
                  model_type: str = "encoder", pooling: str = "last_token",
                  max_length: int = 1024, epochs: float = 1.0,
                  batch_size: int = 8, learning_rate: float = 5e-5,
                  dev_fraction: float = 0.05, seed: int = 42,
                  parallelism: str = "ddp",
                  fsdp_sharding_strategy: str = "shard_grad_op",
                  fsdp_layer_cls: str | None = None) -> None:
    """Train one UniEval Boolean-QA model and save a self-describing checkpoint."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if parallelism == "fsdp":
        if not torch.cuda.is_available() or world_size <= 1:
            raise ValueError("FSDP requires CUDA and WORLD_SIZE greater than one.")
        if not fsdp_layer_cls:
            raise ValueError("fsdp_layer_cls is required when parallelism='fsdp'.")
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    else:
        os.environ.pop("ACCELERATE_USE_FSDP", None)
        os.environ.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)

    dataset = UniEvalDataset.from_jsonl(data_file)
    train_dataset, dev_dataset = dataset.train_dev_split(dev_fraction, seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_type == "encoder":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = UniEvalEncoderClassifier.from_pretrained(
            model_name, pooling=pooling, trust_remote_code=True
        )
        collator = EncoderUniEvalCollator(tokenizer, max_length)
        metrics = _metrics
    elif model_type == "generative":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        collator = GenerativeUniEvalCollator(tokenizer, max_length)
        metrics = None
    else:
        raise ValueError("model_type must be 'encoder' or 'generative'")

    training_kwargs = {}
    if parallelism == "fsdp":
        training_kwargs.update(
            fsdp=f"{fsdp_sharding_strategy} auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": fsdp_layer_cls,
                "use_orig_params": True,
                "limit_all_gathers": True,
                "activation_checkpointing": False,
                "sync_module_states": True,
                "cpu_ram_efficient_loading": True,
                "cpu_offload": False,
            },
        )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **training_kwargs,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset,
        eval_dataset=dev_dataset, data_collator=collator,
        processing_class=tokenizer, compute_metrics=metrics,
    )
    trainer.train()
    metadata = {
        "schema": "unieval-training-v1",
        "model_name": model_name,
        "model_type": model_type,
        "pooling": pooling if model_type == "encoder" else None,
        "data_file": str(Path(data_file)),
        "rows": len(dataset),
        "train_rows": len(train_dataset),
        "dev_rows": len(dev_dataset),
        "max_length": max_length,
        "seed": seed,
        "parallelism": parallelism,
        "fsdp_sharding_strategy": (
            fsdp_sharding_strategy if parallelism == "fsdp" else None
        ),
        "world_size": world_size,
    }
    trainer.accelerator.wait_for_everyone()
    trainer.optimizer = None
    trainer.lr_scheduler = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.accelerator.wait_for_everyone()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    if parallelism == "fsdp":
        complete_state = get_model_state_dict(
            trainer.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    elif trainer.is_world_process_zero():
        complete_state = {
            key: value.detach().cpu()
            for key, value in unwrapped.state_dict().items()
        }
    else:
        complete_state = {}

    if trainer.is_world_process_zero():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Do not rely on Trainer.save_model for this custom nn.Module: it may
        # emit only a generic safetensors state dict and omit the encoder's
        # Hugging Face config. Explicitly invoke the model's serialization
        # method so the directory is directly reloadable for evaluation.
        unwrapped.save_pretrained(output_dir, state_dict=complete_state)
        tokenizer.save_pretrained(output_dir)
        (Path(output_dir) / "unieval_training_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Keep a stable, evaluation-ready copy separate from Trainer's
        # numbered checkpoints.  This is especially important when Trainer
        # saves a safetensors state dict without invoking the custom head's
        # save_pretrained method.
        final_dir = Path(output_dir) / "final"
        unwrapped.save_pretrained(final_dir, state_dict=complete_state)
        tokenizer.save_pretrained(final_dir)
        (final_dir / "unieval_training_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    trainer.accelerator.wait_for_everyone()
