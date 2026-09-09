# This script has been co-created, refactored, and cleaned using GPT 5.6.
import gc
import json
import os
import time
from typing import Optional

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from .modeling import FEModel
from .utils import (
    assert_finite_state_dict,
    assert_uniform_floating_dtype,
    get_preferred_param_dtype,
    logger,
    strip_known_prefixes,
)


TRAINER_CHECKPOINT_CONFIG = "fe_trainer_checkpoint.json"


def write_trainer_checkpoint_metadata(
    *,
    checkpoint_dir: str,
    model_name: str,
    pooling: str,
    max_seq_len: int,
    objective: str,
) -> None:
    """Write the information needed to evaluate a standard Trainer checkpoint."""
    with open(
        os.path.join(checkpoint_dir, TRAINER_CHECKPOINT_CONFIG),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "model_name": model_name,
                "pooling": pooling,
                "max_seq_len": max_seq_len,
                "objective": objective,
            },
            handle,
            indent=2,
        )


def _load_trainer_checkpoint(
    checkpoint_dir: str,
    *,
    attn_implementation: str,
    param_dtype: Optional[torch.dtype],
    map_location: str,
) -> FEModel:
    metadata_path = os.path.join(checkpoint_dir, TRAINER_CHECKPOINT_CONFIG)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported Trainer checkpoint metadata: {metadata_path}")
    model_name = metadata.get("model_name")
    pooling = metadata.get("pooling")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Trainer checkpoint metadata is missing model_name")
    if not isinstance(pooling, str) or not pooling:
        raise ValueError("Trainer checkpoint metadata is missing pooling")

    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    torch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file

        state_dict = load_file(safe_path, device=map_location)
    elif os.path.exists(torch_path):
        state_dict = torch.load(torch_path, map_location=map_location, weights_only=True)
    else:
        raise FileNotFoundError(
            f"Trainer checkpoint has no model weights: {checkpoint_dir}"
        )
    cleaned_state_dict = {
        strip_known_prefixes(name): tensor for name, tensor in state_dict.items()
    }
    if not any(name.startswith("evaluation_head.") for name in cleaned_state_dict):
        raise ValueError("Trainer checkpoint is missing the FE evaluation head")
    model = FEModel(
        model_name=model_name,
        attn_implementation=attn_implementation,
        param_dtype=param_dtype,
        pooling=pooling,
    )
    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def load_fe_model(
    final_dir: str,
    attn_implementation: str = "sdpa",
    param_dtype: Optional[torch.dtype] = None,
    map_location: str = "cpu",
) -> FEModel:
    complete_state_path = os.path.join(final_dir, "fe_model_state.pt")
    complete_config_path = os.path.join(final_dir, "fe_model_config.json")
    if os.path.exists(complete_state_path) and os.path.exists(complete_config_path):
        return FEModel.from_pretrained(
            final_dir,
            attn_implementation=attn_implementation,
            param_dtype=param_dtype,
        )

    trainer_metadata_path = os.path.join(final_dir, TRAINER_CHECKPOINT_CONFIG)
    if os.path.exists(trainer_metadata_path):
        return _load_trainer_checkpoint(
            final_dir,
            attn_implementation=attn_implementation,
            param_dtype=param_dtype,
            map_location=map_location,
        )

    raise FileNotFoundError(
        "No supported FE checkpoint was found. Expected either a complete "
        "FE checkpoint or a Trainer checkpoint with fe_model_config.json."
    )


def cleanup_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_final_model(
    trainer,
    tokenizer,
    output_dir: str,
    rank: int,
    parallelism: str = "ddp",
    metadata: Optional[dict] = None,
) -> str:
    """
    Saves:
        output_dir/final/
            HF encoder files
            tokenizer files
            fe_model_config.json
            fe_model_state.pt
            fe_head.pt
    """
    final_dir = os.path.join(output_dir, "final")

    started_at = time.monotonic()

    if rank == 0:
        os.makedirs(final_dir, exist_ok=True)
        logger.info("Final save: preparing all ranks")

    trainer.accelerator.wait_for_everyone()

    trainer.optimizer = None
    trainer.lr_scheduler = None

    if rank == 0:
        logger.info("Final save: releasing optimizer memory")
    cleanup_memory()

    trainer.accelerator.wait_for_everyone()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    gather_started_at = time.monotonic()
    if parallelism == "fsdp":
        if rank == 0:
            logger.info(
                "Final save: gathering the full FSDP model state on CPU; "
                "all ranks must participate"
            )
        state_dict = get_model_state_dict(
            trainer.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    elif rank == 0:
        logger.info("Final save: reading complete unwrapped DDP/model state on CPU")
        state_dict = {
            key: value.detach().cpu()
            for key, value in unwrapped.state_dict().items()
        }
    else:
        state_dict = {}
    if rank == 0:
        logger.info(
            "Final save: full model state gathered in %.1f seconds (%d tensors)",
            time.monotonic() - gather_started_at,
            len(state_dict),
        )

    if rank == 0:
        validation_started_at = time.monotonic()
        logger.info("Final save: validating gathered tensors")
        cleaned_state_dict = {
            strip_known_prefixes(k): v
            for k, v in state_dict.items()
        }

        assert_finite_state_dict(cleaned_state_dict, "final full model state_dict")

        encoder_state_dict = {
            k.removeprefix("encoder."): v
            for k, v in cleaned_state_dict.items()
            if k.startswith("encoder.")
        }

        evaluation_head_state_dict = {
            k.removeprefix("evaluation_head."): v
            for k, v in cleaned_state_dict.items()
            if k.startswith("evaluation_head.")
        }

        if not encoder_state_dict:
            raise RuntimeError(
                "encoder_state_dict is empty. State dict keys were not parsed correctly. "
                f"Example keys: {list(cleaned_state_dict.keys())[:20]}"
            )

        if not evaluation_head_state_dict:
            raise RuntimeError(
                "evaluation_head_state_dict is empty. State dict keys were not parsed correctly. "
                f"Example keys: {list(cleaned_state_dict.keys())[:20]}"
            )

        logger.info(
            "Final save: tensor validation completed in %.1f seconds",
            time.monotonic() - validation_started_at,
        )

        write_started_at = time.monotonic()
        logger.info(
            "Final save: writing encoder, tokenizer, and complete FE state to %s",
            final_dir,
        )
        unwrapped.save_pretrained(
            final_dir,
            tokenizer=tokenizer,
            metadata=metadata,
            state_dict=cleaned_state_dict,
        )
        torch.save(
            {
                "evaluation_head": evaluation_head_state_dict,
                "head_type": "linear",
            },
            os.path.join(final_dir, "fe_head.pt"),
        )

        logger.info(
            "Final save: disk writes completed in %.1f seconds",
            time.monotonic() - write_started_at,
        )

        del cleaned_state_dict
        del encoder_state_dict
        del evaluation_head_state_dict

    del state_dict

    if rank == 0:
        logger.info("Final save: releasing gathered state and synchronizing ranks")
    cleanup_memory()

    trainer.accelerator.wait_for_everyone()

    if rank == 0:
        logger.info(
            "Final save completed in %.1f seconds: %s",
            time.monotonic() - started_at,
            final_dir,
        )

    return final_dir
