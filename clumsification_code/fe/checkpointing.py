# This script has been co-created, refactored, and cleaned using GPT 5.6.
import gc
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

    head_path = os.path.join(final_dir, "fe_head.pt")
    if not os.path.exists(head_path):
        from clumsification_code.compat.fe_checkpoints import find_legacy_head

        head_path = find_legacy_head(final_dir)

    head_state = torch.load(head_path, map_location=map_location)
    if "evaluation_head" not in head_state:
        from clumsification_code.compat.fe_checkpoints import normalize_legacy_head_state

        head_state = normalize_legacy_head_state(head_state)
    param_dtype = param_dtype or get_preferred_param_dtype()

    evaluation_head_state = head_state["evaluation_head"]

    legacy_head = any(
        key.startswith("net.0.") or key.startswith("net.3.")
        for key in evaluation_head_state
    )
    model = FEModel(
        model_name=final_dir,
        hidden_dim=head_state.get("hidden_dim", 256),
        dropout=head_state.get("dropout", 0.1),
        attn_implementation=attn_implementation,
        param_dtype=param_dtype,
        legacy_head=legacy_head,
        pooling="mean",
    )

    model.evaluation_head.load_state_dict(evaluation_head_state, strict=True)
    model.to(dtype=param_dtype)

    assert_uniform_floating_dtype(
        model,
        expected_dtype=param_dtype,
        name="loaded FEModel",
    )

    return model


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
