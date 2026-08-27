# This script has been co-created, refactored, and cleaned using GPT 5.6.
import gc
import os
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
        torch.cuda.ipc_collect()


def save_final_model(
    trainer,
    tokenizer,
    output_dir: str,
    rank: int,
    metadata: Optional[dict] = None,
) -> str:
    """
    Saves:
        output_dir/final/
            HF encoder files
            tokenizer files
            fe_head.pt
    """
    final_dir = os.path.join(output_dir, "final")

    if rank == 0:
        os.makedirs(final_dir, exist_ok=True)

    trainer.accelerator.wait_for_everyone()

    trainer.optimizer = None
    trainer.lr_scheduler = None

    cleanup_memory()

    trainer.accelerator.wait_for_everyone()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod

    state_dict = get_model_state_dict(
        trainer.model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    if rank == 0:
        unwrapped.save_pretrained(final_dir, tokenizer=tokenizer, metadata=metadata)

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

        assert_finite_state_dict(encoder_state_dict, "encoder_state_dict")
        assert_finite_state_dict(evaluation_head_state_dict, "evaluation_head_state_dict")

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

        unwrapped.encoder.save_pretrained(
            final_dir,
            state_dict=encoder_state_dict,
            safe_serialization=True,
            max_shard_size="2GB",
        )

        tokenizer.save_pretrained(final_dir)

        torch.save(
            {
                "evaluation_head": evaluation_head_state_dict,
                "head_type": "linear",
            },
            os.path.join(final_dir, "fe_head.pt"),
        )

        logger.info(f"Saved final model to {final_dir}")

        del cleaned_state_dict
        del encoder_state_dict
        del evaluation_head_state_dict

    del state_dict

    cleanup_memory()

    trainer.accelerator.wait_for_everyone()

    return final_dir
