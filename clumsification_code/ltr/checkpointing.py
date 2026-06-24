import gc
import os
from typing import Optional

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from .modeling import LTRModel
from .utils import (
    assert_finite_state_dict,
    assert_uniform_floating_dtype,
    get_preferred_param_dtype,
    logger,
    strip_known_prefixes,
)


def load_ltr_model(
    final_dir: str,
    attn_implementation: str = "sdpa",
    param_dtype: Optional[torch.dtype] = None,
    map_location: str = "cpu",
) -> LTRModel:
    head_path = os.path.join(final_dir, "ltr_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"Could not find ranking head at {head_path}. "
            "Pass the trainer final directory, e.g. output_dir/final."
        )

    head_state = torch.load(head_path, map_location=map_location)
    param_dtype = param_dtype or get_preferred_param_dtype()

    scorer_state = {
        k.removeprefix("scorer."): v
        for k, v in head_state["scorer"].items()
    }

    model = LTRModel(
        model_name=final_dir,
        hidden_dim=head_state["hidden_dim"],
        dropout=head_state["dropout"],
        attn_implementation=attn_implementation,
        param_dtype=param_dtype,
    )

    model.scorer.load_state_dict(scorer_state, strict=True)
    model.to(dtype=param_dtype)

    assert_uniform_floating_dtype(
        model,
        expected_dtype=param_dtype,
        name="loaded LTRModel",
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
    hidden_dim: int,
    dropout: float,
    rank: int,
) -> str:
    """
    Saves:
        output_dir/final/
            HF encoder files
            tokenizer files
            ltr_head.pt
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

        scorer_state_dict = {
            k.removeprefix("scorer."): v
            for k, v in cleaned_state_dict.items()
            if k.startswith("scorer.")
        }

        assert_finite_state_dict(encoder_state_dict, "encoder_state_dict")
        assert_finite_state_dict(scorer_state_dict, "scorer_state_dict")

        if not encoder_state_dict:
            raise RuntimeError(
                "encoder_state_dict is empty. State dict keys were not parsed correctly. "
                f"Example keys: {list(cleaned_state_dict.keys())[:20]}"
            )

        if not scorer_state_dict:
            raise RuntimeError(
                "scorer_state_dict is empty. State dict keys were not parsed correctly. "
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
                "scorer": scorer_state_dict,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            },
            os.path.join(final_dir, "ltr_head.pt"),
        )

        logger.info(f"Saved final model to {final_dir}")

        del cleaned_state_dict
        del encoder_state_dict
        del scorer_state_dict

    del state_dict

    cleanup_memory()

    trainer.accelerator.wait_for_everyone()

    return final_dir