# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Model-independent UniEval scoring implementations.

Generative models reproduce UniEval's normalized Yes/No token score.  Encoder
models use the same binary target with a small scalar head, which allows an
embedding model such as Qwen3-Embedding to use the released supervision.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from pathlib import Path
import json


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final unmasked token, as recommended for Qwen embeddings."""
    lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[indices, lengths]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


class UniEvalEncoderClassifier(nn.Module):
    """An encoder plus binary head returning UniEval-compatible scores."""

    def __init__(self, encoder: nn.Module, hidden_size: int,
                 pooling: Literal["last_token", "mean"] = "last_token"):
        super().__init__()
        if pooling not in {"last_token", "mean"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, pooling: Literal["last_token", "mean"] = "last_token", **kwargs):
        """Load any Hugging Face encoder and attach the UniEval head."""
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        # This classifier applies its own last-token or mean pooling.  Remove
        # any pretrained pooler branch so DDP does not see trainable parameters
        # that are disconnected from the loss.
        if getattr(encoder, "pooler", None) is not None:
            encoder.pooler = None
        hidden_size = getattr(encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(encoder.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer encoder hidden size from config")
        return cls(encoder, hidden_size, pooling=pooling)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if not hasattr(output, "last_hidden_state"):
            raise TypeError("Encoder output must expose last_hidden_state")
        pool = last_token_pool if self.pooling == "last_token" else mean_pool
        return pool(output.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        # Some embedding checkpoints (including Qwen3-Embedding) load in
        # BF16 while a freshly-created Linear head defaults to FP32.  Cast the
        # pooled representation to the head dtype so mixed-precision training
        # works before an accelerator/FSDP wrapper manages parameter casting.
        features = self.encode(input_ids, attention_mask).to(self.classifier.weight.dtype)
        logits = self.classifier(features).squeeze(-1)
        result = {"logits": logits, "scores": torch.sigmoid(logits)}
        if labels is not None:
            result["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float()
            )
        return result

    def save_pretrained(
        self,
        save_directory: str,
        *,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """Save encoder weights and the UniEval head as one checkpoint."""
        del kwargs
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        complete_state = self.state_dict() if state_dict is None else state_dict
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in complete_state.items()
            if key.startswith("encoder.")
        }
        classifier_state = {
            key.removeprefix("classifier."): value
            for key, value in complete_state.items()
            if key.startswith("classifier.")
        }
        if not encoder_state or not classifier_state:
            raise RuntimeError(
                "Cannot save UniEval checkpoint: complete encoder/classifier state is missing"
            )
        self.encoder.save_pretrained(directory, state_dict=encoder_state)
        torch.save(classifier_state, directory / "unieval_classifier.pt")
        (directory / "unieval_model_config.json").write_text(
            json.dumps({"pooling": self.pooling, "hidden_size": self.classifier.in_features}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, *, base_model: str | None = None,
                        trust_remote_code: bool = True, **kwargs):
        """Load a checkpoint produced by :meth:`save_pretrained`."""
        directory = Path(checkpoint_dir)
        config_path = directory / "unieval_model_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            # Trainer checkpoints may contain the head in model.safetensors
            # but not the custom sidecar metadata.
            config = {"pooling": "last_token", "hidden_size": None}
        source = str(directory) if (directory / "config.json").exists() else base_model
        if source is None:
            raise ValueError("Checkpoint has no config.json; provide base_model")
        model = cls.from_pretrained(source, pooling=config["pooling"], trust_remote_code=trust_remote_code, **kwargs)
        head_path = directory / "unieval_classifier.pt"
        if head_path.exists():
            state = torch.load(head_path, map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            tensors = load_file(str(directory / "model.safetensors"))
            state = {k.removeprefix("classifier."): v for k, v in tensors.items() if k.startswith("classifier.")}
            encoder_state = {k.removeprefix("encoder."): v for k, v in tensors.items() if k.startswith("encoder.")}
            if encoder_state:
                model.encoder.load_state_dict(encoder_state, strict=False)
            if not state:
                raise FileNotFoundError(f"No UniEval classifier found in {directory}")
        model.classifier.load_state_dict(state)
        return model


def binary_targets(labels: list[str] | torch.Tensor) -> torch.Tensor:
    """Convert released UniEval ``Yes``/``No`` labels to float targets."""
    if isinstance(labels, torch.Tensor):
        return labels.float()
    try:
        values = []
        for label in labels:
            if label not in {"Yes", "No"}:
                raise ValueError(f"Unknown UniEval label: {label!r}")
            values.append(1.0 if label == "Yes" else 0.0)
        return torch.tensor(values)
    except TypeError as exc:
        raise TypeError("labels must be a sequence of 'Yes'/'No' strings") from exc
