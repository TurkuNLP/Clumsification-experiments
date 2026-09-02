# This script has been co-created, refactored, and cleaned using GPT 5.6.
import tempfile
from typing import Dict, List

import datasets
import numpy as np
import torch
import transformers

# metricx24 must be available in PYTHONPATH / environment
from clumsification_code.evals.metricx24 import models


class MetricX24QEInferenceModel:
    """MetricX-24 QE adapter with explicit no-source and source-aware entry points.

    ``score_texts`` preserves the direct benchmark protocol and supplies an
    empty source. ``score_pairs`` is the custom-dataset teacher protocol and
    supplies the original source plus candidate text. Keeping these methods
    separate prevents an accidental change in benchmark semantics.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name: str = "google/mt5-xl",
        batch_size: int = 8,
        max_input_length: int = 1536,
        return_higher_is_better: bool = True,
    ):
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.return_higher_is_better = return_higher_is_better

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = models.MT5ForRegression.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
        )
        self.model.eval()

        if torch.cuda.is_available():
            n_gpus = max(torch.cuda.device_count(), 1)
            per_device_eval_batch_size = max(batch_size // n_gpus, 1)
        else:
            per_device_eval_batch_size = batch_size

        data_collator = transformers.DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )

        tmp_output_dir = tempfile.mkdtemp(prefix="metricx24_eval_")
        training_args = transformers.TrainingArguments(
            output_dir=tmp_output_dir,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_pin_memory=False,
            report_to=[],
        )

        self.trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
        )

    def _prep_one(self, text: str) -> Dict[str, List[int]]:
        # MetricX-24 QE input format:
        #   source: <source> candidate: <hypothesis>
        # Here source is intentionally empty because this benchmark is no-source/no-reference.
        return self._prep_input(source="", candidate=text)

    def _prep_input(self, source: str, candidate: str) -> Dict[str, List[int]]:
        """Tokenize one MetricX-24 QE record and remove its terminal EOS."""
        source = "" if source is None else str(source)
        candidate = "" if candidate is None else str(candidate)
        inp = f"source: {source} candidate: {candidate}"

        tok = self.tokenizer(inp, max_length=self.max_input_length,
                             truncation=True, padding=False)
        # Right truncation can discard the candidate when the source is long.
        # Rebuild overflowed inputs with a guaranteed candidate segment and
        # spend remaining budget on the source/prefix instead.
        untruncated = self.tokenizer(inp, truncation=False, padding=False)
        if len(untruncated["input_ids"]) > self.max_input_length:
            candidate_ids = self.tokenizer(
                str(candidate), truncation=False, padding=False
            )["input_ids"]
            prefix_ids = self.tokenizer(
                f"source: {source} candidate: ", truncation=False, padding=False
            )["input_ids"]
            budget = max(self.max_input_length - len(candidate_ids), 1)
            input_ids = prefix_ids[:budget] + candidate_ids[: self.max_input_length - min(len(prefix_ids), budget)]
            tok = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        # Match official MetricX predict.py behavior: remove EOS.
        if len(tok["input_ids"]) > 0:
            tok["input_ids"] = tok["input_ids"][:-1]
            tok["attention_mask"] = tok["attention_mask"][:-1]

        return tok

    def _prep_pair(self, source: str, candidate: str) -> Dict[str, List[int]]:
        """Prepare the official MetricX-24 QE source-plus-candidate format."""
        return self._prep_input(source=source, candidate=candidate)

    @torch.no_grad()
    def raw_metricx_scores(self, texts: List[str]) -> np.ndarray:
        """
        Returns raw MetricX predictions.

        Important: raw MetricX scores are lower-is-better.
        """
        feats = [self._prep_one(t) for t in texts]
        ds = datasets.Dataset.from_list(feats)

        preds, _, _ = self.trainer.predict(test_dataset=ds)
        preds = np.asarray(preds).reshape(-1).astype(np.float64)
        return preds

    @torch.no_grad()
    def raw_metricx_pair_scores(
        self, sources: List[str], candidates: List[str]
    ) -> np.ndarray:
        """Return raw lower-is-better errors for source/candidate pairs."""
        if len(sources) != len(candidates):
            raise ValueError("sources and candidates must have the same length.")
        feats = [
            self._prep_pair(source, candidate)
            for source, candidate in zip(sources, candidates)
        ]
        ds = datasets.Dataset.from_list(feats)
        preds, _, _ = self.trainer.predict(test_dataset=ds)
        return np.asarray(preds).reshape(-1).astype(np.float64)

    @torch.no_grad()
    def score_pairs(
        self, sources: List[str], candidates: List[str]
    ) -> np.ndarray:
        """Return higher-is-better scores while retaining source context."""
        scores = self.raw_metricx_pair_scores(sources, candidates)
        return -scores if self.return_higher_is_better else scores

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device=None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Adapter expected by evaluate_model_on_benchmark.py.

        By default returns higher-is-better scores by negating raw MetricX
        lower-is-better error scores.
        """
        preds = self.raw_metricx_scores(texts)

        if self.return_higher_is_better:
            return -preds

        return preds
