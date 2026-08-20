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
        candidate = "" if text is None else str(text)
        inp = f"source:  candidate: {candidate}"

        tok = self.tokenizer(
            inp,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )

        # Match official MetricX predict.py behavior: remove EOS.
        if len(tok["input_ids"]) > 0:
            tok["input_ids"] = tok["input_ids"][:-1]
            tok["attention_mask"] = tok["attention_mask"][:-1]

        return tok

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
