"""
Evaluate MetricX-24 in QE (no-reference) mode on the same benchmark suite used by evaluate_model_on_benchmark.py, so results are directly comparable.

Expected layout:
- this file lives alongside evaluate_model_on_benchmark.py
- benchmark data files are in data/benchmarks/...
- metricx24 package is importable (from google-research/metricx)

Example:
python evaluate_metricx24_on_benchmark.py \
  --model-name MetricX24_QE \
  --training-dataset baseline \
  --perturbation-type none \
  --num-layers 0 \
  --context-length 1536 \
  --metricx-model-name-or-path google/metricx-24-hybrid-xl-v2p6 \
  --tokenizer google/mt5-xl \
  --batch-size 8 \
  --max-input-length 1536
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
import transformers

# metricx24 must be available in PYTHONPATH / environment
from scripts.evals.metricx24 import models

# Reuse your existing benchmark loaders/metrics/writer
import scripts.evals.evaluate_model_on_benchmark as bench


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MetricX-24 QE on the full benchmark suite and log JSONL results."
    )

    # Output metadata
    parser.add_argument("--model-name", type=str, default="MetricX24_QE")
    parser.add_argument("--training-dataset", type=str, default="baseline")
    parser.add_argument("--perturbation-type", type=str, default="none")
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=1536)

    # MetricX config
    parser.add_argument(
        "--metricx-model-name-or-path",
        type=str,
        default="google/metricx-24-hybrid-xl-v2p6",
    )
    parser.add_argument("--tokenizer", type=str, default="google/mt5-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=1536)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"MetricX model: {args.metricx_model_name_or_path}")
    print(f"Tokenizer: {args.tokenizer}")

    model = MetricX24QEInferenceModel(
        model_name_or_path=args.metricx_model_name_or_path,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        return_higher_is_better=True,
    )
    # Sanity probe
    probe = [
        "This is a fluent, grammatical sentence.",
        "This sentence broken grammar bad.",
    ]

    raw_probe_scores = model.raw_metricx_scores(probe)
    adapted_probe_scores = model.score_texts(probe)

    print("Raw MetricX scores, lower is better:", raw_probe_scores)
    print("Adapted benchmark scores, higher is better:", adapted_probe_scores)
    print()

    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_input_length  # passed through; adapter uses max_input_length
    all_results: Dict[str, Any] = {}

    print("=" * 60)
    print(" Preference-style HF benchmarks")
    print("=" * 60)

    jfleg_results = bench.eval_jfleg_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in jfleg_results.items():
        all_results.update(bench._flatten_preference_metrics(bench_name, metrics))

    multiblimp_metrics = bench.eval_multiblimp_english_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        bench._flatten_preference_metrics(
            "MultiBLiMP_eng_minimal_pair_preference", multiblimp_metrics
        )
    )

    storycloze_results = bench.eval_story_cloze_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in storycloze_results.items():
        all_results.update(bench._flatten_preference_metrics(bench_name, metrics))

    print()
    print("=" * 60)
    print(" Scalar human-score benchmarks")
    print("=" * 60)

    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]
    raw_preds = bench.getModelPreds(device, model, summeval_texts, BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle([x for y in ds["fluency"] for x in y], raw_preds, "summeval_fluency"))
    all_results.update(bench.correlation_bundle([x for y in ds["coherence"] for x in y], raw_preds, "summeval_coherence"))
    all_results.update(bench.correlation_bundle([x for y in ds["consistency"] for x in y], raw_preds, "summeval_consistency"))

    ellipse_ds = bench.load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    raw_preds = bench.getModelPreds(device, model, ellipse_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(ellipse_ds["overall"], raw_preds, "ellipse_overall"))
    all_results.update(bench.correlation_bundle(ellipse_ds["cohesion"], raw_preds, "ellipse_cohesion"))

    tc_texts, tc_overall_labels = bench.load_data_usr("data/benchmarks/tc_usr_data.json", "Overall")
    raw_preds = bench.getModelPreds(device, model, tc_texts, BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(tc_overall_labels, raw_preds, "tc_overall"))
    _, tc_natural_labels = bench.load_data_usr("data/benchmarks/tc_usr_data.json", "Natural")
    all_results.update(bench.correlation_bundle(tc_natural_labels, raw_preds, "tc_natural"))

    pc_texts, pc_overall_labels = bench.load_data_usr("data/benchmarks/pc_usr_data.json", "Overall")
    raw_preds = bench.getModelPreds(device, model, pc_texts, BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(pc_overall_labels, raw_preds, "pc_overall"))
    _, pc_natural_labels = bench.load_data_usr("data/benchmarks/pc_usr_data.json", "Natural")
    all_results.update(bench.correlation_bundle(pc_natural_labels, raw_preds, "pc_natural"))

    meva_texts_roc, meva_labels_roc = bench.load_data_openmeva("data/benchmarks/mans_roc.json")
    meva_texts_wp, meva_labels_wp = bench.load_data_openmeva("data/benchmarks/mans_wp.json")
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    raw_preds = bench.getModelPreds(device, model, meva_texts, BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(meva_labels, raw_preds, "OpenMEVA_overall"))

    webnlg_texts, webnlg_labels = bench.load_data_webnlg("data/benchmarks/web_nlg_2020_human_evals_en.json")
    raw_preds = bench.getModelPreds(device, model, webnlg_texts, BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(webnlg_labels, raw_preds, "WebNLG_fluency"))

    hanna_ds = bench.load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    raw_preds = bench.getModelPreds(device, model, hanna_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(hanna_ds["coherence"], raw_preds, "HANNA_coherence"))
    all_results.update(bench.correlation_bundle(hanna_ds["complexity"], raw_preds, "HANNA_complexity"))

    arge_ds = bench.load_argessay_data("data/benchmarks/arg-essay.csv")
    raw_preds = bench.getModelPreds(device, model, arge_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(arge_ds["language_mastery"], raw_preds, "ARG-ESSAY_language_mastery"))
    all_results.update(bench.correlation_bundle(arge_ds["complexity"], raw_preds, "ARG-ESSAY_complexity"))
    all_results.update(bench.correlation_bundle(arge_ds["vocabulary"], raw_preds, "ARG-ESSAY_vocabulary"))
    all_results.update(bench.correlation_bundle(arge_ds["language_constructs"], raw_preds, "ARG-ESSAY_language_constructs"))

    hr_ds = bench.load_human_ratings_of_nlg_data("data/benchmarks/human_ratings_of_nlg.csv")
    raw_preds = bench.getModelPreds(device, model, hr_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(hr_ds["quality"], raw_preds, "HumanRatings_quality"))
    all_results.update(bench.correlation_bundle(hr_ds["naturalness"], raw_preds, "HumanRatings_naturalness"))

    turn_ds, whole_ds = bench.load_fed_data("data/benchmarks/fed_data.json")
    raw_preds_turn = bench.getModelPreds(device, model, turn_ds["text"], BATCH_SIZE, MAX_LENGTH)
    raw_preds_whole = bench.getModelPreds(device, model, whole_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(turn_ds["fluent"], raw_preds_turn, "FED_turn_fluency"))
    all_results.update(bench.correlation_bundle(turn_ds["overall"], raw_preds_turn, "FED_turn_overall"))
    all_results.update(bench.correlation_bundle(whole_ds["overall"], raw_preds_whole, "FED_whole_overall"))

    e2e_ds = bench.load_e2e_data("data/benchmarks/E2E_data")
    raw_preds = bench.getModelPreds(device, model, e2e_ds["text"], BATCH_SIZE, MAX_LENGTH)
    all_results.update(bench.correlation_bundle(e2e_ds["naturalness"], raw_preds, "E2E_naturalness"))
    all_results.update(bench.correlation_bundle(e2e_ds["quality"], raw_preds, "E2E_quality"))

    bench.write_results_jsonl(
        model_name=args.model_name,
        training_dataset=args.training_dataset,
        perturbation_type=args.perturbation_type,
        num_layers=args.num_layers,
        context_length=args.context_length,
        model_dir=args.metricx_model_name_or_path,
        results=all_results,
    )


if __name__ == "__main__":
    main()