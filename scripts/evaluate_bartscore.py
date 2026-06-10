"""
Evaluate BARTScore on the same benchmark suite used by evaluate_model_on_benchmark.py.

Usage example:
  python evaluate_bartscore_on_benchmark.py \
      --model-name BARTScore_cnn \
      --training-dataset baseline \
      --perturbation-type none \
      --num-layers 0 \
      --context-length 1024 \
      --checkpoint facebook/bart-large-cnn \
      --batch-size 8 \
      --max-length 512

Notes:
- This baseline is "reference-free" for scalar benchmarks by scoring each text against itself:
    BARTScore(text -> text)
- For pairwise preference benchmarks, each candidate is independently self-scored,
  then compared just like your existing QE model.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

#Reusing existing code
import evaluate_model_on_benchmark as base


#BARTScore implementation
class BARTScorer:
    def __init__(self, device: str = "cuda:0", max_length: int = 1024, checkpoint: str = "facebook/bart-large-cnn"):
        self.device = device
        self.max_length = max_length

        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        self.loss_fct = nn.NLLLoss(reduction="none", ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    @torch.no_grad()
    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4) -> List[float]:
        if len(srcs) != len(tgts):
            raise ValueError(f"Length mismatch: len(srcs)={len(srcs)} vs len(tgts)={len(tgts)}")

        score_list: List[float] = []
        for i in tqdm(range(0, len(srcs), batch_size), desc="BARTScore"):
            src_list = srcs[i:i + batch_size]
            tgt_list = tgts[i:i + batch_size]

            encoded_src = self.tokenizer(
                src_list,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded_tgt = self.tokenizer(
                tgt_list,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            src_tokens = encoded_src["input_ids"].to(self.device)
            src_mask = encoded_src["attention_mask"].to(self.device)

            tgt_tokens = encoded_tgt["input_ids"].to(self.device)
            tgt_mask = encoded_tgt["attention_mask"].to(self.device)
            tgt_len = tgt_mask.sum(dim=1)

            output = self.model(
                input_ids=src_tokens,
                attention_mask=src_mask,
                labels=tgt_tokens,
            )
            logits = output.logits.view(-1, self.model.config.vocab_size)
            loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
            loss = loss.view(tgt_tokens.shape[0], -1)
            loss = loss.sum(dim=1) / tgt_len  # average NLL per token

            curr_scores = (-loss).detach().cpu().float().tolist()
            score_list.extend(curr_scores)

        return score_list


def _clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def bartscore_text_quality_scores(
    scorer: BARTScorer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    src_mode: str = "self",
) -> np.ndarray:
    """
    Convert single-text evaluation into BARTScore source-target pairs.

    src_mode:
      - self: score(text -> text)     [default]
      - empty: score("" -> text)
    """
    texts = [_clean_text(t) for t in texts]
    scorer.max_length = max_length

    if src_mode == "self":
        srcs = texts
    elif src_mode == "empty":
        srcs = [""] * len(texts)
    else:
        raise ValueError(f"Unknown src_mode={src_mode}")

    tgts = texts
    scores = scorer.score(srcs, tgts, batch_size=batch_size)
    arr = np.asarray(scores, dtype=np.float64)

    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0][:5]
        raise RuntimeError(f"Non-finite BARTScore outputs at indices: {bad.tolist()}")

    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BARTScore baseline on LTR benchmark suite and log results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Metadata (kept same schema as original script)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--training-dataset", type=str, required=True)
    parser.add_argument("--perturbation-type", type=str, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)

    # BARTScore config
    parser.add_argument("--checkpoint", type=str, default="facebook/bart-large-cnn")
    parser.add_argument(
        "--bartscore-weights",
        type=str,
        default=None,
        help="Optional path to finetuned bart.pth (ParaBank-style); if omitted, use checkpoint weights only.",
    )
    parser.add_argument("--src-mode", type=str, default="self", choices=["self", "empty"])

    # Inference config
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"src_mode    : {args.src_mode}")
    print()

    scorer = BARTScorer(
        device=device,
        max_length=args.max_length,
        checkpoint=args.checkpoint,
    )

    if args.bartscore_weights:
        if not os.path.exists(args.bartscore_weights):
            raise FileNotFoundError(f"--bartscore-weights not found: {args.bartscore_weights}")
        scorer.load(args.bartscore_weights)
        print(f"Loaded finetuned weights: {args.bartscore_weights}")

    if args.fp16 and device.startswith("cuda"):
        scorer.model.half()
        print("Using fp16 for BART model.")

    # Monkey-patch getModelPreds used inside base preference evaluators
    def _patched_get_model_preds(_device, _model, test_texts, batch_size=32, max_length=512):
        return bartscore_text_quality_scores(
            scorer=_model,
            texts=test_texts,
            batch_size=batch_size,
            max_length=max_length,
            src_mode=args.src_mode,
        )

    base.getModelPreds = _patched_get_model_preds

    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    all_results: Dict[str, Any] = {}

    # ───────────────── Preference-style HF benchmarks ─────────────────
    print("=" * 60)
    print(" Preference-style HF benchmarks")
    print("=" * 60)

    jfleg_results = base.eval_jfleg_preference(
        device=device, model=scorer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in jfleg_results.items():
        all_results.update(base._flatten_preference_metrics(bench_name, metrics))

    multiblimp_metrics = base.eval_multiblimp_english_preference(
        device=device, model=scorer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        base._flatten_preference_metrics("MultiBLiMP_eng_minimal_pair_preference", multiblimp_metrics)
    )

    storycloze_results = base.eval_story_cloze_preference(
        device=device, model=scorer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in storycloze_results.items():
        all_results.update(base._flatten_preference_metrics(bench_name, metrics))

    # ───────────────── Scalar human-score benchmarks ─────────────────
    print()
    print("=" * 60)
    print(" Scalar human-score benchmarks")
    print("=" * 60)

    def score_texts(texts: List[str]) -> np.ndarray:
        return bartscore_text_quality_scores(
            scorer=scorer,
            texts=texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            src_mode=args.src_mode,
        )

    # SummEval
    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]
    preds = score_texts(summeval_texts)
    all_results.update(base.correlation_bundle([x for y in ds["fluency"] for x in y], preds, "summeval_fluency"))
    all_results.update(base.correlation_bundle([x for y in ds["coherence"] for x in y], preds, "summeval_coherence"))
    all_results.update(base.correlation_bundle([x for y in ds["consistency"] for x in y], preds, "summeval_consistency"))

    # ELLIPSE
    ellipse_ds = base.load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    preds = score_texts(ellipse_ds["text"])
    all_results.update(base.correlation_bundle(ellipse_ds["overall"], preds, "ellipse_overall"))
    all_results.update(base.correlation_bundle(ellipse_ds["cohesion"], preds, "ellipse_cohesion"))

    # USR TopicalChat
    tc_texts, tc_overall = base.load_data_usr("data/benchmarks/tc_usr_data.json", "Overall")
    preds = score_texts(tc_texts)
    all_results.update(base.correlation_bundle(tc_overall, preds, "tc_overall"))
    _, tc_natural = base.load_data_usr("data/benchmarks/tc_usr_data.json", "Natural")
    all_results.update(base.correlation_bundle(tc_natural, preds, "tc_natural"))

    # USR PersonaChat
    pc_texts, pc_overall = base.load_data_usr("data/benchmarks/pc_usr_data.json", "Overall")
    preds = score_texts(pc_texts)
    all_results.update(base.correlation_bundle(pc_overall, preds, "pc_overall"))
    _, pc_natural = base.load_data_usr("data/benchmarks/pc_usr_data.json", "Natural")
    all_results.update(base.correlation_bundle(pc_natural, preds, "pc_natural"))

    # OpenMEVA
    t1, y1 = base.load_data_openmeva("data/benchmarks/mans_roc.json")
    t2, y2 = base.load_data_openmeva("data/benchmarks/mans_wp.json")
    preds = score_texts(t1 + t2)
    all_results.update(base.correlation_bundle(y1 + y2, preds, "OpenMEVA_overall"))

    # WebNLG
    webnlg_texts, webnlg_labels = base.load_data_webnlg("data/benchmarks/web_nlg_2020_human_evals_en.json")
    preds = score_texts(webnlg_texts)
    all_results.update(base.correlation_bundle(webnlg_labels, preds, "WebNLG_fluency"))

    # HANNA
    hanna_ds = base.load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    preds = score_texts(hanna_ds["text"])
    all_results.update(base.correlation_bundle(hanna_ds["coherence"], preds, "HANNA_coherence"))
    all_results.update(base.correlation_bundle(hanna_ds["complexity"], preds, "HANNA_complexity"))

    # ARG-ESSAY
    arge_ds = base.load_argessay_data("data/benchmarks/arg-essay.csv")
    preds = score_texts(arge_ds["text"])
    all_results.update(base.correlation_bundle(arge_ds["language_mastery"], preds, "ARG-ESSAY_language_mastery"))
    all_results.update(base.correlation_bundle(arge_ds["complexity"], preds, "ARG-ESSAY_complexity"))
    all_results.update(base.correlation_bundle(arge_ds["vocabulary"], preds, "ARG-ESSAY_vocabulary"))
    all_results.update(base.correlation_bundle(arge_ds["language_constructs"], preds, "ARG-ESSAY_language_constructs"))

    # Human Ratings of NLG
    hr_ds = base.load_human_ratings_of_nlg_data("data/benchmarks/human_ratings_of_nlg.csv")
    preds = score_texts(hr_ds["text"])
    all_results.update(base.correlation_bundle(hr_ds["quality"], preds, "HumanRatings_quality"))
    all_results.update(base.correlation_bundle(hr_ds["naturalness"], preds, "HumanRatings_naturalness"))

    # FED
    turn_ds, whole_ds = base.load_fed_data("data/benchmarks/fed_data.json")
    preds_turn = score_texts(turn_ds["text"])
    preds_whole = score_texts(whole_ds["text"])
    all_results.update(base.correlation_bundle(turn_ds["fluent"], preds_turn, "FED_turn_fluency"))
    all_results.update(base.correlation_bundle(turn_ds["overall"], preds_turn, "FED_turn_overall"))
    all_results.update(base.correlation_bundle(whole_ds["overall"], preds_whole, "FED_whole_overall"))

    # E2E
    e2e_ds = base.load_e2e_data("data/benchmarks/E2E_data")
    preds = score_texts(e2e_ds["text"])
    all_results.update(base.correlation_bundle(e2e_ds["naturalness"], preds, "E2E_naturalness"))
    all_results.update(base.correlation_bundle(e2e_ds["quality"], preds, "E2E_quality"))

    # Write JSONL (same format/path)
    model_dir_str = f"BARTScore(checkpoint={args.checkpoint},weights={args.bartscore_weights},src_mode={args.src_mode})"
    base.write_results_jsonl(
        model_name=args.model_name,
        training_dataset=args.training_dataset,
        perturbation_type=args.perturbation_type,
        num_layers=args.num_layers,
        context_length=args.context_length,
        model_dir=model_dir_str,
        results=all_results,
    )


if __name__ == "__main__":
    main()