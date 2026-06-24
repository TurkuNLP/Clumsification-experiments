"""
Evaluation script for LTR (Learning-to-Rank) quality-estimation models.

Usage example
─────────────
  python eval_benchmarks.py \
      --model-dir /path/to/output_dir/final \
      --model-name QE0.6B \
      --training-dataset "wiki-synth-v3" \
      --perturbation-type "fluency_only" \
      --num-layers 24 \
      --context-length 512 \
      --batch-size 32 \
      --max-length 512 \
      --dtype bfloat16 \
      --attn-implementation flash_attention_2

Results are appended as one JSON-Lines record to
  data/evals/<model_name>.jsonl
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────
#  Model components
# ──────────────────────────────────────────────────────────────────────


class ScoringHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LTRInferenceModel(nn.Module):
    def __init__(
        self,
        model_dir: str,
        head_path: str,
        device: torch.device,
        attn_implementation: str = "flash_attention_2",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Use fp32 by default for stable benchmark correlations
        if dtype is None:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        self.encoder = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.encoder.config.pad_token_id is None:
            self.encoder.config.pad_token_id = self.tokenizer.pad_token_id

        head_state = torch.load(head_path, map_location="cpu", weights_only=False)
        hidden_dim = head_state["hidden_dim"]
        dropout = head_state["dropout"]
        scorer_state = head_state["scorer"]

        # Make compatible with possible saved key prefixes
        scorer_state = {
            (k[len("scorer.") :] if k.startswith("scorer.") else k): v
            for k, v in scorer_state.items()
        }

        emb_dim = self.encoder.config.hidden_size
        self.scorer = ScoringHead(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.scorer.load_state_dict(scorer_state, strict=True)

        self.encoder.to(device=device)
        self.scorer.to(device=device, dtype=torch.float32)  # stable head dtype

        self.encoder.eval()
        self.scorer.eval()

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        device: torch.device,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        all_scores = []

        scorer_dtype = next(self.scorer.parameters()).dtype

        for start in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
            batch_texts = texts[start : start + batch_size]

            tok = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                pad_to_multiple_of=8,
            ).to(device)

            out = self.encoder(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                use_cache=False,
            )

            if not torch.isfinite(out.last_hidden_state).all():
                raise RuntimeError(
                    f"Non-finite encoder hidden states detected in batch starting at {start}."
                )

            emb = self.mean_pool(out.last_hidden_state, tok["attention_mask"])

            if not torch.isfinite(emb).all():
                raise RuntimeError(
                    f"Non-finite pooled embeddings detected in batch starting at {start}."
                )

            emb = emb.to(dtype=scorer_dtype)

            scores = self.scorer(emb)

            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"Non-finite scores detected in batch starting at {start}. "
                    f"Texts: {batch_texts[:3]}"
                )

            all_scores.append(scores.detach().cpu().float())

        return torch.cat(all_scores, dim=0).numpy()


# ──────────────────────────────────────────────────────────────────────
#  Correlation helpers
# ──────────────────────────────────────────────────────────────────────


def safe_spearman(labels, preds, name: str = "metric"):
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    if len(labels) < 2:
        print(f"  {name}: not enough valid points for Spearman.")
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"  {name}: predictions are constant; Spearman undefined.")
        return float("nan"), float("nan")

    rho, p = spearmanr(labels, preds)
    return float(rho), float(p)


def safe_kendall(labels, preds, name: str = "metric"):
    """Kendall τ with the same safety guards as safe_spearman."""
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    if len(labels) < 2:
        print(f"  {name}: not enough valid points for Kendall τ.")
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"  {name}: predictions are constant; Kendall τ undefined.")
        return float("nan"), float("nan")

    tau, p = kendalltau(labels, preds)
    return float(tau), float(p)


def correlation_bundle(labels, preds, name: str = "metric"):
    """
    Return a dict with Spearman ρ and Kendall τ for a scalar benchmark.
    This is the single call-site used throughout main() so that every scalar benchmark gets all metrics consistently.
    """
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    mask = np.isfinite(labels) & np.isfinite(preds)
    labels = labels[mask]
    preds = preds[mask]

    rho, rho_p = safe_spearman(labels, preds, name=name)
    tau, tau_p = safe_kendall(labels, preds, name=name)

    print(f"  Spearman ρ ({name}): {rho:.4f}  (p={rho_p:.2e})")
    print(f"  Kendall  τ ({name}): {tau:.4f}  (p={tau_p:.2e})")

    return {
        f"{name}_spearman_rho": rho,
        f"{name}_spearman_p": rho_p,
        f"{name}_kendall_tau": tau,
        f"{name}_kendall_p": tau_p,
    }


# ──────────────────────────────────────────────────────────────────────
#  Preference / pairwise helpers
# ──────────────────────────────────────────────────────────────────────


def _clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def join_context_and_continuation(context: str, continuation: str) -> str:
    """
    Join context + continuation
    """
    context = _clean_text(context)
    continuation = _clean_text(continuation)

    if not context:
        return continuation
    if not continuation:
        return context

    if continuation[0] in {".", ",", "!", "?", ";", ":", "'", '"', ")", "]"}:
        return context + continuation

    return context + " " + continuation


def preference_metrics(
    preferred_scores,
    dispreferred_scores,
    name: str = "preference",
):
    """
    Metrics for pairwise preference benchmarks.
    """
    preferred_scores = np.asarray(preferred_scores, dtype=np.float64)
    dispreferred_scores = np.asarray(dispreferred_scores, dtype=np.float64)

    if preferred_scores.shape != dispreferred_scores.shape:
        raise ValueError(
            f"{name}: preferred/dispreferred score shape mismatch: "
            f"{preferred_scores.shape} vs {dispreferred_scores.shape}"
        )

    mask = np.isfinite(preferred_scores) & np.isfinite(dispreferred_scores)
    preferred_scores = preferred_scores[mask]
    dispreferred_scores = dispreferred_scores[mask]

    if len(preferred_scores) == 0:
        return {
            "n": 0,
            "tie_aware_acc": float("nan"),
            "strict_acc": float("nan"),
            "tie_rate": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
        }

    deltas = preferred_scores - dispreferred_scores

    wins = deltas > 0
    ties = deltas == 0

    return {
        "n": int(len(deltas)),
        "tie_aware_acc": float(
            np.mean(wins.astype(np.float64) + 0.5 * ties.astype(np.float64))
        ),
        "strict_acc": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }


def multiple_choice_preference_metrics(
    scores_by_item,
    labels,
    name: str = "multiple_choice_preference",
):
    """
    Metrics for k-way preference benchmarks
    """
    scores_by_item = np.asarray(scores_by_item, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if scores_by_item.ndim != 2:
        raise ValueError(
            f"{name}: scores_by_item must be 2D, got shape {scores_by_item.shape}"
        )

    n, k = scores_by_item.shape

    if labels.shape[0] != n:
        raise ValueError(
            f"{name}: labels length mismatch: {labels.shape[0]} labels for {n} score rows"
        )

    valid = (
        np.all(np.isfinite(scores_by_item), axis=1)
        & np.isfinite(labels)
        & (labels >= 0)
        & (labels < k)
    )

    scores_by_item = scores_by_item[valid]
    labels = labels[valid]

    if len(labels) == 0:
        return {
            "n": 0,
            "num_choices": int(k),
            "tie_aware_acc": float("nan"),
            "strict_acc": float("nan"),
            "argmax_acc": float("nan"),
            "pairwise_acc": float("nan"),
            "tie_for_best_rate": float("nan"),
            "mean_margin_vs_best_wrong": float("nan"),
            "median_margin_vs_best_wrong": float("nan"),
        }

    n = len(labels)
    row_ids = np.arange(n)

    correct_scores = scores_by_item[row_ids, labels]
    best_scores = np.max(scores_by_item, axis=1)

    correct_is_best = correct_scores == best_scores
    num_tied_for_best = np.sum(scores_by_item == best_scores[:, None], axis=1)

    tie_aware_credit = np.where(
        correct_is_best,
        1.0 / num_tied_for_best,
        0.0,
    )

    margins_vs_best_wrong = []
    pairwise_credits = []

    for i in range(n):
        wrong = np.delete(scores_by_item[i], labels[i])

        best_wrong = np.max(wrong)
        margins_vs_best_wrong.append(correct_scores[i] - best_wrong)

        deltas = correct_scores[i] - wrong
        pairwise_credits.extend(
            (deltas > 0).astype(np.float64) + 0.5 * (deltas == 0).astype(np.float64)
        )

    margins_vs_best_wrong = np.asarray(margins_vs_best_wrong, dtype=np.float64)
    pairwise_credits = np.asarray(pairwise_credits, dtype=np.float64)

    argmax_preds = np.argmax(scores_by_item, axis=1)

    return {
        "n": int(n),
        "num_choices": int(k),
        "tie_aware_acc": float(np.mean(tie_aware_credit)),
        "strict_acc": float(np.mean(margins_vs_best_wrong > 0)),
        "argmax_acc": float(np.mean(argmax_preds == labels)),
        "pairwise_acc": float(np.mean(pairwise_credits)),
        "tie_for_best_rate": float(np.mean(num_tied_for_best > 1)),
        "mean_margin_vs_best_wrong": float(np.mean(margins_vs_best_wrong)),
        "median_margin_vs_best_wrong": float(np.median(margins_vs_best_wrong)),
    }


def print_preference_metrics(name: str, metrics: dict):
    print(
        f"  {name}: "
        f"n={metrics['n']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"tie rate={metrics['tie_rate']:.4f} | "
        f"mean Δ={metrics['mean_delta']:.4f} | "
        f"median Δ={metrics['median_delta']:.4f}"
    )


def print_mc_preference_metrics(name: str, metrics: dict):
    print(
        f"  {name}: "
        f"n={metrics['n']} | "
        f"k={metrics['num_choices']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"argmax acc={metrics['argmax_acc']:.4f} | "
        f"pairwise acc={metrics['pairwise_acc']:.4f} | "
        f"tie-for-best rate={metrics['tie_for_best_rate']:.4f} | "
        f"mean margin-vs-best-wrong={metrics['mean_margin_vs_best_wrong']:.4f} | "
        f"median margin-vs-best-wrong={metrics['median_margin_vs_best_wrong']:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Preference-style evaluators (JFLEG, MultiBLiMP, StoryCloze)
# ──────────────────────────────────────────────────────────────────────


def eval_pairwise_preference_dataset(
    name: str,
    preferred_texts: List[str],
    dispreferred_texts: List[str],
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
):
    if len(preferred_texts) != len(dispreferred_texts):
        raise ValueError(
            f"{name}: preferred/dispreferred length mismatch: "
            f"{len(preferred_texts)} vs {len(dispreferred_texts)}"
        )

    preferred_texts = [_clean_text(x) for x in preferred_texts]
    dispreferred_texts = [_clean_text(x) for x in dispreferred_texts]

    keep = [
        i
        for i, (p, d) in enumerate(zip(preferred_texts, dispreferred_texts))
        if p and d
    ]

    preferred_texts = [preferred_texts[i] for i in keep]
    dispreferred_texts = [dispreferred_texts[i] for i in keep]

    n = len(preferred_texts)
    print(f"  Total preference pairs in {name}: {n}")

    if n == 0:
        print(f"  {name}: no valid preference pairs.")
        return None

    all_texts = preferred_texts + dispreferred_texts

    raw_scores = getModelPreds(
        device,
        model,
        all_texts,
        batch_size=batch_size,
        max_length=max_length,
    )

    preferred_scores = raw_scores[:n]
    dispreferred_scores = raw_scores[n:]

    metrics = preference_metrics(
        preferred_scores=preferred_scores,
        dispreferred_scores=dispreferred_scores,
        name=name,
    )
    print_preference_metrics(name, metrics)
    return metrics


def load_jfleg_preference_pairs(split: str = "test"):
    ds = datasets.load_dataset("jhu-clsp/jfleg", split=split, download_mode="force_redownload",)

    preferred = []
    dispreferred = []

    for ex in ds:
        src = _clean_text(ex["sentence"])
        corrections = ex["corrections"]

        if not src or corrections is None:
            continue

        for corr in corrections:
            corr = _clean_text(corr)
            if not corr or corr == src:
                continue

            preferred.append(corr)
            dispreferred.append(src)

    return preferred, dispreferred


def eval_jfleg_preference(device, model, batch_size: int = 32, max_length: int = 512):
    results = {}
    for split in ["validation", "test"]:
        preferred, dispreferred = load_jfleg_preference_pairs(split=split)
        name = f"JFLEG_{split}_correction_preference"
        results[name] = eval_pairwise_preference_dataset(
            name=name,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            device=device,
            model=model,
            batch_size=batch_size,
            max_length=max_length,
        )
    return results


def load_multiblimp_english_preference_pairs():
    ds = datasets.load_dataset("jumelet/multiblimp", "eng", split="train")

    preferred = []
    dispreferred = []

    for ex in ds:
        good = _clean_text(ex["sen"])
        bad = _clean_text(ex["wrong_sen"])
        if not good or not bad:
            continue
        preferred.append(good)
        dispreferred.append(bad)

    return preferred, dispreferred


def eval_multiblimp_english_preference(
    device, model, batch_size: int = 32, max_length: int = 512
):
    preferred, dispreferred = load_multiblimp_english_preference_pairs()
    return eval_pairwise_preference_dataset(
        name="MultiBLiMP_eng_minimal_pair_preference",
        preferred_texts=preferred,
        dispreferred_texts=dispreferred,
        device=device,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
    )


def load_story_cloze_preference_pairs(split: str = "eval"):
    ds = datasets.load_dataset("lecslab/story_cloze", split=split)

    preferred = []
    dispreferred = []

    for ex in ds:
        prompt = _clean_text(ex["prompt"])
        chosen = _clean_text(ex["chosen"])
        rejected = _clean_text(ex["rejected"])

        if not prompt or not chosen or not rejected:
            continue

        preferred.append(join_context_and_continuation(prompt, chosen))
        dispreferred.append(join_context_and_continuation(prompt, rejected))

    return preferred, dispreferred


def eval_story_cloze_preference(
    device, model, batch_size: int = 32, max_length: int = 512
):
    results = {}
    for split in ["eval", "test"]:
        preferred, dispreferred = load_story_cloze_preference_pairs(split=split)
        name = f"StoryCloze_{split}_ending_preference"
        results[name] = eval_pairwise_preference_dataset(
            name=name,
            preferred_texts=preferred,
            dispreferred_texts=dispreferred,
            device=device,
            model=model,
            batch_size=batch_size,
            max_length=max_length,
        )
    return results


# ──────────────────────────────────────────────────────────────────────
#  Model loading / inference helpers
# ──────────────────────────────────────────────────────────────────────


def load_model_for_benchmark(
    model_dir: str,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
    dtype: Optional[torch.dtype] = None,
):
    head_path = os.path.join(model_dir, "ltr_head.pt")

    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"Could not find ranking head at {head_path}. "
            f"Make sure you pass the trainer final directory, e.g. output_dir/final"
        )

    model = LTRInferenceModel(
        model_dir=model_dir,
        head_path=head_path,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )

    return model


def getModelPreds(
    device,
    model,
    test_texts,
    batch_size: int = 32,
    max_length: int = 512,
):
    return model.score_texts(
        texts=test_texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


# ──────────────────────────────────────────────────────────────────────
#  WebNLG helper
# ──────────────────────────────────────────────────────────────────────


def collect_webnlg_texts(
    records: List[Dict[str, Any]],
    base_dir: str = "data/benchmarks/rdf2text/en",
) -> List[str]:
    base = Path(base_dir)
    texts: List[str] = []
    file_cache: Dict[str, List[str]] = {}

    for rec in records:
        submission_id = str(rec["submission_id"])

        if not os.path.exists(base / submission_id):
            continue

        # BUG-FIX: comment said "0-based" but subtracted 1 → clarified as 1-based
        line_idx = int(rec["sample_id"]) - 1  # sample_id is 1-based

        if submission_id not in file_cache:
            file_path = base / submission_id / "primary.en"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file: {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                file_cache[submission_id] = f.readlines()

        lines = file_cache[submission_id]

        if line_idx < 0 or line_idx >= len(lines):
            file_path = base / submission_id / "primary.en"
            raise IndexError(
                f"sample_id {rec['sample_id']} → line_idx {line_idx} out of range "
                f"for {file_path} (0..{len(lines)-1})"
            )

        texts.append(lines[line_idx].rstrip("\n"))

    return texts


# ──────────────────────────────────────────────────────────────────────
#  Benchmark data loaders
# ──────────────────────────────────────────────────────────────────────


def load_e2e_data(folder_path: str):
    import pandas as pd

    nat_df = pd.read_csv(os.path.join(folder_path, "naturalness.csv"))
    qual_df = pd.read_csv(os.path.join(folder_path, "quality.csv"))

    ref_cols = ["ref1", "ref2", "ref3", "ref4", "ref5"]
    nat_cols = ["natur1", "natur2", "natur3", "natur4", "natur5"]
    qual_cols = ["quality1", "quality2", "quality3", "quality4", "quality5"]

    texts = []
    naturalness_scores = []
    quality_scores = []

    num_groups = len(nat_df) // 3

    for g in range(num_groups):
        start = g * 3
        end = start + 3

        nat_group = nat_df.iloc[start:end]
        qual_group = qual_df.iloc[start:end]

        for ref_col, nat_col, qual_col in zip(ref_cols, nat_cols, qual_cols):
            texts.append(nat_group[ref_col].iloc[0])
            naturalness_scores.append(nat_group[nat_col].astype(float).mean())
            quality_scores.append(qual_group[qual_col].astype(float).mean())

    return datasets.Dataset.from_dict(
        {
            "text": texts,
            "naturalness": naturalness_scores,
            "quality": quality_scores,
        }
    )


def load_fed_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        test_data = json.loads(reader.read().strip())

    turn_dial = []
    whole_dial = []
    for x in test_data:
        if x.get("response", None):
            turn_dial.append(
                {
                    "text": x["response"][7:],
                    "fluent": np.mean(
                        [
                            int(y)
                            for y in x["annotations"]["Fluent"]
                            if isinstance(y, int)
                        ]
                    ),
                    "overall": np.mean(
                        [
                            int(y)
                            for y in x["annotations"]["Overall"]
                            if isinstance(y, int)
                        ]
                    ),
                }
            )
        else:
            whole_dial.append(
                {
                    "text": x["context"],
                    "overall": np.mean(
                        [
                            int(y)
                            for y in x["annotations"]["Overall"]
                            if isinstance(y, int)
                        ]
                    ),
                }
            )
    return datasets.Dataset.from_list(turn_dial), datasets.Dataset.from_list(
        whole_dial
    )


def load_human_ratings_of_nlg_data(file_path: str):
    with open(file_path, newline="\n") as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        headers = next(reader)
        head_id_dict = {headers[i]: i for i in range(len(headers))}
        for row in reader:
            data.append(
                {
                    "text": row[head_id_dict["sys_ref"]],
                    "quality": row[head_id_dict["quality"]],
                    "naturalness": row[head_id_dict["naturalness"]],
                }
            )
    return datasets.Dataset.from_list(data)


def load_argessay_data(file_path: str):
    with open(file_path, newline="\n") as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        header = next(reader)
        head_id_dict = {header[i]: i for i in range(len(header))}
        for row in reader:
            # Human text
            data.append(
                {
                    "text": row[head_id_dict["Student"]],
                    "language_mastery": float(row[head_id_dict["STUD_LangMastery"]]),
                    "complexity": float(row[head_id_dict["STUD_Complexity"]]),
                    "vocabulary": float(row[head_id_dict["STUD_Vocab"]]),
                    "language_constructs": float(
                        row[head_id_dict["STUD_LangConstructs"]]
                    ),
                }
            )
            # GPT3 text
            data.append(
                {
                    "text": row[head_id_dict["ChatGPT-3"]],
                    "language_mastery": float(row[head_id_dict["GPT3_LangMastery"]]),
                    "complexity": float(row[head_id_dict["GPT3_Complexity"]]),
                    "vocabulary": float(row[head_id_dict["GPT3_Vocab"]]),
                    "language_constructs": float(
                        row[head_id_dict["GPT3_LangConstructs"]]
                    ),
                }
            )
            # GPT4 text
            data.append(
                {
                    "text": row[head_id_dict["ChatGPT-4"]],
                    "language_mastery": float(row[head_id_dict["GPT4_LangMastery"]]),
                    "complexity": float(row[head_id_dict["GPT4_Complexity"]]),
                    "vocabulary": float(row[head_id_dict["GPT4_Vocab"]]),
                    "language_constructs": float(
                        row[head_id_dict["GPT4_LangConstructs"]]
                    ),
                }
            )
    return datasets.Dataset.from_list(data)


def load_hanna_data(file_path: str):
    with open(file_path, newline="\n") as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers

        current_id = None  # BUG-FIX: was 0, which would skip the first story
        coh = []
        comp = []
        story = ""

        for row in reader:
            story_id = int(row[0])
            if current_id is not None and story_id != current_id:
                data.append(
                    {
                        "text": story,
                        "coherence": float(np.mean(coh)),
                        "complexity": float(np.mean(comp)),
                    }
                )
                coh = []
                comp = []
            current_id = story_id
            story = row[3]
            coh.append(int(row[6]))
            comp.append(int(row[10]))

        # BUG-FIX: flush the last story group (was silently dropped)
        if current_id is not None and coh:
            data.append(
                {
                    "text": story,
                    "coherence": float(np.mean(coh)),
                    "complexity": float(np.mean(comp)),
                }
            )

    return datasets.Dataset.from_list(data)


def load_data_webnlg(file_path: str):
    # BUG-FIX: was ignoring the `file_path` argument and hardcoding the path
    with open(file_path) as reader:
        data = json.loads(reader.read().strip())
    texts = collect_webnlg_texts(data)
    labels = [
        x["Fluency"]
        for x in data
        if os.path.exists("data/benchmarks/rdf2text/en/" + x["submission_id"])
    ]
    return texts, labels


def load_data_openmeva(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        data = json.loads(reader.read().strip())

    texts = [
        data[str(y)]["gen"][x]["text"]
        for y in list(data.keys())
        for x in list(data[str(y)]["gen"].keys())
    ]
    labels = [
        float(np.mean(data[str(y)]["gen"][x]["score"]))
        for y in list(data.keys())
        for x in list(data[str(y)]["gen"].keys())
    ]

    return texts, labels


def load_data_usr(file_path: str, label_dimension: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        its = json.loads(reader.read())
    texts = [y["response"].replace("\n", "") for x in its for y in x["responses"]]
    labels = [
        float(np.mean(y[label_dimension])) for x in its for y in x["responses"]
    ]
    return texts, labels


def load_data_ellipse(file_path: str):
    data_set = []
    with open(file_path, newline="\n") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(spamreader, None)  # Skip header
        for row in spamreader:
            text = row[1]
            oa = row[18]
            cohesion = row[19]
            syntax = row[20]
            vocab = row[21]
            grammar = row[23]
            data_set.append(
                {
                    "text": text,
                    "overall": oa,
                    "cohesion": cohesion,
                    "syntax": syntax,
                    "vocab": vocab,
                    "grammar": grammar,
                }
            )
    return datasets.Dataset.from_list(data_set)


def load_test_data_cohesentia(
    file_paths: Union[str, List[str]],
) -> Tuple[List[str], List[float]]:
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    test_texts = []
    test_labels = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = data.values()
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                f"Unexpected top-level JSON type in {path}: {type(data)}"
            )

        for entry in entries:
            test_texts.append(entry["Text"])
            test_labels.append(entry["HolisticData"]["consensus_score"])

    return test_texts, test_labels


# ──────────────────────────────────────────────────────────────────────
#  JSONL results writer
# ──────────────────────────────────────────────────────────────────────

EVAL_DIR = Path("data/evals")


def _flatten_preference_metrics(
    name: str, metrics: Optional[dict]
) -> Dict[str, Any]:
    """Prefix every key in a preference-metric dict with the benchmark name."""
    if metrics is None:
        return {}
    return {f"{name}_{k}": v for k, v in metrics.items()}


def write_results_jsonl(
    model_name: str,
    training_dataset: str,
    perturbation_type: str,
    num_layers: int,
    context_length: int,
    model_dir: str,
    results: Dict[str, Any],
) -> Path:
    """
    Append one JSON-Lines record to  data/evals/<model_name>.jsonl

    The record contains the user-supplied metadata fields followed by
    every benchmark result (one field per metric).
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{model_name}.jsonl"

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_dir": model_dir,
        "training_dataset": training_dataset,
        "perturbation_type": perturbation_type,
        "num_layers": num_layers,
        "context_length": context_length,
    }
    record.update(results)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ Results appended to {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────
#  Argument parser
# ──────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LTR quality-estimation benchmarks and log results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / checkpoint ──────────────────────────────────────────
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the final trainer directory (must contain ltr_head.pt).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Short architecture tag used as the JSONL filename (e.g. QE0.6B, E5, QE4B).",
    )

    # ── Metadata written to JSONL ───────────────────────────────────
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Maximum context length the model was trained with.",
    )
    # ── Inference settings ──────────────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for scoring.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length passed to the tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(_DTYPE_MAP.keys()),
        help="Torch dtype for the encoder.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation passed to AutoModel.from_pretrained.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPE_MAP[args.dtype]

    print(f"Device : {device}")
    print(f"Model  : {args.model_dir}")
    print(f"Arch   : {args.model_name}")
    print(f"dtype  : {dtype}")
    print()

    model = load_model_for_benchmark(
        model_dir=args.model_dir,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    # Quick sanity check
    probe = ["short text", "a much much longer text with different content"]
    probe_scores = model.score_texts(probe, device=device, batch_size=2, max_length=64)
    print("Sanity probe scores:", probe_scores)
    print()

    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length

    # Accumulate every metric into this dict → flushed to JSONL at the end.
    all_results: Dict[str, Any] = {}

    # ==================================================================
    # Preference-style HF benchmarks
    # ==================================================================

    print("=" * 60)
    print(" Preference-style HF benchmarks")
    print("=" * 60)

    # ── JFLEG ─────────────────────────────────────────────────────
    jfleg_results = eval_jfleg_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in jfleg_results.items():
        all_results.update(_flatten_preference_metrics(bench_name, metrics))

    # ── MultiBLiMP ────────────────────────────────────────────────
    multiblimp_metrics = eval_multiblimp_english_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        _flatten_preference_metrics(
            "MultiBLiMP_eng_minimal_pair_preference", multiblimp_metrics
        )
    )

    # ── Story Cloze ───────────────────────────────────────────────
    storycloze_results = eval_story_cloze_preference(
        device=device, model=model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    for bench_name, metrics in storycloze_results.items():
        all_results.update(_flatten_preference_metrics(bench_name, metrics))

    # ==================================================================
    # Scalar human-score benchmarks
    # ==================================================================

    print()
    print("=" * 60)
    print(" Scalar human-score benchmarks")
    print("=" * 60)

    # ── SummEval ──────────────────────────────────────────────────
    ds = datasets.load_dataset("mteb/summeval")["test"]
    summeval_texts = [x for y in ds["machine_summaries"] for x in y]
    print(f"\nSummEval texts: {len(summeval_texts)}")
    raw_preds = getModelPreds(
        device, model, summeval_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    summeval_fluency_labels = [x for y in ds["fluency"] for x in y]
    all_results.update(
        correlation_bundle(summeval_fluency_labels, raw_preds, "summeval_fluency")
    )

    summeval_coherence_labels = [x for y in ds["coherence"] for x in y]
    all_results.update(
        correlation_bundle(summeval_coherence_labels, raw_preds, "summeval_coherence")
    )

    summeval_consistency_labels = [x for y in ds["consistency"] for x in y]
    all_results.update(
        correlation_bundle(
            summeval_consistency_labels, raw_preds, "summeval_consistency"
        )
    )

    # ── ELLIPSE ───────────────────────────────────────────────────
    ellipse_ds = load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    ellipse_texts = ellipse_ds["text"]
    print(f"\nELLIPSE texts: {len(ellipse_texts)}")
    raw_preds = getModelPreds(
        device, model, ellipse_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    all_results.update(
        correlation_bundle(ellipse_ds["overall"], raw_preds, "ellipse_overall")
    )
    all_results.update(
        correlation_bundle(ellipse_ds["cohesion"], raw_preds, "ellipse_cohesion")
    )

    # ── USR – Topical Chat ────────────────────────────────────────
    tc_texts, tc_overall_labels = load_data_usr(
        "data/benchmarks/tc_usr_data.json", "Overall"
    )
    print(f"\nTopicalChat texts: {len(tc_texts)}")
    raw_preds = getModelPreds(
        device, model, tc_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(tc_overall_labels, raw_preds, "tc_overall")
    )

    _, tc_natural_labels = load_data_usr(
        "data/benchmarks/tc_usr_data.json", "Natural"
    )
    all_results.update(
        correlation_bundle(tc_natural_labels, raw_preds, "tc_natural")
    )

    # ── USR – Persona Chat ────────────────────────────────────────
    pc_texts, pc_overall_labels = load_data_usr(
        "data/benchmarks/pc_usr_data.json", "Overall"
    )
    print(f"\nPersonaChat texts: {len(pc_texts)}")
    raw_preds = getModelPreds(
        device, model, pc_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(pc_overall_labels, raw_preds, "pc_overall")
    )

    _, pc_natural_labels = load_data_usr(
        "data/benchmarks/pc_usr_data.json", "Natural"
    )
    all_results.update(
        correlation_bundle(pc_natural_labels, raw_preds, "pc_natural")
    )

    # ── OpenMEVA ──────────────────────────────────────────────────
    meva_texts_roc, meva_labels_roc = load_data_openmeva(
        "data/benchmarks/mans_roc.json"
    )
    meva_texts_wp, meva_labels_wp = load_data_openmeva(
        "data/benchmarks/mans_wp.json"
    )
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    print(f"\nOpenMEVA texts: {len(meva_texts)}")
    raw_preds = getModelPreds(
        device, model, meva_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(meva_labels, raw_preds, "OpenMEVA_overall")
    )

    # ── WebNLG ────────────────────────────────────────────────────
    webnlg_texts, webnlg_labels = load_data_webnlg(
        "data/benchmarks/web_nlg_2020_human_evals_en.json"
    )
    print(f"\nWebNLG texts: {len(webnlg_texts)}")
    raw_preds = getModelPreds(
        device, model, webnlg_texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(webnlg_labels, raw_preds, "WebNLG_fluency")
    )

    # ── HANNA ─────────────────────────────────────────────────────
    hanna_ds = load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    print(f"\nHANNA texts: {len(hanna_ds)}")
    raw_preds = getModelPreds(
        device, model, hanna_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(hanna_ds["coherence"], raw_preds, "HANNA_coherence")
    )
    all_results.update(
        correlation_bundle(hanna_ds["complexity"], raw_preds, "HANNA_complexity")
    )

    # ── ARG-ESSAY ─────────────────────────────────────────────────
    arge_ds = load_argessay_data("data/benchmarks/arg-essay.csv")
    print(f"\nARG-ESSAY texts: {len(arge_ds)}")
    raw_preds = getModelPreds(
        device, model, arge_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(
            arge_ds["language_mastery"], raw_preds, "ARG-ESSAY_language_mastery"
        )
    )
    all_results.update(
        correlation_bundle(arge_ds["complexity"], raw_preds, "ARG-ESSAY_complexity")
    )
    all_results.update(
        correlation_bundle(arge_ds["vocabulary"], raw_preds, "ARG-ESSAY_vocabulary")
    )
    all_results.update(
        correlation_bundle(
            arge_ds["language_constructs"],
            raw_preds,
            "ARG-ESSAY_language_constructs",
        )
    )

    # ── Human Ratings of NLG ──────────────────────────────────────
    hr_ds = load_human_ratings_of_nlg_data("data/benchmarks/human_ratings_of_nlg.csv")
    print(f"\nHumanRatings texts: {len(hr_ds)}")
    raw_preds = getModelPreds(
        device, model, hr_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(hr_ds["quality"], raw_preds, "HumanRatings_quality")
    )
    all_results.update(
        correlation_bundle(
            hr_ds["naturalness"], raw_preds, "HumanRatings_naturalness"
        )
    )

    # ── FED ───────────────────────────────────────────────────────
    turn_ds, whole_ds = load_fed_data("data/benchmarks/fed_data.json")
    print(f"\nFED turn-level texts: {len(turn_ds)}")
    raw_preds_turn = getModelPreds(
        device, model, turn_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    print(f"FED whole-dialogue texts: {len(whole_ds)}")
    raw_preds_whole = getModelPreds(
        device, model, whole_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    all_results.update(
        correlation_bundle(turn_ds["fluent"], raw_preds_turn, "FED_turn_fluency")
    )
    all_results.update(
        correlation_bundle(turn_ds["overall"], raw_preds_turn, "FED_turn_overall")
    )
    all_results.update(
        correlation_bundle(
            whole_ds["overall"], raw_preds_whole, "FED_whole_overall"
        )
    )
    del raw_preds_turn, raw_preds_whole

    # ── E2E ───────────────────────────────────────────────────────
    """
    e2e_ds = load_e2e_data("data/benchmarks/E2E_data")
    print(f"\nE2E texts: {len(e2e_ds)}")
    raw_preds = getModelPreds(
        device, model, e2e_ds["text"], batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    all_results.update(
        correlation_bundle(e2e_ds["naturalness"], raw_preds, "E2E_naturalness")
    )
    all_results.update(
        correlation_bundle(e2e_ds["quality"], raw_preds, "E2E_quality")
    )
    """

    # ==================================================================
    # Write JSONL
    # ==================================================================

    #Parse information from the DS name
    def info_ds_parser(name:str):
        num_layers = 5
        pert_type = "clumsy"

        #Getting model name
        model_name=name[:name.find('_')]
        name=name[name.find('_')+1:]
        #Parsing the language info
        lan=name[:name.find('_')]
        name=name[name.find('_')+1:]
        #Parsing num_layers and pert_type
        training_ds_name=name
        if name[name.rfind('_')+1].isnumeric():
            pert_type = name[:name.rfind('_')]
            num_layers = name[name.rfind('_')+1:]
        else:
            pert_type = name
        return model_name, lan, pert_type, num_layers, training_ds_name
    
    model_name, language, pert_type, num_layers, training_ds_name = info_ds_parser(args.model_name)



    write_results_jsonl(
        model_name=model_name,
        training_dataset=language+"/"+training_ds_name,
        perturbation_type=pert_type,
        num_layers=num_layers,
        context_length=args.context_length,
        model_dir=args.model_dir,
        results=all_results,
    )


if __name__ == "__main__":
    main()