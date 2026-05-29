import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List
from tqdm.auto import tqdm
import datasets
import os
import json
from scipy.stats import spearmanr
import csv
import numpy as np
from typing import Union, List, Tuple
import sys


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
            torch_dtype=dtype,  # <-- FIXED
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.encoder.config.pad_token_id is None:
            self.encoder.config.pad_token_id = self.tokenizer.pad_token_id

        head_state = torch.load(head_path, map_location="cpu")
        hidden_dim = head_state["hidden_dim"]
        dropout = head_state["dropout"]
        scorer_state = head_state["scorer"]

        # Make compatible with possible saved key prefixes
        scorer_state = {
            (k[len("scorer."):] if k.startswith("scorer.") else k): v
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
            batch_texts = texts[start:start + batch_size]

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
    
def safe_spearman(labels, preds, name="metric"):
    labels = np.asarray(labels, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    mask = np.isfinite(labels) & np.isfinite(preds)
    labels = labels[mask]
    preds = preds[mask]

    if len(labels) < 2:
        print(f"{name}: not enough valid points for Spearman.")
        return float("nan"), float("nan")

    if np.all(preds == preds[0]):
        print(f"{name}: predictions are constant; Spearman undefined.")
        return float("nan"), float("nan")

    rho, p = spearmanr(labels, preds)
    return float(rho), float(p)


    ### Functions for using HellaSwag, Multiblimp, JFLEG, and StoryCloze as evals

def _clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def join_context_and_continuation(context: str, continuation: str) -> str:
    """
    Join context + continuation while preserving HellaSwag-style punctuation.
    Many HellaSwag endings start with ',' or '.', so blindly adding a space can
    produce unnatural text like 'then , the man...'.
    """
    context = _clean_text(context)
    continuation = _clean_text(continuation)

    if not context:
        return continuation
    if not continuation:
        return context

    # If the continuation starts with punctuation, attach directly.
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

    preferred_scores[i] should be greater than dispreferred_scores[i].

    Reports:
      - tie_aware_acc: counts preferred > rejected as 1, tie as 0.5
      - strict_acc: counts only preferred > rejected as correct
      - tie_rate
      - mean_delta: mean(preferred_score - dispreferred_score)
      - median_delta
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
        "tie_aware_acc": float(np.mean(wins.astype(np.float64) + 0.5 * ties.astype(np.float64))),
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
    Metrics for k-way preference benchmarks such as HellaSwag.

    scores_by_item: shape [num_items, num_choices]
    labels: integer index of preferred/correct choice for each item.

    Reports:
      - tie_aware_acc:
          If the correct option is tied for best with m options, gets 1/m.
          If it is not tied for best, gets 0.
      - strict_acc:
          Correct option must be strictly greater than every distractor.
      - argmax_acc:
          np.argmax accuracy, useful but biased toward lower-index choices on ties.
      - pairwise_acc:
          Average over all correct-vs-distractor comparisons, tie = 0.5.
      - mean_margin_vs_best_wrong:
          mean(correct_score - max_wrong_score)
    """
    scores_by_item = np.asarray(scores_by_item, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if scores_by_item.ndim != 2:
        raise ValueError(f"{name}: scores_by_item must be 2D, got shape {scores_by_item.shape}")

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

    # Strict accuracy: correct must be strictly greater than every wrong option.
    wrong_scores = []
    margins_vs_best_wrong = []
    pairwise_credits = []

    for i in range(n):
        wrong = np.delete(scores_by_item[i], labels[i])
        wrong_scores.append(wrong)

        best_wrong = np.max(wrong)
        margins_vs_best_wrong.append(correct_scores[i] - best_wrong)

        deltas = correct_scores[i] - wrong
        pairwise_credits.extend((deltas > 0).astype(np.float64) + 0.5 * (deltas == 0).astype(np.float64))

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
        f"{name}: "
        f"n={metrics['n']} | "
        f"tie-aware acc={metrics['tie_aware_acc']:.4f} | "
        f"strict acc={metrics['strict_acc']:.4f} | "
        f"tie rate={metrics['tie_rate']:.4f} | "
        f"mean Δ={metrics['mean_delta']:.4f} | "
        f"median Δ={metrics['median_delta']:.4f}"
    )


def print_mc_preference_metrics(name: str, metrics: dict):
    print(
        f"{name}: "
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


def eval_pairwise_preference_dataset(
    name: str,
    preferred_texts: List[str],
    dispreferred_texts: List[str],
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    Generic evaluator for datasets where each item is:
      preferred_text > dispreferred_text
    """
    if len(preferred_texts) != len(dispreferred_texts):
        raise ValueError(
            f"{name}: preferred/dispreferred length mismatch: "
            f"{len(preferred_texts)} vs {len(dispreferred_texts)}"
        )

    preferred_texts = [_clean_text(x) for x in preferred_texts]
    dispreferred_texts = [_clean_text(x) for x in dispreferred_texts]

    keep = [
        i for i, (p, d) in enumerate(zip(preferred_texts, dispreferred_texts))
        if p and d
    ]

    preferred_texts = [preferred_texts[i] for i in keep]
    dispreferred_texts = [dispreferred_texts[i] for i in keep]

    n = len(preferred_texts)
    print(f"Total number of preference pairs in {name}: {n}")

    if n == 0:
        print(f"{name}: no valid preference pairs.")
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
    """
    JFLEG:
      preferred = each human correction
      dispreferred = original learner sentence

    This creates one pair per non-empty, non-identical correction.
    """
    ds = datasets.load_dataset("jhu-clsp/jfleg", split=split)

    preferred = []
    dispreferred = []

    for ex in ds:
        src = _clean_text(ex["sentence"])
        corrections = ex["corrections"]

        if not src or corrections is None:
            continue

        for corr in corrections:
            corr = _clean_text(corr)

            # Skip empty corrections and exact duplicates.
            if not corr or corr == src:
                continue

            preferred.append(corr)
            dispreferred.append(src)

    return preferred, dispreferred


def eval_jfleg_preference(
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    Evaluate both JFLEG validation and test splits.
    """
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
    """
    MultiBLiMP English:
      preferred = grammatical sentence, field 'sen'
      dispreferred = minimal-pair corrupted sentence, field 'wrong_sen'

    English subset/config is 'eng'.
    """
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
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
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
    """
    Story Cloze:
      preferred = prompt + chosen ending
      dispreferred = prompt + rejected ending
    """
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
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    Evaluate both Story Cloze eval and test splits.
    """
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


def load_hellaswag_multiple_choice(split: str = "validation"):
    """
    HellaSwag:
      Each example has 4 endings.
      Score ctx + each ending.
      The correct label should receive the highest score.

    We use validation by default because it is the standard labeled eval split.
    """
    ds = datasets.load_dataset("Rowan/hellaswag", split=split)

    all_choice_texts = []
    labels = []

    for ex in ds:
        raw_label = ex.get("label", None)

        try:
            label = int(raw_label)
        except Exception:
            # Some splits/configurations may not have usable labels.
            continue

        endings = ex["endings"]

        if endings is None or len(endings) != 4:
            continue

        if label < 0 or label >= len(endings):
            continue

        ctx = _clean_text(ex.get("ctx", ""))

        # Fallback if ctx is missing for some reason.
        if not ctx:
            ctx = join_context_and_continuation(
                _clean_text(ex.get("ctx_a", "")),
                _clean_text(ex.get("ctx_b", "")),
            )

        if not ctx:
            continue

        for ending in endings:
            all_choice_texts.append(join_context_and_continuation(ctx, ending))

        labels.append(label)

    return all_choice_texts, labels, 4


def eval_hellaswag_preference(
    device,
    model,
    batch_size: int = 32,
    max_length: int = 512,
    split: str = "validation",
):
    all_choice_texts, labels, num_choices = load_hellaswag_multiple_choice(split=split)

    n_items = len(labels)
    print(f"Total number of HellaSwag {split} items: {n_items}")
    print(f"Total number of HellaSwag {split} scored continuations: {len(all_choice_texts)}")

    if n_items == 0:
        print(f"HellaSwag_{split}: no valid labeled examples.")
        return None

    raw_scores = getModelPreds(
        device,
        model,
        all_choice_texts,
        batch_size=batch_size,
        max_length=max_length,
    )

    scores_by_item = raw_scores.reshape(n_items, num_choices)

    metrics = multiple_choice_preference_metrics(
        scores_by_item=scores_by_item,
        labels=labels,
        name=f"HellaSwag_{split}",
    )

    print_mc_preference_metrics(f"HellaSwag_{split}_ending_preference", metrics)
    return metrics

# Loading of the model and doing predictions functions

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
        dtype=dtype
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

# Other helpers

from pathlib import Path
from typing import List, Dict, Any
import os


def collect_webnlg_texts(
    records: List[Dict[str, Any]],
    base_dir: str = "data/benchmarks/rdf2text/en",
) -> List[str]:
    """
    Given records with keys:
      - 'submission_id'
      - 'sample_id' (0-based line index in primary.en)

    Returns list of selected lines in the same order as `records`.
    """
    base = Path(base_dir)
    texts: List[str] = []

    # Cache lines per submission_id so each file is read once
    file_cache: Dict[str, List[str]] = {}

    for rec in records:
        submission_id = str(rec["submission_id"])

        if not os.path.exists(base / submission_id):
            continue

        line_idx = int(rec["sample_id"])-1  # assumes 0-based indexing

        # Load file once per submission_id
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
                f"sample_id {line_idx} out of range for {file_path} "
                f"(0..{len(lines)-1})"
            )

        texts.append(lines[line_idx].rstrip("\n"))

    return texts

# Helpers for loading a specific benchmark / dataset

#E2E generations

def load_e2e_data(folder_path: str):
    import os
    import pandas as pd
    from datasets import Dataset

    nat_df = pd.read_csv(os.path.join(folder_path, "naturalness.csv"))
    qual_df = pd.read_csv(os.path.join(folder_path, "quality.csv"))

    # Identify and sort columns to ensure ref1↔natur1↔quality1, etc.
    ref_cols = ['ref1', 'ref2', 'ref3', 'ref4', 'ref5']
    nat_cols = ['natur1', 'natur2', 'natur3', 'natur4', 'natur5']
    qual_cols = ['quality1', 'quality2', 'quality3', 'quality4', 'quality5']

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
            # Text is the same across the 3 annotator rows — take the first
            texts.append(nat_group[ref_col].iloc[0])
            # Mean of the 3 annotator scores
            naturalness_scores.append(nat_group[nat_col].astype(float).mean())
            quality_scores.append(qual_group[qual_col].astype(float).mean())

    return Dataset.from_dict({
        "text": texts,
        "naturalness": naturalness_scores,
        "quality": quality_scores,
    })

# FED benchmark
def load_fed_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as reader:
        test_data = json.loads(reader.read().strip())

    turn_dial = []
    whole_dial = []
    for x in test_data:
        if x.get('response', None):
            turn_dial.append({
                'text':x['response'][7:],
                'fluent':np.mean([int(y) for y in x['annotations']['Fluent'] if isinstance(y, int)]),
                'overall':np.mean([int(y) for y in x['annotations']['Overall'] if isinstance(y, int)])
            })
        else:
            whole_dial.append({
                'text':x['context'],
                'overall':np.mean([int(y) for y in x['annotations']['Overall'] if isinstance(y, int)])
            })
    return datasets.Dataset.from_list(turn_dial), datasets.Dataset.from_list(whole_dial)


def load_human_ratings_of_nlg_data(file_path:str):
    with open(file_path, newline='\n') as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=',', quotechar = '"')
        headers = next(reader)
        head_id_dict = {headers[i]:i for i in range(len(headers))}
        for row in reader:
            data.append({
                'text':row[head_id_dict['sys_ref']],
                'quality':row[head_id_dict['quality']],
                'naturalness':row[head_id_dict['naturalness']],
            })
    return datasets.Dataset.from_list(data)

def load_argessay_data(file_path:str):
    with open(file_path, newline='\n') as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=',', quotechar = '"')
        header = next(reader)
        head_id_dict = {header[i]:i for i in range(len(header))}
        for row in reader:
                #Human text
                data.append({
                    'text':row[head_id_dict['Student']],
                    'language_mastery':float(row[head_id_dict['STUD_LangMastery']]),
                    'complexity':float(row[head_id_dict['STUD_Complexity']]),
                    'vocabulary':float(row[head_id_dict['STUD_Vocab']]),
                    'language_constructs':float(row[head_id_dict['STUD_LangConstructs']]),
                })
                #GPT3 text
                data.append({
                    'text':row[head_id_dict['ChatGPT-3']],
                    'language_mastery':float(row[head_id_dict['GPT3_LangMastery']]),
                    'complexity':float(row[head_id_dict['GPT3_Complexity']]),
                    'vocabulary':float(row[head_id_dict['GPT3_Vocab']]),
                    'language_constructs':float(row[head_id_dict['GPT3_LangConstructs']]),
                })
                #GPT4 text
                data.append({
                    'text':row[head_id_dict['ChatGPT-4']],
                    'language_mastery':float(row[head_id_dict['GPT4_LangMastery']]),
                    'complexity':float(row[head_id_dict['GPT4_Complexity']]),
                    'vocabulary':float(row[head_id_dict['GPT4_Vocab']]),
                    'language_constructs':float(row[head_id_dict['GPT4_LangConstructs']]),
                })
    return datasets.Dataset.from_list(data)

def load_hanna_data(file_path: str):
    with open(file_path, newline='\n') as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        current_id = 0
        coh = []
        comp = []
        next(reader, None)  # skip the headers
        for row in reader:
            story_id = int(row[0])
            if story_id != current_id:
                data.append({'text':story, 'coherence':float(np.mean(coh)), 'complexity':float(np.mean(comp))})
                current_id = story_id
                coh = []
                comp = []
            story = row[3]
            coh.append(int(row[6]))
            comp.append(int(row[10]))
    return datasets.Dataset.from_list(data)

def load_data_webnlg(file_path: str):
    with open("data/benchmarks/web_nlg_2020_human_evals_en.json") as reader:
        data = json.loads(reader.read().strip())
    texts = collect_webnlg_texts(data)
    labels = [x['Fluency'] for x in data if os.path.exists("data/benchmarks/rdf2text/en/"+x['submission_id'])]
    return texts, labels
    
def load_data_openmeva(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as reader:
        data = json.loads(reader.read().strip())

    texts = [data[str(y)]['gen'][x]['text'] for y in list(data.keys()) for x in list(data[str(y)]['gen'].keys())]
    labels = [float(np.mean(data[str(y)]['gen'][x]['score'])) for y in list(data.keys()) for x in list(data[str(y)]['gen'].keys())]

    return texts, labels
    

def load_data_usr(file_path: str, label_dimension: str):
    with open(file_path, 'r', encoding='utf-8') as reader:
        its = json.loads(reader.read())
    texts = [y['response'].replace('\n', '') for x in its for y in x['responses']]
    labels = [float(np.mean(y[label_dimension])) for x in its for y in x['responses']]
    return texts, labels

def load_data_ellipse(file_path: str):
    data_set = []
    with open(file_path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar = '"')
        next(spamreader, None) #Skip header
        for row in spamreader:
            text = row[1]
            oa = row[18]
            cohesion = row[19]
            syntax = row[20]
            vocab = row[21]
            grammar = row[23]
            data_set.append({'text':text, 'overall':oa, 'cohesion':cohesion, 'syntax':syntax, 'vocab':vocab, 'grammar':grammar})
    return datasets.Dataset.from_list(data_set)

def load_test_data_cohesentia(file_paths: Union[str, List[str]]) -> Tuple[List[str], List[float]]:
    """
    Load texts and holistic consensus scores from one or more JSON files.

    Parameters
    ----------
    file_paths : str or list of str
        Path(s) to JSON file(s). Each file can contain either:
        - A JSON object whose values are story entries, or
        - A JSON array of story entries.

    Returns
    -------
    test_texts : list of str
        The "Text" field from each story entry.
    test_labels : list of float
        The "consensus_score" from "HolisticData" for each story entry.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    test_texts = []
    test_labels = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If the top-level structure is a dict (keyed by story ID strings),
        # iterate over its values. If it's a list, iterate directly.
        if isinstance(data, dict):
            entries = data.values()
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(f"Unexpected top-level JSON type in {path}: {type(data)}")

        for entry in entries:
            test_texts.append(entry["Text"])
            test_labels.append(entry["HolisticData"]["consensus_score"])

    return test_texts, test_labels


def main(cmd_args):

    device = torch.device('cuda')
    #Add to this
    MODEL_PATH = cmd_args[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = cmd_args[0]

    # This should be the final directory produced by the trainer, e.g.
    # /path/to/output_dir/final
    model = load_model_for_benchmark(
        model_dir=MODEL_PATH,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    probe = ["short text", "a much much longer text with different content"]
    probe_scores = model.score_texts(probe, device=device, batch_size=2, max_length=64)
    print("Sanity probe scores:", probe_scores)

    MAX_LENGTH = int(cmd_args[1]) if len(cmd_args) > 1 else 512
    BATCH_SIZE = int(cmd_args[2]) if len(cmd_args) > 2 else 32

        # ------------------------------------------------------------------
    # Preference-style HF benchmarks
    # These do NOT have scalar human scores, so we report ranking accuracy
    # rather than Spearman correlation.
    # ------------------------------------------------------------------

    print("\n================ Preference-style HF benchmarks ================\n")

    # JFLEG: corrected sentence should score higher than original sentence.
    eval_jfleg_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # MultiBLiMP English only: grammatical sentence should score higher than
    # minimally corrupted sentence.
    eval_multiblimp_english_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # Story Cloze: prompt + chosen ending should score higher than
    # prompt + rejected ending.
    eval_story_cloze_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # HellaSwag: correct continuation should be highest among 4 endings.
    # Use validation because it is the standard labeled evaluation split.
    eval_hellaswag_preference(
        device=device,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        split="validation",
    )

    print("\n================ Scalar human-score benchmarks ================\n")

    ### From here we use benchmarks with human ratings or socre annotations for a relevant dimension

    # Eval for cohesentia
    cohesentia = [
        "data/benchmarks/CohesentiaTestData.json",
        "data/benchmarks/CohesentiaTrainData.json"
    ]
    cohesentia_texts, cohesentia_labels = load_test_data_cohesentia(cohesentia)
    print(f"Total number of test texts in cohesentia: {len(cohesentia_texts)}")
    raw_preds = getModelPreds(
        device, model, cohesentia_texts,
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    # Spearman works directly — no rescaling required
    spearman, _ = safe_spearman(cohesentia_labels, raw_preds)
    print(f"Spearman ρ (cohesentia): {spearman:.4f}")

    #SummEval
    ds = datasets.load_dataset("mteb/summeval")['test']
    summeval_texts = [x for y in ds['machine_summaries'] for x in y]
    print(f"Total number of test texts in summeval: {len(summeval_texts):.4f}") 
    raw_preds = getModelPreds(
        device,
        model,
        summeval_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #fluency
    summeval_fluency_labels = [x for y in ds['fluency'] for x in y]
    spearman, _ = safe_spearman(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_fluency): {spearman:.4f}")
    #coherence
    summeval_fluency_labels = [x for y in ds['coherence'] for x in y]
    spearman, _ = safe_spearman(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_coherence): {spearman:.4f}")
    #consistency
    summeval_fluency_labels = [x for y in ds['consistency'] for x in y]
    spearman, _ = safe_spearman(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_consistency): {spearman:.4f}")

    #ELLIPSE
    ds = load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    ellipse_texts = ds['text']
    print(f"Total number of test texts in ELLIPSE: {len(ellipse_texts):.4f}") 
    raw_preds = getModelPreds(
        device,
        model,
        ellipse_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #overall
    ellipse_labels = ds['overall']
    spearman, _ = safe_spearman(ellipse_labels, raw_preds)
    print(f"Spearman ρ (ellipse_overall): {spearman:.4f}")
    #cohesion
    ellipse_labels = ds['cohesion']
    spearman, _ = safe_spearman(ellipse_labels, raw_preds)
    print(f"Spearman ρ (ellipse_cohesion): {spearman:.4f}")

    #USR
    #Topical chat
    #Overall
    tc_texts, tc_labels = load_data_usr("data/benchmarks/tc_usr_data.json", 'Overall')
    print(f"Total number of test texts in TopicalChat: {len(tc_texts):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        tc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    spearman, _ = safe_spearman(tc_labels, raw_preds)
    print(f"Spearman ρ (tc_overall): {spearman:.4f}")
    #Natural
    _, tc_labels = load_data_usr("data/benchmarks/tc_usr_data.json", 'Natural')
    spearman, _ = safe_spearman(tc_labels, raw_preds)
    print(f"Spearman ρ (tc_natural): {spearman:.4f}")
    #Persona chat
    #Overall
    pc_texts, pc_labels = load_data_usr("data/benchmarks/pc_usr_data.json", 'Overall')
    print(f"Total number of test texts in PersonaChat: {len(pc_texts):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        pc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    spearman, _ = safe_spearman(pc_labels, raw_preds)
    print(f"Spearman ρ (pc_overall): {spearman:.4f}")
    #Natural
    _, pc_labels = load_data_usr("data/benchmarks/pc_usr_data.json", 'Natural')
    spearman, _ = safe_spearman(pc_labels, raw_preds)
    print(f"Spearman ρ (pc_natural): {spearman:.4f}")

    #OpenMEVA
    meva_texts_roc, meva_labels_roc = load_data_openmeva("data/benchmarks/mans_roc.json")
    meva_texts_wp, meva_labels_wp = load_data_openmeva("data/benchmarks/mans_wp.json")
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    del meva_texts_roc
    del meva_texts_wp
    print(f"Total number of test texts in OpenMEVA: {len(meva_texts):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        meva_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    spearman, _ = safe_spearman(meva_labels, raw_preds)
    print(f"Spearman ρ (OpenMEVA_overall): {spearman:.4f}")

    #WebNLG
    #One turn utterance Fluency
    webnlg_texts, webnlg_labels = load_data_webnlg("data/benchmarks/web_nlg_2020_human_evals_en.json")
    print(f"Total number of test texts in WebNLG: {len(webnlg_texts):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        webnlg_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    spearman, _ = safe_spearman(webnlg_labels, raw_preds)
    print(f"Spearman ρ (WebNLG_overall): {spearman:.4f}")

    #HANNA
    hanna_ds = load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    print(f"Total number of test texts in HANNA: {len(hanna_ds):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        hanna_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #Coherency
    spearman, _ = safe_spearman(hanna_ds['coherence'], raw_preds)
    print(f"Spearman ρ (HANNA_coherence): {spearman:.4f}")
    #Complexity
    spearman, _ = safe_spearman(hanna_ds['complexity'], raw_preds)
    print(f"Spearman ρ (HANNA_complexity): {spearman:.4f}")

    #ARG-ESSAY
    arge_ds = load_argessay_data("data/benchmarks/arg-essay.csv")
    print(f"Total number of test texts in ARG-ESSAY: {len(arge_ds):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        arge_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #Language mastery
    spearman, _ = safe_spearman(arge_ds['language_mastery'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_language_mastery): {spearman:.4f}")
    #Complexity
    spearman, _ = safe_spearman(arge_ds['complexity'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_complexity): {spearman:.4f}")
    #Vocabulary
    spearman, _ = safe_spearman(arge_ds['vocabulary'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_vocabulary): {spearman:.4f}")
    #Language constructs
    spearman, _ = safe_spearman(arge_ds['language_constructs'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_language_constructs): {spearman:.4f}")

    #Human ratings of NLG (includes BAGEL etc.)
    hr_ds = load_human_ratings_of_nlg_data("data/benchmarks/human_ratings_of_nlg.csv")
    print(f"Total number of test texts in Human Ratings of NLG: {len(hr_ds):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        hr_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #Overall quality
    spearman, _ = safe_spearman(hr_ds['quality'], raw_preds)
    print(f"Spearman ρ (HumanRatings_quality): {spearman:.4f}")
    #Naturalness
    spearman, _ = safe_spearman(hr_ds['naturalness'], raw_preds)
    print(f"Spearman ρ (HumanRatings_naturalness): {spearman:.4f}")

    #FED
    turn_ds, whole_ds = load_fed_data("data/benchmarks/fed_data.json")
    print(f"Total number of turn level texts in FED: {len(turn_ds):.4f}")
    raw_preds_turn = getModelPreds(
        device,
        model,
        turn_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    raw_preds_whole = getModelPreds(
        device,
        model,
        whole_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #Turn, fluency
    spearman, _ = safe_spearman(turn_ds['fluent'], raw_preds_turn)
    print(f"Spearman ρ (FED_turn_fluency): {spearman:.4f}")
    #Turn, overall
    spearman, _ = safe_spearman(turn_ds['overall'], raw_preds_turn)
    print(f"Spearman ρ (FED_turn_overall): {spearman:.4f}")
    #Whole, overall
    print(f"Total number of whole dialogues in FED: {len(whole_ds):.4f}")
    spearman, _ = safe_spearman(whole_ds['overall'], raw_preds_whole)
    print(f"Spearman ρ (FED_whole_overall): {spearman:.4f}")
    del raw_preds_turn
    del raw_preds_whole

    #E2E_text_generations
    e2e_ds = load_e2e_data("data/benchmarks/E2E_data")
    print(f"Total number of texts in E2E: {len(e2e_ds):.4f}")
    raw_preds = getModelPreds(
        device,
        model,
        e2e_ds['text'],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    #Naturalness
    spearman, _ = safe_spearman(e2e_ds['naturalness'], raw_preds)
    print(f"Spearman ρ (E2E_naturalness): {spearman:.4f}")
    #Quality
    spearman, _ = safe_spearman(e2e_ds['quality'], raw_preds)
    print(f"Spearman ρ (E2E_quality): {spearman:.4f}")




if __name__ == "__main__":
     main(sys.argv[1:])