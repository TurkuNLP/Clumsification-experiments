import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Dict, Tuple
from tqdm.auto import tqdm
import os
import numpy as np
from datasets import Dataset, DatasetDict
from collections import defaultdict
import json
from itertools import combinations
from scipy import stats as scipy_stats
import pandas as pd


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

        head_state = torch.load(head_path, map_location="cpu")
        hidden_dim = head_state["hidden_dim"]
        dropout = head_state["dropout"]
        scorer_state = head_state["scorer"]

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
        self.scorer.to(device=device, dtype=torch.float32)

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


# Function for loading regenerated tdt data
def form_custom_ud_ds(folder: str) -> DatasetDict:
    og = []
    regens = []
    with open(folder + "ud_data.jsonl") as reader:
        for l in reader:
            if len(l) > 1:
                og.append(json.loads(l.strip()))
    with open(folder + "regenerations.jsonl") as reader:
        for l in reader:
            if len(l) > 1:
                regens.append(json.loads(l.strip()))

    groups = defaultdict(list)
    for item in regens:
        key = f"{item['model']}_{item['effort']}"
        groups[key].append({"id": item["id"], "text": item["text"]})

    ds_dict = {
        "original": Dataset.from_list(
            [{"id": item["id"], "text": item["text"]} for item in og]
        )
    }

    for key, items in groups.items():
        ds_dict[key] = Dataset.from_list(items)

    return DatasetDict(ds_dict)


# ---------------------------------------------------------------------------
#  Evaluation pipeline
# ---------------------------------------------------------------------------

def score_dataset_dict(
    model: LTRInferenceModel,
    ds_dict: DatasetDict,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """
    Score every item in every split of the DatasetDict.

    Returns
    -------
    scored : dict[str, dict[str, float]]
        Mapping  split_name  ->  { id: score }
    """
    scored: Dict[str, Dict[str, float]] = {}

    for split_name, ds in ds_dict.items():
        print(f"\n>>> Scoring split '{split_name}' ({len(ds)} items)")
        texts = ds["text"]
        ids = ds["id"]

        scores = model.score_texts(
            texts=texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        scored[split_name] = {uid: float(s) for uid, s in zip(ids, scores)}

    return scored


def compute_mean_scores(
    scored: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-split summary statistics: mean, std, median, min, max.
    """
    summary = {}
    for split, id_scores in scored.items():
        vals = np.array(list(id_scores.values()))
        summary[split] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }
    return summary


def _align_scores_by_id(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return aligned numpy arrays for items whose id appears in both splits."""
    common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    arr_a = np.array([scores_a[uid] for uid in common_ids])
    arr_b = np.array([scores_b[uid] for uid in common_ids])
    return arr_a, arr_b, common_ids


def compute_pairwise_winrates(
    scored: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    For every ordered pair (A, B) of splits, compute:
      - win_rate_A : fraction of shared ids where A scores higher
      - tie_rate   : fraction of shared ids where scores are equal
      - n_compared : number of shared ids
      - binom_p    : two-sided binomial test p-value (H0: win_rate = 0.5,
                     ties excluded)

    Returns a DataFrame with one row per pair.
    """
    splits = sorted(scored.keys())
    rows = []

    for a, b in combinations(splits, 2):
        arr_a, arr_b, common = _align_scores_by_id(scored[a], scored[b])
        n = len(common)
        if n == 0:
            rows.append({
                "split_a": a, "split_b": b,
                "n_compared": 0,
                "win_rate_a": np.nan, "win_rate_b": np.nan,
                "tie_rate": np.nan, "binom_p": np.nan,
            })
            continue

        wins_a = int((arr_a > arr_b).sum())
        wins_b = int((arr_b > arr_a).sum())
        ties = int((arr_a == arr_b).sum())
        n_decisive = wins_a + wins_b  # exclude ties for the binomial test

        wr_a = wins_a / n
        wr_b = wins_b / n
        tie_r = ties / n

        # Two-sided binomial test (H0: p = 0.5) on decisive comparisons
        if n_decisive > 0:
            binom_p = scipy_stats.binomtest(
                wins_a, n_decisive, 0.5
            ).pvalue
        else:
            binom_p = 1.0

        rows.append({
            "split_a": a,
            "split_b": b,
            "n_compared": n,
            "win_rate_a": wr_a,
            "win_rate_b": wr_b,
            "tie_rate": tie_r,
            "binom_p": binom_p,
        })

    return pd.DataFrame(rows)


def compute_pairwise_correlations(
    scored: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    For every pair of splits, compute Pearson and Spearman correlation
    (with p-values) on the scores of their shared ids.
    """
    splits = sorted(scored.keys())
    rows = []

    for a, b in combinations(splits, 2):
        arr_a, arr_b, common = _align_scores_by_id(scored[a], scored[b])
        n = len(common)

        if n < 3:
            rows.append({
                "split_a": a, "split_b": b, "n": n,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
            })
            continue

        pr, pp = scipy_stats.pearsonr(arr_a, arr_b)
        sr, sp = scipy_stats.spearmanr(arr_a, arr_b)

        rows.append({
            "split_a": a, "split_b": b, "n": n,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        })

    return pd.DataFrame(rows)


def compute_pairwise_paired_tests(
    scored: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    For every pair of splits run a Wilcoxon signed-rank test and a
    paired t-test on the score differences (shared ids only).
    """
    splits = sorted(scored.keys())
    rows = []

    for a, b in combinations(splits, 2):
        arr_a, arr_b, common = _align_scores_by_id(scored[a], scored[b])
        n = len(common)
        diffs = arr_a - arr_b

        if n < 2:
            rows.append({
                "split_a": a, "split_b": b, "n": n,
                "mean_diff": np.nan,
                "ttest_stat": np.nan, "ttest_p": np.nan,
                "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
            })
            continue

        t_stat, t_p = scipy_stats.ttest_rel(arr_a, arr_b)

        # Wilcoxon needs at least one non-zero difference
        if np.any(diffs != 0):
            w_stat, w_p = scipy_stats.wilcoxon(arr_a, arr_b)
        else:
            w_stat, w_p = np.nan, 1.0

        rows.append({
            "split_a": a, "split_b": b, "n": n,
            "mean_diff": float(np.mean(diffs)),
            "ttest_stat": t_stat, "ttest_p": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        })

    return pd.DataFrame(rows)


def build_winrate_matrix(
    scored: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Build a square matrix where cell (A, B) = win-rate of A vs B.
    Diagonal is NaN. Useful for a quick heatmap overview.
    """
    splits = sorted(scored.keys())
    mat = pd.DataFrame(np.nan, index=splits, columns=splits)

    for a, b in combinations(splits, 2):
        arr_a, arr_b, _ = _align_scores_by_id(scored[a], scored[b])
        n = len(arr_a)
        if n == 0:
            continue
        mat.loc[a, b] = float((arr_a > arr_b).sum()) / n
        mat.loc[b, a] = float((arr_b > arr_a).sum()) / n

    return mat


def compute_winrate_vs_mean_score_correlation(
    mean_scores: Dict[str, Dict[str, float]],
    winrate_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Correlate each split's mean score with its average win-rate
    against all other splits. Reports Spearman rho + p-value.
    """
    splits = list(winrate_matrix.index)
    avg_winrates = {}
    for s in splits:
        row = winrate_matrix.loc[s].dropna()
        avg_winrates[s] = row.mean() if len(row) > 0 else np.nan

    means = np.array([mean_scores[s]["mean"] for s in splits])
    wrs = np.array([avg_winrates[s] for s in splits])

    valid = ~(np.isnan(means) | np.isnan(wrs))
    if valid.sum() < 3:
        return {
            "spearman_r": np.nan, "spearman_p": np.nan,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "n_splits": int(valid.sum()),
        }

    sr, sp = scipy_stats.spearmanr(means[valid], wrs[valid])
    pr, pp = scipy_stats.pearsonr(means[valid], wrs[valid])

    return {
        "spearman_r": sr, "spearman_p": sp,
        "pearson_r": pr, "pearson_p": pp,
        "n_splits": int(valid.sum()),
    }


def run_full_evaluation(
    model: LTRInferenceModel,
    ds_dict: DatasetDict,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    output_dir: Optional[str] = None,
):
    """
    End-to-end evaluation pipeline:
      1. Score every item
      2. Compute per-split summary statistics
      3. Compute pairwise win-rates (+ binomial significance)
      4. Compute pairwise score correlations (Pearson & Spearman)
      5. Compute paired statistical tests (t-test & Wilcoxon)
      6. Build win-rate matrix
      7. Correlate mean score ↔ average win-rate across splits

    If output_dir is given, all tables are saved as CSVs there.
    """

    # 1) Score
    scored = score_dataset_dict(
        model, ds_dict, device,
        batch_size=batch_size,
        max_length=max_length,
    )

    # 2) Summary stats
    summary = compute_mean_scores(scored)
    df_summary = pd.DataFrame(summary).T
    df_summary.index.name = "split"
    print("\n" + "=" * 70)
    print("PER-SPLIT SUMMARY STATISTICS")
    print("=" * 70)
    print(df_summary.to_string())

    # 3) Pairwise win-rates
    df_winrates = compute_pairwise_winrates(scored)
    print("\n" + "=" * 70)
    print("PAIRWISE WIN-RATES  (split_a vs split_b)")
    print("=" * 70)
    print(df_winrates.to_string(index=False))

    # 4) Pairwise correlations
    df_corr = compute_pairwise_correlations(scored)
    print("\n" + "=" * 70)
    print("PAIRWISE SCORE CORRELATIONS  (shared ids)")
    print("=" * 70)
    print(df_corr.to_string(index=False))

    # 5) Paired tests
    df_paired = compute_pairwise_paired_tests(scored)
    print("\n" + "=" * 70)
    print("PAIRED STATISTICAL TESTS  (split_a - split_b)")
    print("=" * 70)
    print(df_paired.to_string(index=False))

    # 6) Win-rate matrix
    wr_matrix = build_winrate_matrix(scored)
    print("\n" + "=" * 70)
    print("WIN-RATE MATRIX  (row beats column)")
    print("=" * 70)
    print(wr_matrix.to_string())

    # 7) Mean score ↔ average win-rate correlation
    wr_vs_mean = compute_winrate_vs_mean_score_correlation(summary, wr_matrix)
    print("\n" + "=" * 70)
    print("CORRELATION: MEAN SCORE  ↔  AVG WIN-RATE ACROSS SPLITS")
    print("=" * 70)
    for k, v in wr_vs_mean.items():
        print(f"  {k}: {v}")

    # Save if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df_summary.to_csv(os.path.join(output_dir, "summary_stats.csv"))
        df_winrates.to_csv(os.path.join(output_dir, "pairwise_winrates.csv"), index=False)
        df_corr.to_csv(os.path.join(output_dir, "pairwise_correlations.csv"), index=False)
        df_paired.to_csv(os.path.join(output_dir, "paired_tests.csv"), index=False)
        wr_matrix.to_csv(os.path.join(output_dir, "winrate_matrix.csv"))

        # Save raw scores as JSONL
        with open(os.path.join(output_dir, "raw_scores.jsonl"), "w") as f:
            for split_name, id_scores in scored.items():
                for uid, score in id_scores.items():
                    f.write(json.dumps({
                        "split": split_name, "id": uid, "score": score
                    }) + "\n")

        with open(os.path.join(output_dir, "mean_vs_winrate_corr.json"), "w") as f:
            json.dump(wr_vs_mean, f, indent=2)

        print(f"\n>>> All results saved to {output_dir}/")

    return {
        "scored": scored,
        "summary": df_summary,
        "winrates": df_winrates,
        "correlations": df_corr,
        "paired_tests": df_paired,
        "winrate_matrix": wr_matrix,
        "mean_vs_winrate_corr": wr_vs_mean,
    }

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a model on Finnish text scoring."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for scoring (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for scoring (default: 32).",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data/",
        help="Path to the input data folder (default: data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/",
        help="Path to the output results directory (default: eval_results/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_for_benchmark(
        model_dir=args.model,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    fin_sanity_check = [
        "Eilen menimme luokan kanssa retkelle, ja ensimmäinen paikka oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan. Rakensimme laudan avulla pienen sillan puron yli, ja se jäi metsään paikalle, jonka muistamme varmasti seuraavalla retkellä.",
        "Menimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Rakensimme sen avulla pienen sillan puron yli. Jätimme laudan metsään sellaiseen paikkaan, jonka varmasti muistamme seuraavalla retkellä.",
        "Eilen menimme luokan kanssa retkelle, ja ensimmäinen paikka oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän laudan, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan. Rakensimme laudan avulla pienen sillan puron yli, ja se jäi metsään paikalle, jonka muistamme varmasti seuraavalla retkellä.",
    ]
    san_check_scores = model.score_texts(
        fin_sanity_check, device=device, batch_size=3, max_length=128
    )
    print("Sanity check scores:", san_check_scores)

    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size
    DATA_FOLDER = args.data_folder
    OUTPUT_DIR = args.output_dir

    # Load the DatasetDict
    ds_dict = form_custom_ud_ds(DATA_FOLDER)
    print(f"\nLoaded DatasetDict with splits: {list(ds_dict.keys())}")
    for name, ds in ds_dict.items():
        print(f"  {name}: {len(ds)} items")

    # Run full evaluation
    results = run_full_evaluation(
        model=model,
        ds_dict=ds_dict,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()