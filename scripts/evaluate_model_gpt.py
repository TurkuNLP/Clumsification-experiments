"""
Evaluate a scalar-scoring SentenceTransformer model on multiple NLG benchmarks.

Refactor notes:
- The scoring head is assumed to return a *float score* per text (not rank).
- Evaluation now reports:
  - Spearman rho (rank correlation; still useful for monotonic agreement)
  - Pearson r   (linear correlation for scalar scores)
  - MAE         (absolute error; useful if scales are comparable)
- Fixed multiple bugs / robustness issues in loaders.
- Added optional score calibration (affine fit) for MAE reporting.
"""

#Refactored code with GPT of the *_on_benchmark* variant

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union, Optional

import datasets
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer


# ---------------------------
# Model / inference utilities
# ---------------------------

def load_model(
    model_path: str,
    device: Union[str, torch.device] = "cpu",
    use_bfloat16: bool = True,
) -> SentenceTransformer:
    # SentenceTransformer supports model_kwargs passed to underlying HF model.
    model_kwargs = {}
    if use_bfloat16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = SentenceTransformer(model_path, model_kwargs=model_kwargs)
    model.to(device)
    model.eval()
    return model


def get_model_scores(
    model: SentenceTransformer,
    texts: Sequence[str],
    device: Union[str, torch.device],
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Assumes model.encode returns one scalar float per text:
      - shape [N] OR [N,1]
    """
    with torch.no_grad():
        scores = model.encode(
            list(texts),
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    scores = scores.detach().float().cpu().numpy()
    scores = np.asarray(scores).squeeze()
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores after squeeze, got shape: {scores.shape}")
    return scores


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r)
    except Exception:
        return float("nan")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _fit_affine(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Fit y_true ≈ a * y_pred + b (least squares).
    Returns (a, b).
    """
    X = np.vstack([y_pred, np.ones_like(y_pred)]).T
    a, b = np.linalg.lstsq(X, y_true, rcond=None)[0]
    return float(a), float(b)


def evaluate_metric(
    name: str,
    labels: Sequence[float],
    scores: np.ndarray,
    calibrate_for_mae: bool = False,
) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(scores, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(f"{name}: label/score length mismatch ({len(y_true)} vs {len(y_pred)})")

    spearman = _safe_spearman(y_true, y_pred)
    pearson = _safe_pearson(y_true, y_pred)

    if calibrate_for_mae and len(y_true) >= 2:
        a, b = _fit_affine(y_pred, y_true)
        y_pred_mae = a * y_pred + b
        mae = _mae(y_true, y_pred_mae)
    else:
        mae = _mae(y_true, y_pred)

    print(f"[{name}] Spearman ρ: {spearman:.4f} | Pearson r: {pearson:.4f} | MAE: {mae:.4f}")
    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
    }


# ---------------------------
# Data loading helpers
# ---------------------------

def collect_webnlg_texts(
    records: List[Dict[str, Any]],
    base_dir: str = "data/benchmarks/rdf2text/en",
) -> List[str]:
    base = Path(base_dir)
    texts: List[str] = []
    file_cache: Dict[str, List[str]] = {}

    for rec in records:
        submission_id = str(rec["submission_id"])
        subdir = base / submission_id
        if not subdir.exists():
            # Keep behavior len-safe: skip if submission folder is absent
            continue

        line_idx = int(rec["sample_id"]) - 1

        if submission_id not in file_cache:
            file_path = subdir / "primary.en"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file: {file_path}")
            file_cache[submission_id] = file_path.read_text(encoding="utf-8").splitlines()

        lines = file_cache[submission_id]
        if not (0 <= line_idx < len(lines)):
            raise IndexError(f"sample_id {line_idx} out of range for {submission_id} (0..{len(lines)-1})")

        texts.append(lines[line_idx])

    return texts


def load_human_ratings_of_nlg_data(file_path: str):
    data = []
    with open(file_path, newline="\n", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        headers = next(reader)
        idx = {h: i for i, h in enumerate(headers)}
        for row in reader:
            data.append(
                {
                    "text": row[idx["sys_ref"]],
                    "quality": float(row[idx["quality"]]),
                    "naturalness": float(row[idx["naturalness"]]),
                }
            )
    return datasets.Dataset.from_list(data)


def load_argessay_data(file_path: str):
    data = []
    with open(file_path, newline="\n", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        header = next(reader)
        idx = {h: i for i, h in enumerate(header)}
        for row in reader:
            data.append({
                "text": row[idx["Student"]],
                "language_mastery": float(row[idx["STUD_LangMastery"]]),
                "complexity": float(row[idx["STUD_Complexity"]]),
                "vocabulary": float(row[idx["STUD_Vocab"]]),
                "language_constructs": float(row[idx["STUD_LangConstructs"]]),
            })
            data.append({
                "text": row[idx["ChatGPT-3"]],
                "language_mastery": float(row[idx["GPT3_LangMastery"]]),
                "complexity": float(row[idx["GPT3_Complexity"]]),
                "vocabulary": float(row[idx["GPT3_Vocab"]]),
                "language_constructs": float(row[idx["GPT3_LangConstructs"]]),
            })
            data.append({
                "text": row[idx["ChatGPT-4"]],
                "language_mastery": float(row[idx["GPT4_LangMastery"]]),
                "complexity": float(row[idx["GPT4_Complexity"]]),
                "vocabulary": float(row[idx["GPT4_Vocab"]]),
                "language_constructs": float(row[idx["GPT4_LangConstructs"]]),
            })
    return datasets.Dataset.from_list(data)


def load_hanna_data(file_path: str):
    data = []
    with open(file_path, newline="\n", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)

        current_id = None
        story = None
        coh, comp = [], []

        for row in reader:
            story_id = int(row[0])
            if current_id is None:
                current_id = story_id

            if story_id != current_id:
                data.append({
                    "text": story,
                    "coherence": float(np.mean(coh)),
                    "complexity": float(np.mean(comp)),
                })
                current_id = story_id
                coh, comp = [], []

            story = row[3]
            coh.append(float(row[6]))
            comp.append(float(row[10]))

        if story is not None and coh and comp:
            data.append({
                "text": story,
                "coherence": float(np.mean(coh)),
                "complexity": float(np.mean(comp)),
            })

    return datasets.Dataset.from_list(data)


def load_data_webnlg(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        raw = json.load(reader)

    base = Path("data/benchmarks/rdf2text/en")
    filtered = [x for x in raw if (base / str(x["submission_id"])).exists()]
    texts = collect_webnlg_texts(filtered, base_dir=str(base))
    labels = [float(x["Fluency"]) for x in filtered]

    if len(texts) != len(labels):
        raise ValueError(f"WebNLG mismatch: {len(texts)} texts vs {len(labels)} labels")
    return texts, labels


def load_data_openmeva(file_path: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        data = json.load(reader)

    texts, labels = [], []
    for y in data.keys():
        gens = data[str(y)]["gen"]
        for x in gens.keys():
            texts.append(gens[x]["text"])
            labels.append(float(np.mean(gens[x]["score"])))
    return texts, labels


def load_data_usr(file_path: str, label_dimension: str):
    with open(file_path, "r", encoding="utf-8") as reader:
        its = json.load(reader)
    texts = [y["response"].replace("\n", " ") for x in its for y in x["responses"]]
    labels = [float(np.mean(y[label_dimension])) for x in its for y in x["responses"]]
    return texts, labels


def load_data_ellipse(file_path: str):
    data = []
    with open(file_path, newline="\n", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            data.append({
                "text": row[1],
                "overall": float(row[18]),
                "cohesion": float(row[19]),
                "syntax": float(row[20]),
                "vocab": float(row[21]),
                "grammar": float(row[23]),
            })
    return datasets.Dataset.from_list(data)


def load_test_data_cohesentia(file_paths: Union[str, List[str]]) -> Tuple[List[str], List[float]]:
    if isinstance(file_paths, (str, Path)):
        file_paths = [str(file_paths)]

    texts, labels = [], []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.values() if isinstance(data, dict) else data
        for entry in entries:
            texts.append(entry["Text"])
            labels.append(float(entry["HolisticData"]["consensus_score"]))

    return texts, labels


# ---------------------------
# Benchmark spec + generic eval
# ---------------------------

@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[], Tuple[Sequence[str], Mapping[str, Sequence[float]]]]


def build_benchmark_specs() -> List[BenchmarkSpec]:
    def load_cohesentia():
        texts, labels = load_test_data_cohesentia([
            "data/benchmarks/CohesentiaTestData.json",
            "data/benchmarks/CohesentiaTrainData.json",
        ])
        return texts, {"cohesentia": labels}

    def load_summeval():
        ds = datasets.load_dataset("mteb/summeval")["test"]
        texts = [x for y in ds["machine_summaries"] for x in y]
        return texts, {
            "summeval_fluency": [x for y in ds["fluency"] for x in y],
            "summeval_coherence": [x for y in ds["coherence"] for x in y],
            "summeval_consistency": [x for y in ds["consistency"] for x in y],
        }

    def load_ellipse():
        ds = load_data_ellipse("data/benchmarks/ELLIPSE.csv")
        return ds["text"], {
            "ellipse_overall": ds["overall"],
            "ellipse_cohesion": ds["cohesion"],
        }

    def load_usr_tc():
        texts, overall = load_data_usr("data/benchmarks/tc_usr_data.json", "Overall")
        _, natural = load_data_usr("data/benchmarks/tc_usr_data.json", "Natural")
        return texts, {
            "tc_overall": overall,
            "tc_natural": natural,
        }

    def load_usr_pc():
        texts, overall = load_data_usr("data/benchmarks/pc_usr_data.json", "Overall")
        _, natural = load_data_usr("data/benchmarks/pc_usr_data.json", "Natural")
        return texts, {
            "pc_overall": overall,
            "pc_natural": natural,
        }

    def load_openmeva():
        roc_texts, roc_labels = load_data_openmeva("data/benchmarks/mans_roc.json")
        wp_texts, wp_labels = load_data_openmeva("data/benchmarks/mans_wp.json")
        texts = roc_texts + wp_texts
        labels = roc_labels + wp_labels
        return texts, {"OpenMEVA_overall": labels}

    def load_webnlg():
        texts, labels = load_data_webnlg("data/benchmarks/web_nlg_2020_human_evals_en.json")
        return texts, {"WebNLG_overall": labels}

    def load_hanna():
        ds = load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
        return ds["text"], {
            "HANNA_coherence": ds["coherence"],
            "HANNA_complexity": ds["complexity"],
        }

    def load_argessay():
        ds = load_argessay_data("data/benchmarks/arg-essay.csv")
        return ds["text"], {
            "ARG-ESSAY_language_mastery": ds["language_mastery"],
            "ARG-ESSAY_complexity": ds["complexity"],
            "ARG-ESSAY_vocabulary": ds["vocabulary"],
            "ARG-ESSAY_language_constructs": ds["language_constructs"],
        }

    def load_human_ratings():
        ds = load_human_ratings_of_nlg_data("data/benchmarks/human_ratings_of_nlg.csv")
        return ds["text"], {
            "HumanRatings_quality": ds["quality"],
            "HumanRatings_naturalness": ds["naturalness"],
        }

    return [
        BenchmarkSpec("Cohesentia", load_cohesentia),
        BenchmarkSpec("SummEval", load_summeval),
        BenchmarkSpec("ELLIPSE", load_ellipse),
        BenchmarkSpec("USR-TopicalChat", load_usr_tc),
        BenchmarkSpec("USR-PersonaChat", load_usr_pc),
        BenchmarkSpec("OpenMEVA", load_openmeva),
        BenchmarkSpec("WebNLG", load_webnlg),
        BenchmarkSpec("HANNA", load_hanna),
        BenchmarkSpec("ARG-ESSAY", load_argessay),
        BenchmarkSpec("Human Ratings of NLG", load_human_ratings),
    ]


def evaluate_benchmarks(
    model: SentenceTransformer,
    specs: List[BenchmarkSpec],
    device: Union[str, torch.device],
    batch_size: int,
    calibrate_for_mae: bool = False,
    show_progress_bar: bool = False,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for spec in specs:
        texts, label_map = spec.loader()
        print(f"\n=== {spec.name} ===")
        print(f"Total number of test texts: {len(texts)}")

        scores = get_model_scores(
            model,
            texts,
            device=device,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        for metric_name, labels in label_map.items():
            results[metric_name] = evaluate_metric(
                metric_name,
                labels,
                scores,
                calibrate_for_mae=calibrate_for_mae,
            )

    return results


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_bfloat16", action="store_true")
    parser.add_argument("--show_progress_bar", action="store_true")
    parser.add_argument(
        "--calibrate_for_mae",
        action="store_true",
        help="Fit affine mapping score->label before MAE (does not affect correlations).",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save full results JSON.",
    )
    args = parser.parse_args()

    model = load_model(
        model_path=args.model_path,
        device=args.device,
        use_bfloat16=not args.no_bfloat16,
    )

    specs = build_benchmark_specs()
    results = evaluate_benchmarks(
        model=model,
        specs=specs,
        device=args.device,
        batch_size=args.batch_size,
        calibrate_for_mae=args.calibrate_for_mae,
        show_progress_bar=args.show_progress_bar,
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()