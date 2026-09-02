# This script has been co-created, refactored, and cleaned using GPT 5.6.
import datasets
import json
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import os
from pathlib import Path
import pandas as pd
import csv

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

# Loaders

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


def load_human_chatgpt_essay_data(file_path: str):
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


# Compatibility alias retained for older evaluation entry points.
load_argessay_data = load_human_chatgpt_essay_data


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
