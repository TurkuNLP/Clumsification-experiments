from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import sys
import os
import json
from typing import List, Tuple, Union
from pathlib import Path
import datasets
import csv

class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # constrains to [0, 1]

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)
    
def load_model_and_head(
        model_path: str,
        head_path: str,
        emb_dim: int = 768,
        device: torch.device = torch.device("cpu"),   # <-- accept device
    ):
        loaded_model = SentenceTransformer(model_path, model_kwargs={
            "dtype": torch.bfloat16,
        })

        loaded_head = nn.Linear(emb_dim, 1)
        loaded_head.load_state_dict(torch.load(head_path, map_location=device))
        loaded_head.to(device=device, dtype=torch.bfloat16)  # <-- match device AND dtype
        loaded_head.eval()

        return loaded_model, loaded_head

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

def load_argessay_data(file_path:str):
      with open(file_path, newline='\n') as csvfile:
            data = []
            reader = csv.reader(csvfile, delimiter=',', quotechar = '"')
            header = next(reader)
            head_id_dict = {header[i]:i for i in range(len(header))}
            print(head_id_dict)
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
    labels = [x['Fluency'] for x in data]
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

def getModelPreds(device, model, score_head, test_texts):
    # Inference
    with torch.no_grad():
        embeddings = model.encode(test_texts, convert_to_tensor=True, device=device)
        raw_preds = score_head(embeddings).cpu().float().numpy()  
    return raw_preds


def main(cmd_args):

    device = torch.device('cuda')
    #Add to this
    MODEL_PATH = cmd_args[0]
    HEAD_PATH = MODEL_PATH+"scoring_head.pt"

    model, score_head = load_model_and_head(
         MODEL_PATH,
         HEAD_PATH,
         device=device,
    )
    score_head.eval()

    # Eval for cohesentia
    cohesentia = [
        "data/benchmarks/CohesentiaTestData.json",
        "data/benchmarks/CohesentiaTrainData.json"
    ]
    cohesentia_texts, cohesentia_labels = load_test_data_cohesentia(cohesentia)
    print(f"Total number of test texts in cohesentia: {len(cohesentia_texts):.4f}")   

    raw_preds = getModelPreds(device, model, score_head, cohesentia_texts)
    # Spearman works directly — no rescaling required
    spearman, _ = spearmanr(cohesentia_labels, raw_preds)
    print(f"Spearman ρ (cohesentia): {spearman:.4f}")

    #SummEval
    ds = datasets.load_dataset("mteb/summeval")['test']
    summeval_texts = [x for y in ds['machine_summaries'] for x in y]
    print(f"Total number of test texts in summeval: {len(summeval_texts):.4f}") 
    raw_preds = getModelPreds(device, model, score_head, summeval_texts)
    #fluency
    summeval_fluency_labels = [x for y in ds['fluency'] for x in y]
    spearman, _ = spearmanr(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_fluency): {spearman:.4f}")
    #coherence
    summeval_fluency_labels = [x for y in ds['coherence'] for x in y]
    spearman, _ = spearmanr(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_coherence): {spearman:.4f}")
    #consistency
    summeval_fluency_labels = [x for y in ds['consistency'] for x in y]
    spearman, _ = spearmanr(summeval_fluency_labels, raw_preds)
    print(f"Spearman ρ (summeval_consistency): {spearman:.4f}")

    #ELLIPSE
    ds = load_data_ellipse("data/benchmarks/ELLIPSE.csv")
    ellipse_texts = ds['text']
    print(f"Total number of test texts in ELLIPSE: {len(ellipse_texts):.4f}") 
    raw_preds = getModelPreds(device, model, score_head, ellipse_texts)
    #overall
    ellipse_labels = ds['overall']
    spearman, _ = spearmanr(ellipse_labels, raw_preds)
    print(f"Spearman ρ (ellipse_overall): {spearman:.4f}")
    #cohesion
    ellipse_labels = ds['cohesion']
    spearman, _ = spearmanr(ellipse_labels, raw_preds)
    print(f"Spearman ρ (ellipse_cohesion): {spearman:.4f}")

    #USR
    #Topical chat
    #Overall
    tc_texts, tc_labels = load_data_usr("data/benchmarks/tc_usr_data.json", 'Overall')
    print(f"Total number of test texts in TopicalChat: {len(tc_texts):.4f}")
    raw_preds = getModelPreds(device, model, score_head, tc_texts)
    spearman, _ = spearmanr(tc_labels, raw_preds)
    print(f"Spearman ρ (tc_overall): {spearman:.4f}")
    #Natural
    _, tc_labels = load_data_usr("data/benchmarks/tc_usr_data.json", 'Natural')
    spearman, _ = spearmanr(tc_labels, raw_preds)
    print(f"Spearman ρ (tc_natural): {spearman:.4f}")
    #Persona chat
    #Overall
    pc_texts, pc_labels = load_data_usr("data/benchmarks/pc_usr_data.json", 'Overall')
    print(f"Total number of test texts in PersonaChat: {len(pc_texts):.4f}")
    raw_preds = getModelPreds(device, model, score_head, pc_texts)
    spearman, _ = spearmanr(pc_labels, raw_preds)
    print(f"Spearman ρ (pc_overall): {spearman:.4f}")
    #Natural
    _, pc_labels = load_data_usr("data/benchmarks/pc_usr_data.json", 'Natural')
    spearman, _ = spearmanr(pc_labels, raw_preds)
    print(f"Spearman ρ (pc_natural): {spearman:.4f}")

    #OpenMEVA
    meva_texts_roc, meva_labels_roc = load_data_openmeva("data/benchmarks/mans_roc.json")
    meva_texts_wp, meva_labels_wp = load_data_openmeva("data/benchmarks/mans_wp.json")
    meva_texts = meva_texts_roc + meva_texts_wp
    meva_labels = meva_labels_roc + meva_labels_wp
    del meva_texts_roc
    del meva_texts_wp
    print(f"Total number of test texts in OpenMEVA: {len(meva_texts):.4f}")
    raw_preds = getModelPreds(device, model, score_head, meva_texts)
    spearman, _ = spearmanr(meva_labels, raw_preds)
    print(f"Spearman ρ (OpenMEVA_overall): {spearman:.4f}")

    #WebNLG
    #One turn utterance Fluency
    webnlg_texts, webnlg_labels = load_data_webnlg("data/benchmarks/web_nlg_2020_human_evals_en.json")
    print(f"Total number of test texts in WebNLG: {len(pc_texts):.4f}")
    raw_preds = getModelPreds(device, model, score_head, webnlg_texts)
    spearman, _ = spearmanr(webnlg_labels, raw_preds)
    print(f"Spearman ρ (WebNLG_overall): {spearman:.4f}")

    #HANNA
    hanna_ds = load_hanna_data("data/benchmarks/hanna_stories_annotations.csv")
    print(f"Total number of test texts in HANNA: {len(hanna_ds):.4f}")
    raw_preds = getModelPreds(device, model, score_head, hanna_ds['text'])
    #Coherency
    spearman, _ = spearmanr(hanna_ds['coherency'], raw_preds)
    print(f"Spearman ρ (HANNA_coherency): {spearman:.4f}")
    #Complexity
    spearman, _ = spearmanr(hanna_ds['complexity'], raw_preds)
    print(f"Spearman ρ (HANNA_complexity): {spearman:.4f}")

    #ARG-ESSAY
    arge_ds = load_argessay_data("data/benchmarks/arg-essay.csv")
    print(f"Total number of test texts in ARG-ESSAY: {len(arge_ds):.4f}")
    raw_preds = getModelPreds(device, model, score_head, arge_ds['text'])
    #Language mastery
    spearman, _ = spearmanr(arge_ds['language_mastery'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_language_mastery): {spearman:.4f}")
    #Complexity
    spearman, _ = spearmanr(arge_ds['complexity'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_complexity): {spearman:.4f}")
    #Vocabulary
    spearman, _ = spearmanr(arge_ds['vocabulary'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_vocabulary): {spearman:.4f}")
    #Language constructs
    spearman, _ = spearmanr(arge_ds['language_constructs'], raw_preds)
    print(f"Spearman ρ (ARG-ESSAY_language_constructs): {spearman:.4f}")



if __name__ == "__main__":
     main(sys.argv[1:])