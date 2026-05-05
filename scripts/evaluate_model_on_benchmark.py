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


if __name__ == "__main__":
     main(sys.argv[1:])