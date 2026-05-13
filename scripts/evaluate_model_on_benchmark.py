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
        attn_implementation: str = "sdpa",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if dtype is None:
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif device.type == "cuda":
                dtype = torch.float16
            else:
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

        emb_dim = self.encoder.config.hidden_size

        self.scorer = ScoringHead(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.scorer.load_state_dict(scorer_state)

        self.encoder.to(device=device, dtype=dtype)
        self.scorer.to(device=device, dtype=dtype)

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

            emb = self.mean_pool(out.last_hidden_state, tok["attention_mask"])
            scores = self.scorer(emb)

            all_scores.append(scores.detach().cpu().float())

        return torch.cat(all_scores, dim=0).numpy()


def load_model_for_benchmark(
    model_dir: str,
    device: torch.device,
    attn_implementation: str = "sdpa",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = cmd_args[0]

    # This should be the final directory produced by the trainer, e.g.
    # /path/to/output_dir/final
    model = load_model_for_benchmark(
        model_dir=MODEL_PATH,
        device=device,
        attn_implementation="sdpa",
    )

    MAX_LENGTH = int(cmd_args[1]) if len(cmd_args) > 1 else 512
    BATCH_SIZE = int(cmd_args[2]) if len(cmd_args) > 2 else 32

    # Eval for cohesentia
    cohesentia = [
        "data/benchmarks/CohesentiaTestData.json",
        "data/benchmarks/CohesentiaTrainData.json"
    ]
    cohesentia_texts, cohesentia_labels = load_test_data_cohesentia(cohesentia)
    print(f"Total number of test texts in cohesentia: {len(cohesentia_texts):.4f}")   

    raw_preds = getModelPreds(
        device,
        model,
        cohesentia_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    # Spearman works directly — no rescaling required
    spearman, _ = spearmanr(cohesentia_labels, raw_preds)
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
    raw_preds = getModelPreds(
        device,
        model,
        ellipse_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
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
    raw_preds = getModelPreds(
        device,
        model,
        tc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
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
    raw_preds = getModelPreds(
        device,
        model,
        pc_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
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
    raw_preds = getModelPreds(
        device,
        model,
        meva_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    spearman, _ = spearmanr(meva_labels, raw_preds)
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
    spearman, _ = spearmanr(webnlg_labels, raw_preds)
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
    spearman, _ = spearmanr(hanna_ds['coherence'], raw_preds)
    print(f"Spearman ρ (HANNA_coherence): {spearman:.4f}")
    #Complexity
    spearman, _ = spearmanr(hanna_ds['complexity'], raw_preds)
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
    spearman, _ = spearmanr(hr_ds['quality'], raw_preds)
    print(f"Spearman ρ (HumanRatings_quality): {spearman:.4f}")
    #Naturalness
    spearman, _ = spearmanr(hr_ds['naturalness'], raw_preds)
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
    spearman, _ = spearmanr(turn_ds['fluent'], raw_preds_turn)
    print(f"Spearman ρ (FED_turn_fluency): {spearman:.4f}")
    #Turn, overall
    spearman, _ = spearmanr(turn_ds['overall'], raw_preds_turn)
    print(f"Spearman ρ (FED_turn_overall): {spearman:.4f}")
    #Whole, overall
    spearman, _ = spearmanr(whole_ds['overall'], raw_preds_turn)
    print(f"Spearman ρ (FED_whole_overall): {spearman:.4f}")




if __name__ == "__main__":
     main(sys.argv[1:])