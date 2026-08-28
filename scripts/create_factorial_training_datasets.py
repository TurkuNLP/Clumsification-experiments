"""Build the fixed UniEval/FE binary and pairwise experiment datasets."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset, DatasetDict, load_from_disk

from clumsification_code.data.flattening import _aligned_items, flatten_pairwise_dataset
from clumsification_code.data.pairing import get_aligned_candidate_items


def _split_rows(rows, seed: int, dev_fraction: float = .15, test_fraction: float = .15):
    """Split duplicate groups, ensuring identical inputs stay in one split."""
    groups = {}
    for row in rows:
        groups.setdefault(row["text"], []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    n_dev = max(1, round(len(keys) * dev_fraction))
    names = ("test", "dev", "train")
    chunks = [keys[:n_test], keys[n_test:n_test + n_dev], keys[n_test + n_dev:]]
    return DatasetDict({name: Dataset.from_list([r for k in chunk for r in groups[k]]) for name, chunk in zip(names, chunks)})


def build_unieval_binary(path: str, seed: int, strip_prompt: bool = False) -> DatasetDict:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value["src"]
            if strip_prompt:
                marker = "paragraph:"
                if marker in text:
                    text = text.split(marker, 1)[1].strip()
            rows.append({"text": text, "label": int(value["tgt"] == "Yes"), "source_id": f"unieval:{index}"})
    return _split_rows(rows, seed)


def build_unmatched_pairs(binary: DatasetDict, seed: int) -> DatasetDict:
    """Create a documented fallback pair set when UniEval provenance is absent."""
    result = {}
    for split, data in binary.items():
        yes = [r for r in data if r["label"] == 1]
        no = [r for r in data if r["label"] == 0]
        random.Random(seed).shuffle(yes)
        random.Random(seed + 1).shuffle(no)
        result[split] = Dataset.from_list([
            {"pair_id": f"unmatched:{split}:{i}", "source_id": "unmatched",
             "chosen_id": yes[i]["source_id"], "rejected_id": no[i]["source_id"],
             "chosen_text": yes[i]["text"], "rejected_text": no[i]["text"],
             "chosen_layer": 0, "rejected_layer": 1, "weight": 1.0}
            for i in range(min(len(yes), len(no)))
        ])
    return DatasetDict(result)


def build_fe_binary(grouped: DatasetDict) -> DatasetDict:
    result = {}
    for split, data in grouped.items():
        rows = []
        for chain in data:
            items = _aligned_items(chain) if {"texts", "candidate_ids", "perturbation_sources"}.issubset(chain) else [
                {"source_id": chain["id"], "text": x["text"], "layer": x["label"],
                 "candidate_id": x["candidate_id"], "perturbation_source": x["perturbation_source"]}
                for x in get_aligned_candidate_items(chain)
            ]
            for item in items:
                rows.append({"text": item["text"], "label": int(item["layer"] == 0),
                             "source_id": item["source_id"], "candidate_id": item["candidate_id"],
                             "layer": item["layer"], "perturbation_source": item["perturbation_source"]})
        result[split] = Dataset.from_list(rows)
    return DatasetDict(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unieval-file", required=True)
    parser.add_argument("--fe-formatted-dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strip-unieval-prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    datasets = {
        "unieval_binary": build_unieval_binary(args.unieval_file, args.seed, args.strip_unieval_prompt),
    }
    datasets["unieval_pairwise_unmatched"] = build_unmatched_pairs(datasets["unieval_binary"], args.seed)
    grouped = load_from_disk(args.fe_formatted_dataset)
    if not {"texts", "candidate_ids", "perturbation_sources"}.issubset(grouped["train"].column_names):
        # Normalize the historical text_label_pairs schema for the canonical
        # flattening helper while retaining all provenance fields.
        normalized = {}
        for split, data in grouped.items():
            normalized[split] = Dataset.from_list([{
                **{k: row[k] for k in ("id", "dataset_name", "source_original_ids") if k in row},
                "texts": row["texts"] if "texts" in row else [x[0] for x in row["text_label_pairs"]],
                "labels": row["labels"] if "labels" in row else [x[1] for x in row["text_label_pairs"]],
                "candidate_ids": row.get("candidate_ids", [f"{row['id']}:{i}" for i in range(len(row["texts"]))]),
                "perturbation_sources": row.get("perturbation_sources", ["original"] + ["trad"] * (len(row["texts"]) - 1)),
            } for row in data])
        grouped = DatasetDict(normalized)
    datasets["fe_binary"] = build_fe_binary(grouped)
    try:
        datasets["fe_pairwise"] = DatasetDict({s: flatten_pairwise_dataset(grouped[s], policy="original_only") for s in ("train", "dev", "test")})
    except ValueError as exc:
        # Historical formatted datasets may not identify their original row;
        # retain them as a valid fallback, but make the policy explicit.
        if "exactly one original" not in str(exc):
            raise
        datasets["fe_pairwise"] = DatasetDict({s: flatten_pairwise_dataset(grouped[s], policy="all_unequal_layers") for s in ("train", "dev", "test")})
    for name, dataset in datasets.items():
        path = root / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite")
        dataset.save_to_disk(str(path))
        (path / "metadata.json").write_text(json.dumps({"dataset": name, "seed": args.seed, "unmatched_pairs": name == "unieval_pairwise_unmatched"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
