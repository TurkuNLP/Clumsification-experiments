"""Validate schemas and split invariants of factorial DatasetDicts."""
from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_from_disk

EXPECTED = {
    "unieval_binary": {"text", "label"},
    "fe_binary": {"text", "label"},
    "unieval_pairwise_unmatched": {"chosen_text", "rejected_text", "chosen_layer", "rejected_layer"},
    "fe_pairwise": {"chosen_text", "rejected_text", "chosen_layer", "rejected_layer"},
}

def validate(root: str) -> None:
    for name, required in EXPECTED.items():
        dataset = load_from_disk(str(Path(root) / name))
        if set(dataset) != {"train", "dev", "test"}:
            raise ValueError(f"{name}: expected train/dev/test, got {set(dataset)}")
        for split, rows in dataset.items():
            missing = required - set(rows.column_names)
            if missing or not len(rows):
                raise ValueError(f"{name}/{split}: missing={sorted(missing)}, rows={len(rows)}")
        print(name, {split: len(rows) for split, rows in dataset.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/hf_datasets/factorial")
    validate(parser.parse_args().root)
