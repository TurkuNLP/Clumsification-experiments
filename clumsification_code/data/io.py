import os
import json

def default_formatted_dataset_path(dataset_name: str) -> str:
    return os.path.join(
        "data",
        "hf_datasets",
        dataset_name,
    )

def read_ds(ds_path: str):
    #The most simple of helper functions
    rows = []

    with open(ds_path, "r", encoding="utf-8") as reader:
        for line in reader:
            if len(line.strip()) > 0:
                rows.append(json.loads(line.strip()))

    return rows