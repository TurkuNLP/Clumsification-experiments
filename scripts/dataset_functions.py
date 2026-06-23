import json
import os
import random
import shutil
from collections import defaultdict
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


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

# Data should be in format:
# {col, col_id, text, under_col, under_col_id}
def format_datasets(dss:list[dict[str]]):
    # Dicts are in format: {ds_path, ds_name, under_ds_name}
    ds_items = []
    for x in dss:
        ds_path = x['ds_path']
        ds_name = x['ds_name']
        ds_under = x['under_ds_name']
        with open(ds_path, 'r', encoding='utf-8') as reader:
            for i,l in enumerate(reader):
                if len(l)>0:
                    line = json.loads(l.strip())
                    if ds_under:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':ds_under, 'under_collection_id':line['custom_id']})
                    else:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':None, 'under_collection_id':None})



    return ds_items

def init_og_dataset(source_txt_path: str, new_ds_name: str, overwrite: bool = False):
    # If a txt-file with one text per line, then create a jsonl-file so that it fits with the rest of the code
    # If the file already contains valid JSON objects, preserve them and just add custom_id
    # Also creates the source folder for the specified dataset if necessary
    if not os.path.exists("data/custom_datasets/" + new_ds_name):
        os.makedirs("data/custom_datasets/" + new_ds_name + "/perturbed_layers", exist_ok=True)
        os.makedirs("data/custom_datasets/" + new_ds_name + "/trad_perturbed_layers", exist_ok=True)
    if not os.path.exists("data/custom_datasets/" + new_ds_name + "/original.jsonl") or overwrite:

        # Detection pass: check if every non-empty line is valid JSON
        is_jsonl = True
        has_content = False
        with open(source_txt_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                stripped = line.strip()
                if not stripped:
                    continue
                has_content = True
                try:
                    obj = json.loads(stripped)
                    if not isinstance(obj, dict):
                        is_jsonl = False
                        break
                except (json.JSONDecodeError, ValueError):
                    is_jsonl = False
                    break

        if not has_content:
            is_jsonl = False

        # Writing pass
        with open("data/custom_datasets/" + new_ds_name + "/original.jsonl", 'w', encoding='utf-8') as writer:
            with open(source_txt_path, 'r', encoding='utf-8') as reader:
                i = 0
                for line in reader:
                    stripped = line.strip()
                    if is_jsonl:
                        if not stripped:
                            continue
                        obj = json.loads(stripped)
                        if obj.get('passes_filters', None):
                            if obj['passes_filters'] == "Yes":
                                obj['custom_id'] = str(i)
                                writer.write(json.dumps(obj) + '\n')
                            else:
                                continue
                        else:
                            obj['custom_id'] = str(i)
                            writer.write(json.dumps(obj) + '\n')
                    else:
                        writer.write(json.dumps({'custom_id': str(i), 'text': line.replace('\n', '')}) + '\n')
                    i += 1

    return "data/custom_datasets/" + new_ds_name + "/original.jsonl"


def format_custom_dataset(
    custom_dataset_name: str,
    max_layers: Optional[int] = None,
    layer_type: str = "clumsy",
    seed: int = 42,
    reuse_limit: int = 5,
):
    """
    Load one raw custom dataset and return a list of dictionaries:

        {
            "id": int,
            "text_label_pairs": [(text, label), ...]
        }

    layer_type controls which perturbation subfolder(s) to use:

      - "clumsy": perturbed_layers only
      - "trad":   trad_perturbed_layers only
      - "mix":    50/50 split by original-text ID
      - "all":    all data from both folders; duplicate originals are allowed
    """

    work_path = os.path.join("data", "custom_datasets", custom_dataset_name)
    original_path = os.path.join(work_path, "original.jsonl")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Missing original dataset file: {original_path}")

    original_texts = read_ds(original_path)

    def _build_id_dict(layer_dir: str, id_filter: Optional[set[int]] = None):
        if id_filter is not None:
            id_dict = {
                i: {"id": i, "text_label_pairs": [(x["text"], 0)]}
                for i, x in enumerate(original_texts)
                if i in id_filter
            }
        else:
            id_dict = {
                i: {"id": i, "text_label_pairs": [(x["text"], 0)]}
                for i, x in enumerate(original_texts)
            }

        layer_path = os.path.join(work_path, layer_dir)

        if not os.path.isdir(layer_path):
            raise FileNotFoundError(f"Missing perturbation layer directory: {layer_path}")

        for file_name in os.listdir(layer_path):
            if not file_name.endswith(".jsonl"):
                continue

            try:
                layer = int(file_name.replace(".jsonl", ""))
            except ValueError:
                continue

            if max_layers is not None and layer >= max_layers:
                continue

            contents = read_ds(os.path.join(layer_path, file_name))

            for x in contents:
                head_id = x["head_id"]

                if isinstance(head_id, str):
                    head_id = int(head_id)

                if head_id not in id_dict:
                    continue

                id_dict[head_id]["text_label_pairs"].append((x["text"], layer))

        return id_dict

    if layer_type == "clumsy":
        id_dict = _build_id_dict("perturbed_layers")
        return list(id_dict.values())

    if layer_type == "trad":
        id_dict = _build_id_dict("trad_perturbed_layers")
        return list(id_dict.values())

    if layer_type == "mix":
        all_ids = list(range(len(original_texts)))
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        mid = len(all_ids) // 2
        clumsy_ids = set(all_ids[:mid])
        trad_ids = set(all_ids[mid:])

        clumsy_dict = _build_id_dict("perturbed_layers", id_filter=clumsy_ids)
        trad_dict = _build_id_dict("trad_perturbed_layers", id_filter=trad_ids)

        return list(clumsy_dict.values()) + list(trad_dict.values())

    if layer_type == "all":
        def _prefix_ids(records: list[dict], prefix: str) -> list[dict]:
            prefixed = []

            for entry in records:
                prefixed.append(
                    {
                        "id": f"{custom_dataset_name}__{prefix}__{entry['id']}",
                        "text_label_pairs": list(entry["text_label_pairs"]),
                    }
                )

            return prefixed

        clumsy_chains = _prefix_ids(
            list(_build_id_dict("perturbed_layers").values()),
            "clumsy_chain",
        )

        trad_chains = _prefix_ids(
            list(_build_id_dict("trad_perturbed_layers").values()),
            "trad_chain",
        )

        clumsy_pairs = generate_training_pairs_random(
            formatted_dataset=clumsy_chains,
            seed=seed,
            n=reuse_limit,
        )

        trad_pairs = generate_training_pairs_random(
            formatted_dataset=trad_chains,
            seed=seed + 1 if seed is not None else None,
            n=reuse_limit,
        )

        clumsy_pairs = _prefix_ids(clumsy_pairs, "clumsy_pair")
        trad_pairs = _prefix_ids(trad_pairs, "trad_pair")

        return (
            clumsy_chains
            + trad_chains
            + clumsy_pairs
            + trad_pairs
        )

    raise ValueError(f"Unknown layer_type: {layer_type!r}")

    
def generate_training_pairs_random(
    formatted_dataset: list[dict],
    seed: Optional[int] = None,
    n: int = 5,
):
    """
    Construct random two-item training examples without materializing all
    O(num_items^2) possible pairs.

    Each individual text may be reused at most n times.
    """

    rng = random.Random(seed)

    items = []

    for entry in formatted_dataset:
        entry_id = entry["id"]

        for text, label in entry["text_label_pairs"]:
            items.append(
                {
                    "source_id": entry_id,
                    "text": text,
                    "label": label,
                }
            )

    if len(items) < 2:
        return []

    by_label = defaultdict(list)

    for idx, item in enumerate(items):
        by_label[item["label"]].append(idx)

    if len(by_label) < 2:
        return []

    usage = [0] * len(items)
    seen_pairs = set()
    result = []

    max_pairs = max(1, (len(items) * n) // 2)
    failed_attempts = 0
    max_failed_attempts = max(1000, max_pairs * 50)

    def available_indices_for_label(label):
        return [idx for idx in by_label[label] if usage[idx] < n]

    while len(result) < max_pairs and failed_attempts < max_failed_attempts:
        available_labels = [
            label
            for label in by_label
            if len(available_indices_for_label(label)) > 0
        ]

        if len(available_labels) < 2:
            break

        label_a, label_b = rng.sample(available_labels, 2)

        candidates_a = available_indices_for_label(label_a)
        candidates_b = available_indices_for_label(label_b)

        if not candidates_a or not candidates_b:
            failed_attempts += 1
            continue

        i = rng.choice(candidates_a)
        j = rng.choice(candidates_b)

        if i == j:
            failed_attempts += 1
            continue

        pair_key = tuple(sorted((i, j)))

        if pair_key in seen_pairs:
            failed_attempts += 1
            continue

        seen_pairs.add(pair_key)

        item_i = items[i]
        item_j = items[j]

        usage[i] += 1
        usage[j] += 1

        combined_id = f"{item_i['source_id']}_{item_j['source_id']}_{len(result)}"

        result.append(
            {
                "id": combined_id,
                "text_label_pairs": [
                    (item_i["text"], item_i["label"]),
                    (item_j["text"], item_j["label"]),
                ],
            }
        )

        failed_attempts = 0

    return result


def shuffle_and_transform_formatted_dataset(
    formatted_dataset: list[dict],
    seed: Optional[int] = None,
):
    """
    Convert list records of this form:
        {
            "id": ...,
            "text_label_pairs": [(text, label), ...]
        }

    into a Hugging Face Dataset with this form:
        {
            "id": ...,
            "texts": [...],
            "labels": [...]
        }
    """

    rng = random.Random(seed)
    rows = []

    for entry in formatted_dataset:
        tl_list = list(entry.get("text_label_pairs", []))

        if len(tl_list) < 2:
            # Pairwise training needs at least two items.
            continue

        rng.shuffle(tl_list)

        rows.append(
            {
                "id": entry["id"],
                "texts": [x[0] for x in tl_list],
                "labels": [x[1] for x in tl_list],
            }
        )

    return Dataset.from_list(rows)

        
def create_formatted_dataset_dict(
    dataset_names: list[str],
    max_layers: Optional[int] = None,
    layer_type: str = "clumsy",
    seed: int = 42,
    random_pairs: bool = False,
    reuse_limit: int = 5,
    downsample_size: Optional[int] = None,
    heldout_ratio: float = 0.3,
    test_ratio_within_heldout: float = 0.5,
):
    """
    Build the final fixed train/dev/test DatasetDict.

    This contains all formatting, random pair construction, shuffling and
    splitting. Training should not redo any of this.
    """

    if not dataset_names:
        raise ValueError("At least one dataset name must be supplied.")

    all_datasets = []

    for ds_name in dataset_names:
        formatted = format_custom_dataset(
            custom_dataset_name=ds_name,
            max_layers=max_layers,
            layer_type=layer_type,
            seed=seed,
            reuse_limit=reuse_limit,
        )

        if random_pairs and layer_type != "all":
            formatted = generate_training_pairs_random(
                formatted_dataset=formatted,
                seed=seed,
                n=reuse_limit,
            )

        ds = shuffle_and_transform_formatted_dataset(
            formatted_dataset=formatted,
            seed=seed,
        )

        all_datasets.append(ds)

    if len(all_datasets) == 1:
        merged = all_datasets[0]
    else:
        merged = concatenate_datasets(all_datasets)

    merged = merged.shuffle(seed=seed)

    if downsample_size is not None:
        merged = merged.select(range(min(downsample_size, len(merged))))

    if len(merged) < 3:
        raise ValueError(
            f"Dataset is too small after formatting/downsampling: {len(merged)} examples."
        )

    split = merged.train_test_split(
        test_size=heldout_ratio,
        seed=seed,
    )

    train_dataset = split["train"].shuffle(seed=seed)

    dev_test = split["test"].train_test_split(
        test_size=test_ratio_within_heldout,
        seed=seed,
    )

    dev_dataset = dev_test["train"].shuffle(seed=seed)
    test_dataset = dev_test["test"].shuffle(seed=seed)

    return DatasetDict(
        {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset,
        }
    )

def save_formatted_dataset_dict(
    dataset_dict: DatasetDict,
    output_path: str,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
):
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Formatted dataset already exists: {output_path}. "
                f"Pass --overwrite to replace it."
            )

        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset_dict.save_to_disk(output_path)

    if metadata is not None:
        metadata_path = os.path.join(output_path, "metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def load_formatted_dataset_dict(path: str) -> DatasetDict:
    dataset_dict = load_from_disk(path)

    required_splits = {"train", "dev", "test"}
    actual_splits = set(dataset_dict.keys())

    missing = required_splits - actual_splits

    if missing:
        raise ValueError(
            f"Saved dataset at {path} is missing required split(s): {sorted(missing)}"
        )

    return dataset_dict


def sample_reference_corpus(dict_list, reference_name, reference_size):
    """
    Sample dictionaries from a list where the 'collection' field equals reference_name,
    and remove both sampled dictionaries and related dictionaries based on collection_id.
    
    Args:
        dict_list: List of dictionaries with format {'collection', 'collection_id', 'text', 
                  'under_collection', 'under_collection_id'}.
        reference_name: The value to match with the 'collection' field.
        reference_size: Number of dictionaries to sample.
    
    Returns:
        A tuple of (sampled_list, remaining_list) where:
        - sampled_list: List of sampled dictionaries from the specified collection
        - remaining_list: Original list with sampled dictionaries and related dictionaries removed
    """
    # Filter the list to find all dictionaries where 'collection' equals reference_name
    candidates = [d for d in dict_list if d.get('collection') == reference_name]
    
    # Ensure we don't try to sample more than what's available
    sample_size = min(reference_size, len(candidates))
    
    # Randomly sample from the candidates
    sampled = random.sample(candidates, sample_size)
    
    # Get collection_ids of all sampled items
    sampled_collection_ids = {d.get('collection_id') for d in sampled}
    
    # Create a new list that excludes:
    # 1. The sampled dictionaries
    # 2. Dictionaries where 'under_collection_id' equals 'collection_id' of any sampled dictionary
    remaining = [d for d in dict_list if d not in sampled and 
                 d.get('under_collection_id') not in sampled_collection_ids]
    
    return sampled, remaining

