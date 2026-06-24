import os
from io import read_ds
from pairing import generate_training_pairs_random
from typing import Optional
import json
import random

def _coerce_custom_id(value, *, context: str) -> int:
    """custom_id/head_id is stored as a string in several JSONL files; normalize it."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer id in {context}: {value!r}") from exc


def _read_originals_by_custom_id(custom_dataset_name: str) -> dict[int, dict]:
    work_path = os.path.join("data", "custom_datasets", custom_dataset_name)
    original_path = os.path.join(work_path, "original.jsonl")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Missing original dataset file: {original_path}")

    originals_by_id: dict[int, dict] = {}

    for row_no, row in enumerate(read_ds(original_path), start=1):
        if "custom_id" not in row:
            raise ValueError(f"{original_path}:{row_no} is missing custom_id")
        if "text" not in row:
            raise ValueError(f"{original_path}:{row_no} is missing text")

        custom_id = _coerce_custom_id(
            row["custom_id"],
            context=f"{original_path}:{row_no}:custom_id",
        )

        if custom_id in originals_by_id:
            raise ValueError(
                f"Duplicate custom_id={custom_id} in {original_path}; "
                "custom_id must be unique and stable."
            )

        originals_by_id[custom_id] = row

    return originals_by_id

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


def format_custom_dataset(
    custom_dataset_name: str,
    max_layers: Optional[int] = None,
    layer_type: str = "clumsy",
    seed: int = 42,
    reuse_limit: int = 5,
    original_id_filter: Optional[set[int]] = None,
):
    """
    Load one raw custom dataset and return a list of dictionaries:

      {
        "id": ...,
        "dataset_name": ...,
        "source_original_ids": [...],
        "text_label_pairs": [(text, label), ...],
      }

    The canonical document identity is original.jsonl::custom_id.

    If original_id_filter is supplied, only those originals are formatted.
    This is what prevents train/dev/test leakage when called after group
    splitting.
    """
    work_path = os.path.join("data", "custom_datasets", custom_dataset_name)

    originals_by_id = _read_originals_by_custom_id(custom_dataset_name)
    all_original_ids = set(originals_by_id)

    if original_id_filter is not None:
        unknown = set(original_id_filter) - all_original_ids
        if unknown:
            raise ValueError(
                f"{custom_dataset_name}: original_id_filter contains IDs not "
                f"present in original.jsonl: {sorted(unknown)[:20]}"
            )

        selected_ids = set(original_id_filter)
    else:
        selected_ids = set(all_original_ids)

    def _base_record(original_id: int) -> dict:
        return {
            "id": f"{custom_dataset_name}__orig__{original_id}",
            "dataset_name": custom_dataset_name,
            "source_original_ids": [original_id],
            "text_label_pairs": [(originals_by_id[original_id]["text"], 0)],
        }

    def _build_id_dict(
        layer_dir: str,
        id_filter: Optional[set[int]] = None,
    ):
        active_ids = selected_ids if id_filter is None else selected_ids & set(id_filter)

        id_dict = {
            original_id: _base_record(original_id)
            for original_id in sorted(active_ids)
        }

        layer_path = os.path.join(work_path, layer_dir)

        if not os.path.isdir(layer_path):
            raise FileNotFoundError(f"Missing perturbation layer directory: {layer_path}")

        missing_head_ids: set[int] = set()

        for file_name in os.listdir(layer_path):
            if not file_name.endswith(".jsonl"):
                continue

            try:
                layer = int(file_name.replace(".jsonl", ""))
            except ValueError:
                continue

            if max_layers is not None and layer >= max_layers:
                continue

            file_path = os.path.join(layer_path, file_name)
            contents = read_ds(file_path)

            for row_no, x in enumerate(contents, start=1):
                if "head_id" not in x:
                    raise ValueError(f"{file_path}:{row_no} is missing head_id")

                if "text" not in x:
                    raise ValueError(f"{file_path}:{row_no} is missing text")

                head_id = _coerce_custom_id(
                    x["head_id"],
                    context=f"{file_path}:{row_no}:head_id",
                )

                # Missing from original.jsonl is a schema/data-integrity error.
                if head_id not in all_original_ids:
                    missing_head_ids.add(head_id)
                    continue

                # Present in original.jsonl but not selected for this split/layer
                # subtype is expected; just do not include it.
                if head_id not in id_dict:
                    continue

                id_dict[head_id]["text_label_pairs"].append((x["text"], layer))

        if missing_head_ids:
            raise ValueError(
                f"{custom_dataset_name}/{layer_dir}: perturbation head_id values "
                f"not found in original.jsonl custom_id set: "
                f"{sorted(missing_head_ids)[:50]}"
            )

        return id_dict

    def _prefix_records(records: list[dict], prefix: str) -> list[dict]:
        prefixed = []

        for entry in records:
            new_entry = dict(entry)
            new_entry["id"] = f"{custom_dataset_name}__{prefix}__{entry['id']}"
            prefixed.append(new_entry)

        return prefixed

    if layer_type == "clumsy":
        return list(_build_id_dict("perturbed_layers").values())

    if layer_type == "trad":
        return list(_build_id_dict("trad_perturbed_layers").values())

    if layer_type == "mix":
        split_candidate_ids = sorted(selected_ids)

        rng = random.Random(seed)
        rng.shuffle(split_candidate_ids)

        mid = len(split_candidate_ids) // 2

        clumsy_ids = set(split_candidate_ids[:mid])
        trad_ids = set(split_candidate_ids[mid:])

        clumsy_dict = _build_id_dict("perturbed_layers", id_filter=clumsy_ids)
        trad_dict = _build_id_dict("trad_perturbed_layers", id_filter=trad_ids)

        return list(clumsy_dict.values()) + list(trad_dict.values())

    if layer_type == "all":
        clumsy_chains = _prefix_records(
            list(_build_id_dict("perturbed_layers").values()),
            "clumsy_chain",
        )

        trad_chains = _prefix_records(
            list(_build_id_dict("trad_perturbed_layers").values()),
            "trad_chain",
        )

        # Important: these pairs are now generated only from the selected
        # split's originals.
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

        clumsy_pairs = _prefix_records(clumsy_pairs, "clumsy_pair")
        trad_pairs = _prefix_records(trad_pairs, "trad_pair")

        return clumsy_chains + trad_chains + clumsy_pairs + trad_pairs

    raise ValueError(f"Unknown layer_type: {layer_type!r}")
