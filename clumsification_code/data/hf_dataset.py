from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import random
from typing import Optional
from format_dataset import *
from splitting import *
import shutil


def shuffle_and_transform_formatted_dataset(
    formatted_dataset: list[dict],
    seed: Optional[int] = None,
):
    """
    Convert list records of this form:

        {
          "id": ...,
          "source_original_ids": [...],
          "text_label_pairs": [(text, label), ...],
        }

    into a Hugging Face Dataset with this form:

        {
          "id": ...,
          "source_original_ids": [...],
          "texts": [...],
          "labels": [...],
        }
    """
    rng = random.Random(seed)
    rows = []

    for entry in formatted_dataset:
        tl_list = list(entry.get("text_label_pairs", []))

        if len(tl_list) < 2:
            # Pairwise/listwise training needs at least two items.
            continue

        rng.shuffle(tl_list)

        rows.append(
            {
                "id": str(entry["id"]),
                "dataset_name": entry.get("dataset_name"),
                "source_original_ids": list(entry.get("source_original_ids", [])),
                "texts": [x[0] for x in tl_list],
                "labels": [x[1] for x in tl_list],
            }
        )

    return Dataset.from_list(rows)

        
def _concat_nonempty_datasets(datasets: list[Dataset]) -> Dataset:
    nonempty = [ds for ds in datasets if len(ds) > 0]

    if not nonempty:
        return Dataset.from_list([])

    if len(nonempty) == 1:
        return nonempty[0]

    return concatenate_datasets(nonempty)


def _downsample_dataset_dict(
    dataset_dict: DatasetDict,
    downsample_size: int,
    seed: int,
) -> DatasetDict:
    total = sum(len(dataset_dict[name]) for name in ("train", "dev", "test"))

    if downsample_size >= total:
        return dataset_dict

    if downsample_size < 3:
        raise ValueError("downsample_size must be at least 3 to keep train/dev/test.")

    split_names = ("train", "dev", "test")

    quotas = {
        name: max(1, int(round(downsample_size * len(dataset_dict[name]) / total)))
        for name in split_names
    }

    # Fix rounding while preserving at least one example per split.
    while sum(quotas.values()) > downsample_size:
        candidates = [name for name in split_names if quotas[name] > 1]
        name = max(candidates, key=lambda n: quotas[n])
        quotas[name] -= 1

    while sum(quotas.values()) < downsample_size:
        candidates = [
            name
            for name in split_names
            if quotas[name] < len(dataset_dict[name])
        ]

        if not candidates:
            break

        name = max(candidates, key=lambda n: len(dataset_dict[n]) - quotas[n])
        quotas[name] += 1

    return DatasetDict(
        {
            name: dataset_dict[name].shuffle(seed=seed).select(range(quotas[name]))
            for name in split_names
        }
    )


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
    return_metadata: bool = False,
):
    """
    Build the final fixed train/dev/test DatasetDict.

    IMPORTANT: train/dev/test are split by stable original-document custom_id
    before chains and random pairs are created. This prevents perturbations of
    the same original text from crossing split boundaries.
    """
    if not dataset_names:
        raise ValueError("At least one dataset name must be supplied.")

    split_ids = split_original_ids_by_dataset(
        dataset_names=dataset_names,
        heldout_ratio=heldout_ratio,
        test_ratio_within_heldout=test_ratio_within_heldout,
        seed=seed,
    )

    assert_no_original_id_leakage(split_ids)

    final_splits: dict[str, Dataset] = {}

    for split_name in ("train", "dev", "test"):
        split_parts = []

        for ds_name in dataset_names:
            formatted = format_custom_dataset(
                custom_dataset_name=ds_name,
                max_layers=max_layers,
                layer_type=layer_type,
                seed=seed,
                reuse_limit=reuse_limit,
                original_id_filter=split_ids[split_name][ds_name],
            )

            # For layer_type="all", format_custom_dataset already creates the
            # historical extra random pairs, but now only inside this split.
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

            if len(ds) > 0:
                split_parts.append(ds)

        split_dataset = _concat_nonempty_datasets(split_parts).shuffle(seed=seed)
        final_splits[split_name] = split_dataset

    if any(len(final_splits[name]) == 0 for name in ("train", "dev", "test")):
        sizes = {
            name: len(final_splits[name])
            for name in ("train", "dev", "test")
        }

        raise ValueError(
            "At least one split has no usable formatted examples after "
            f"chain/pair construction: {sizes}"
        )

    dataset_dict = DatasetDict(final_splits)

    if downsample_size is not None:
        dataset_dict = _downsample_dataset_dict(
            dataset_dict=dataset_dict,
            downsample_size=downsample_size,
            seed=seed,
        )

    metadata = {
        "split_strategy": "original_custom_id_group_split_before_formatting",
        "split_original_ids": split_ids_to_metadata(split_ids),
        "num_original_ids": {
            split_name: {
                ds_name: len(ids)
                for ds_name, ids in by_dataset.items()
            }
            for split_name, by_dataset in split_ids.items()
        },
    }

    if return_metadata:
        return dataset_dict, metadata

    return dataset_dict

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