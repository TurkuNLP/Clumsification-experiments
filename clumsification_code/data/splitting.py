# This script has been co-created, refactored, and cleaned using GPT 5.6.
import random
from clumsification_code.data.format_dataset import _read_originals_by_custom_id

def split_original_ids_by_dataset(
    dataset_names: list[str],
    heldout_ratio: float = 0.3,
    test_ratio_within_heldout: float = 0.5,
    seed: int = 42,
) -> dict[str, dict[str, set[int]]]:
    """
    Split stable original-document IDs, not generated training examples.

    Returns:
        {
          "train": {"dataset_a": {1, 2, ...}, ...},
          "dev":   {"dataset_a": {...}, ...},
          "test":  {"dataset_a": {...}, ...},
        }
    """
    if not 0.0 < heldout_ratio < 1.0:
        raise ValueError(f"heldout_ratio must be between 0 and 1, got {heldout_ratio}")

    if not 0.0 < test_ratio_within_heldout < 1.0:
        raise ValueError(
            "test_ratio_within_heldout must be between 0 and 1, "
            f"got {test_ratio_within_heldout}"
        )

    groups: list[tuple[str, int]] = []

    for ds_name in dataset_names:
        original_ids = sorted(_read_originals_by_custom_id(ds_name).keys())
        groups.extend((ds_name, original_id) for original_id in original_ids)

    if len(groups) < 3:
        raise ValueError(
            f"Need at least 3 original documents for train/dev/test splitting; "
            f"got {len(groups)}."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_total = len(groups)

    n_heldout = int(round(n_total * heldout_ratio))
    n_heldout = max(2, min(n_total - 1, n_heldout))

    n_test = int(round(n_heldout * test_ratio_within_heldout))
    n_test = max(1, min(n_heldout - 1, n_test))

    n_dev = n_heldout - n_test

    test_groups = groups[:n_test]
    dev_groups = groups[n_test:n_test + n_dev]
    train_groups = groups[n_test + n_dev:]

    split_ids: dict[str, dict[str, set[int]]] = {
        split_name: {ds_name: set() for ds_name in dataset_names}
        for split_name in ("train", "dev", "test")
    }

    for ds_name, original_id in train_groups:
        split_ids["train"][ds_name].add(original_id)

    for ds_name, original_id in dev_groups:
        split_ids["dev"][ds_name].add(original_id)

    for ds_name, original_id in test_groups:
        split_ids["test"][ds_name].add(original_id)

    return split_ids


def split_ids_to_metadata(
    split_ids: dict[str, dict[str, set[int]]],
) -> dict[str, dict[str, list[int]]]:
    """JSON-serializable audit trail for the stable original IDs in each split."""
    return {
        split_name: {
            ds_name: sorted(original_ids)
            for ds_name, original_ids in by_dataset.items()
        }
        for split_name, by_dataset in split_ids.items()
    }


def assert_no_original_id_leakage(
    split_ids: dict[str, dict[str, set[int]]],
) -> None:
    """Fail fast if a dataset/original_id is assigned to multiple splits."""
    seen: dict[tuple[str, int], str] = {}

    for split_name, by_dataset in split_ids.items():
        for ds_name, original_ids in by_dataset.items():
            for original_id in original_ids:
                key = (ds_name, original_id)
                previous = seen.get(key)

                if previous is not None:
                    raise ValueError(
                        f"Original-document leakage: {ds_name}:{original_id} "
                        f"appears in both {previous!r} and {split_name!r}."
                    )

                seen[key] = split_name
