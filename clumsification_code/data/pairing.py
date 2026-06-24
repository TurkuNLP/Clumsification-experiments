import random
from typing import Optional
from collections import defaultdict


def generate_training_pairs_random(
    formatted_dataset: list[dict],
    seed: Optional[int] = None,
    n: int = 5,
):
    """
    Construct random two-item training examples without materializing all
    O(num_items^2) possible pairs.

    Each individual text may be reused at most n times.

    The caller is responsible for passing only one train/dev/test split at a
    time. create_formatted_dataset_dict() now does that.
    """
    rng = random.Random(seed)

    items = []

    for entry in formatted_dataset:
        entry_id = entry["id"]
        source_original_ids = list(entry.get("source_original_ids", []))
        dataset_name = entry.get("dataset_name")

        for text, label in entry["text_label_pairs"]:
            items.append(
                {
                    "source_id": entry_id,
                    "dataset_name": dataset_name,
                    "source_original_ids": source_original_ids,
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

        combined_id = (
            f"{item_i['source_id']}__{item_j['source_id']}__pair_{len(result)}"
        )

        source_original_ids = sorted(
            set(item_i["source_original_ids"])
            | set(item_j["source_original_ids"])
        )

        result.append(
            {
                "id": combined_id,
                "dataset_name": item_i.get("dataset_name"),
                "source_original_ids": source_original_ids,
                "source_example_ids": [item_i["source_id"], item_j["source_id"]],
                "text_label_pairs": [
                    (item_i["text"], item_i["label"]),
                    (item_j["text"], item_j["label"]),
                ],
            }
        )

        failed_attempts = 0

    return result