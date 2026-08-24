# This script has been co-created, refactored, and cleaned using GPT 5.6.
import random
from collections import defaultdict
from typing import Optional


def get_aligned_candidate_items(entry: dict) -> list[dict]:
    """Return one readable record per aligned candidate in a chain."""
    entry_id = entry["id"]
    text_label_pairs = list(entry["text_label_pairs"])
    if "candidate_ids" not in entry:
        raise ValueError(f"Entry {entry_id!r} is missing candidate_ids.")
    if "perturbation_sources" not in entry:
        raise ValueError(f"Entry {entry_id!r} is missing perturbation_sources.")

    candidate_ids = list(entry["candidate_ids"])
    perturbation_sources = list(entry["perturbation_sources"])
    item_score_dicts = list(
        entry.get("item_score_dicts", [{} for _ in text_label_pairs])
    )

    fields = {
        "text_label_pairs": text_label_pairs,
        "candidate_ids": candidate_ids,
        "perturbation_sources": perturbation_sources,
        "item_score_dicts": item_score_dicts,
    }
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Entry {entry_id!r} has misaligned candidate fields: {lengths}.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"Entry {entry_id!r} contains duplicate candidate_ids.")

    return [
        {
            "source_id": entry_id,
            "dataset_name": entry.get("dataset_name"),
            "source_original_ids": list(entry.get("source_original_ids", [])),
            "text": text,
            "label": label,
            "candidate_id": candidate_id,
            "perturbation_source": perturbation_source,
            "score_dict": score_dict,
        }
        for (text, label), candidate_id, perturbation_source, score_dict in zip(
            text_label_pairs,
            candidate_ids,
            perturbation_sources,
            item_score_dicts,
        )
    ]


def generate_training_pairs_random(
    formatted_dataset: list[dict],
    seed: Optional[int] = None,
    n: int = 5,
):
    """
    Construct random two-item pairwise examples.

    Score dictionaries are retained and remain aligned with the two selected
    texts. Pair selection itself still uses the existing perturbation-layer
    labels, preserving current pairwise behaviour.
    """
    rng = random.Random(seed)
    items = []

    for entry in formatted_dataset:
        items.extend(get_aligned_candidate_items(entry))

    if len(items) < 2:
        return []

    by_label = defaultdict(list)

    for index, item in enumerate(items):
        by_label[item["label"]].append(index)

    if len(by_label) < 2:
        return []

    usage = [0] * len(items)
    seen_pairs = set()
    result = []

    max_pairs = max(1, (len(items) * n) // 2)
    failed_attempts = 0
    max_failed_attempts = max(1000, max_pairs * 50)

    def available_indices_for_label(label):
        return [index for index in by_label[label] if usage[index] < n]

    while len(result) < max_pairs and failed_attempts < max_failed_attempts:
        available_labels = [
            label
            for label in by_label
            if available_indices_for_label(label)
        ]

        if len(available_labels) < 2:
            break

        label_a, label_b = rng.sample(available_labels, 2)
        candidates_a = available_indices_for_label(label_a)
        candidates_b = available_indices_for_label(label_b)

        if not candidates_a or not candidates_b:
            failed_attempts += 1
            continue

        index_a = rng.choice(candidates_a)
        index_b = rng.choice(candidates_b)

        if index_a == index_b:
            failed_attempts += 1
            continue

        pair_key = tuple(sorted((index_a, index_b)))

        if pair_key in seen_pairs:
            failed_attempts += 1
            continue

        seen_pairs.add(pair_key)

        item_a = items[index_a]
        item_b = items[index_b]
        usage[index_a] += 1
        usage[index_b] += 1

        source_original_ids = sorted(
            set(item_a["source_original_ids"])
            | set(item_b["source_original_ids"])
        )

        result.append(
            {
                "id": (
                    f"{item_a['source_id']}__{item_b['source_id']}__"
                    f"pair_{len(result)}"
                ),
                "dataset_name": item_a.get("dataset_name"),
                "source_original_ids": source_original_ids,
                "source_example_ids": [item_a["source_id"], item_b["source_id"]],
                "text_label_pairs": [
                    (item_a["text"], item_a["label"]),
                    (item_b["text"], item_b["label"]),
                ],
                "candidate_ids": [item_a["candidate_id"], item_b["candidate_id"]],
                "perturbation_sources": [
                    item_a["perturbation_source"],
                    item_b["perturbation_source"],
                ],
                "item_score_dicts": [
                    dict(item_a["score_dict"]),
                    dict(item_b["score_dict"]),
                ],
            }
        )

        failed_attempts = 0

    return result
