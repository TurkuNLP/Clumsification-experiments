# This script has been co-created, refactored, and cleaned using GPT 5.6.
import json
import math
import numbers
import os
import random
from pathlib import Path
from typing import Optional

from clumsification_code.data.io import read_ds
from clumsification_code.data.pairing import generate_training_pairs_random
from clumsification_code.data.candidate_identity import (
    candidate_id_from_raw_row,
    canonical_perturbation_source,
    make_original_candidate_id,
    VALID_PERTURBATION_SOURCES,
)


def _coerce_custom_id(value, *, context: str) -> int:
    """Normalize custom_id/head_id values stored as strings in JSONL files."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer id in {context}: {value!r}") from exc


def _extract_numeric_scores(
    row: dict,
    *,
    excluded_fields: set[str],
) -> dict[str, float | None]:
    """
    Extract candidate score fields from a raw item.

    Scores are expected to be top-level scalar numeric fields, for example
    {"method_1": 0.71}. Identifier and text fields must be supplied in
    excluded_fields. Missing values are retained as None so score-list
    alignment is preserved in the formatted dataset.
    """
    scores: dict[str, float | None] = {}

    for field_name, value in row.items():
        if field_name in excluded_fields or isinstance(value, bool):
            continue

        if value is None:
            scores[field_name] = None
        elif isinstance(value, numbers.Real) and math.isfinite(float(value)):
            scores[field_name] = float(value)

    return scores


def _read_originals_by_custom_id(custom_dataset_name: str) -> dict[int, dict]:
    work_path = os.path.join("data", "custom_datasets", custom_dataset_name)
    original_path = os.path.join(work_path, "original.jsonl")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Missing original dataset file: {original_path}")

    originals_by_id: dict[int, dict] = {}

    for row_no, row in enumerate(read_ds(original_path), start=1):
        # Older custom datasets used head_id for originals as well as their
        # perturbations. Accept that schema without rewriting the source data.
        # New data should continue to use custom_id.
        if "custom_id" in row and "head_id" in row:
            custom_id = _coerce_custom_id(
                row["custom_id"],
                context=f"{original_path}:{row_no}:custom_id",
            )
            legacy_head_id = _coerce_custom_id(
                row["head_id"],
                context=f"{original_path}:{row_no}:head_id",
            )
            if custom_id != legacy_head_id:
                raise ValueError(
                    f"{original_path}:{row_no} has conflicting custom_id="
                    f"{custom_id} and head_id={legacy_head_id}"
                )
        elif "custom_id" in row:
            custom_id = _coerce_custom_id(
                row["custom_id"],
                context=f"{original_path}:{row_no}:custom_id",
            )
        elif "head_id" in row:
            custom_id = _coerce_custom_id(
                row["head_id"],
                context=f"{original_path}:{row_no}:head_id",
            )
        else:
            raise ValueError(
                f"{original_path}:{row_no} is missing custom_id "
                "(legacy head_id is also accepted)"
            )
        if "text" not in row:
            raise ValueError(f"{original_path}:{row_no} is missing text")

        if custom_id in originals_by_id:
            raise ValueError(
                f"Duplicate custom_id={custom_id} in {original_path}; "
                "custom_id must be unique and stable."
            )

        originals_by_id[custom_id] = row

    return originals_by_id


def load_external_scores(
    custom_dataset_name: str,
) -> dict[tuple[str, int, str], dict[str, float]]:
    """Read canonical score files keyed by source, original, and candidate ID."""
    score_root = Path("data") / "custom_datasets" / custom_dataset_name / "scores"
    if not score_root.is_dir():
        return {}

    scores: dict[tuple[str, int, str], dict[str, float]] = {}
    for score_path in sorted(score_root.glob("*.jsonl")):
        if score_path.name.endswith(".errors.jsonl"):
            continue

        for row_no, row in enumerate(read_ds(str(score_path)), start=1):
            required_fields = {
                "base_text_id",
                "perturbation_source",
                "candidate_id",
                "source_layer",
                "target_layer",
                "score_name",
                "score_value",
            }
            missing_fields = required_fields - set(row)
            if missing_fields:
                raise ValueError(
                    f"{score_path}:{row_no} is missing score field(s): "
                    f"{sorted(missing_fields)}. Re-run scoring to produce the "
                    "current score schema."
                )

            perturbation_source = row["perturbation_source"]
            candidate_id = row["candidate_id"]
            score_name = row["score_name"]
            if perturbation_source not in VALID_PERTURBATION_SOURCES:
                raise ValueError(
                    f"{score_path}:{row_no}: invalid perturbation_source "
                    f"{perturbation_source!r}; expected one of "
                    f"{sorted(VALID_PERTURBATION_SOURCES)}."
                )
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"{score_path}:{row_no}: candidate_id must be a string.")
            if not isinstance(score_name, str) or not score_name:
                raise ValueError(f"{score_path}:{row_no}: score_name must be a string.")
            source_layer = _coerce_custom_id(
                row["source_layer"], context=f"{score_path}:{row_no}:source_layer"
            )
            if source_layer != 0:
                raise ValueError(
                    f"{score_path}:{row_no}: source_layer must be 0 when scores "
                    "are aligned to original texts."
                )
            base_text_id = _coerce_custom_id(
                row["base_text_id"], context=f"{score_path}:{row_no}:base_text_id"
            )
            target_layer = _coerce_custom_id(
                row["target_layer"], context=f"{score_path}:{row_no}:target_layer"
            )
            score_value = row["score_value"]
            if isinstance(score_value, bool) or not isinstance(score_value, numbers.Real):
                raise ValueError(f"{score_path}:{row_no}: score_value must be numeric.")
            if not math.isfinite(float(score_value)):
                raise ValueError(f"{score_path}:{row_no}: score_value must be finite.")

            key = (perturbation_source, base_text_id, candidate_id)
            score_dict = scores.setdefault(key, {})
            if score_name in score_dict:
                raise ValueError(
                    f"Duplicate score {score_name!r} for {key} in {score_path}:{row_no}."
                )
            score_dict[score_name] = float(score_value)

    return scores


def scored_original_ids(
    custom_dataset_name: str,
    *,
    layer_folders: set[str],
    score_names: Optional[set[str]] = None,
) -> set[int]:
    """Return source IDs that have at least one requested usable score."""
    external_scores = load_external_scores(custom_dataset_name)
    requested_sources = {
        canonical_perturbation_source(folder) for folder in layer_folders
    }
    requested_sources.add("original")
    return {
        original_id
        for (perturbation_source, original_id, _), score_dict in external_scores.items()
        if perturbation_source in requested_sources
        and (score_names is None or score_names & set(score_dict))
    }


# Historical generic formatter retained for callers outside the FE pipeline.
def format_datasets(dss: list[dict[str]]):
    ds_items = []

    for entry in dss:
        ds_path = entry["ds_path"]
        ds_name = entry["ds_name"]
        ds_under = entry["under_ds_name"]

        with open(ds_path, "r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                if not line:
                    continue

                row = json.loads(line.strip())

                if ds_under:
                    ds_items.append(
                        {
                            "collection": ds_name,
                            "collection_id": f"{ds_name}_{index}",
                            "text": row["text"],
                            "under_collection": ds_under,
                            "under_collection_id": row["custom_id"],
                        }
                    )
                else:
                    ds_items.append(
                        {
                            "collection": ds_name,
                            "collection_id": f"{ds_name}_{index}",
                            "text": row["text"],
                            "under_collection": None,
                            "under_collection_id": None,
                        }
                    )

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
    Load one raw custom dataset into chain records.

    Each resulting record has aligned lists:
      - text_label_pairs: [(text, perturbation_layer), ...]
      - item_score_dicts: [{score_name: score_or_none, ...}, ...]
      - candidate_ids: [stable candidate ID, ...]
      - perturbation_sources: [source label, ...]

    The first item is the original text with layer label 0. It is retained even
    when it has no score. Regression later selects only items whose chosen
    score is valid; pairwise training continues using layer labels.
    """
    work_path = os.path.join("data", "custom_datasets", custom_dataset_name)
    originals_by_id = _read_originals_by_custom_id(custom_dataset_name)
    external_scores = load_external_scores(custom_dataset_name)
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
        original = originals_by_id[original_id]
        original_candidate_id = make_original_candidate_id(
            dataset_name=custom_dataset_name,
            base_text_id=original_id,
        )
        original_scores = _extract_numeric_scores(
            original,
            excluded_fields={"custom_id", "head_id", "text"},
        )
        for score_name, score_value in external_scores.get(
            ("original", original_id, original_candidate_id), {}
        ).items():
            if score_name in original_scores:
                raise ValueError(
                    f"{custom_dataset_name}/original.jsonl contains score "
                    f"{score_name!r} both inline and in the external score file."
                )
            original_scores[score_name] = score_value

        return {
            "id": f"{custom_dataset_name}__orig__{original_id}",
            "dataset_name": custom_dataset_name,
            "source_original_ids": [original_id],
            "text_label_pairs": [(original["text"], 0)],
            "candidate_ids": [
                original_candidate_id
            ],
            "perturbation_sources": ["original"],
            "item_score_dicts": [original_scores],
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
            raise FileNotFoundError(
                f"Missing perturbation layer directory: {layer_path}"
            )

        missing_head_ids: set[int] = set()

        perturbation_source = canonical_perturbation_source(layer_dir)
        candidate_indices: dict[tuple[int, int], int] = {}

        for file_name in sorted(os.listdir(layer_path)):
            if not file_name.endswith(".jsonl"):
                continue

            try:
                layer = int(file_name.replace(".jsonl", ""))
            except ValueError:
                continue

            if max_layers is not None and layer >= max_layers:
                continue

            file_path = os.path.join(layer_path, file_name)

            for row_no, row in enumerate(read_ds(file_path), start=1):
                if "head_id" not in row:
                    raise ValueError(f"{file_path}:{row_no} is missing head_id")
                if "text" not in row:
                    raise ValueError(f"{file_path}:{row_no} is missing text")

                head_id = _coerce_custom_id(
                    row["head_id"],
                    context=f"{file_path}:{row_no}:head_id",
                )

                if head_id not in all_original_ids:
                    missing_head_ids.add(head_id)
                    continue

                if head_id not in id_dict:
                    continue

                candidate_key = (head_id, layer)
                candidate_index = candidate_indices.get(candidate_key, 0)
                candidate_indices[candidate_key] = candidate_index + 1
                candidate_id = candidate_id_from_raw_row(
                    row,
                    perturbation_source=perturbation_source,
                    base_text_id=head_id,
                    target_layer=layer,
                    candidate_index=candidate_index,
                )
                id_dict[head_id]["text_label_pairs"].append((row["text"], layer))
                id_dict[head_id]["candidate_ids"].append(candidate_id)
                id_dict[head_id]["perturbation_sources"].append(perturbation_source)
                item_scores = _extract_numeric_scores(
                    row,
                    excluded_fields={"head_id", "text"},
                )
                score_key = (perturbation_source, head_id, candidate_id)
                for score_name, score_value in external_scores.get(score_key, {}).items():
                    if score_name in item_scores:
                        raise ValueError(
                            f"{custom_dataset_name}/{layer_dir}/{file_name}:{row_no} "
                            f"contains score {score_name!r} both in the text row and "
                            "the external score file."
                        )
                    item_scores[score_name] = score_value
                id_dict[head_id]["item_score_dicts"].append(item_scores)

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

        midpoint = len(split_candidate_ids) // 2
        clumsy_ids = set(split_candidate_ids[:midpoint])
        trad_ids = set(split_candidate_ids[midpoint:])

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
