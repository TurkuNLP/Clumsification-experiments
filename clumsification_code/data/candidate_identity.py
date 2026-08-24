# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Canonical identity helpers for original and perturbation candidates."""


SOURCE_BY_LAYER_DIRECTORY = {
    "perturbed_layers": "LLM",
    "trad_perturbed_layers": "trad",
}
VALID_PERTURBATION_SOURCES = {"LLM", "trad", "original"}


def canonical_perturbation_source(layer_directory: str) -> str:
    """Map a raw perturbation directory to its canonical source label."""
    try:
        return SOURCE_BY_LAYER_DIRECTORY[layer_directory]
    except KeyError as exc:
        raise ValueError(
            "layer_directory must be 'perturbed_layers' or "
            f"'trad_perturbed_layers', got {layer_directory!r}."
        ) from exc


def make_candidate_id(
    *,
    perturbation_source: str,
    base_text_id: int,
    target_layer: int,
    candidate_index: int,
) -> str:
    """Create a readable, deterministic ID for one candidate text."""
    return (
        f"{perturbation_source}__base_{base_text_id}__layer_{target_layer}__"
        f"candidate_{candidate_index:06d}"
    )


def make_original_candidate_id(*, dataset_name: str, base_text_id: int) -> str:
    """Create the stable candidate ID for an unperturbed original text."""
    return f"original__{dataset_name}__base_{base_text_id}"


def candidate_id_from_raw_row(
    row: dict,
    *,
    perturbation_source: str,
    base_text_id: int,
    target_layer: int,
    candidate_index: int,
) -> str:
    """Read an explicit raw ID, with a temporary deterministic fallback."""
    explicit_id = row.get("candidate_id")
    if explicit_id is not None:
        if not isinstance(explicit_id, str) or not explicit_id:
            raise ValueError("candidate_id must be a non-empty string when supplied.")
        return explicit_id

    return make_candidate_id(
        perturbation_source=perturbation_source,
        base_text_id=base_text_id,
        target_layer=target_layer,
        candidate_index=candidate_index,
    )


def add_candidate_ids_to_rows(
    rows: list[dict],
    *,
    perturbation_source: str,
    target_layer: int,
) -> list[dict]:
    """Add stable IDs to newly generated rows before they are written."""
    next_index: dict[int, int] = {}
    identified_rows = []

    for row in rows:
        base_text_id = int(row["head_id"])
        candidate_index = next_index.get(base_text_id, 0)
        next_index[base_text_id] = candidate_index + 1
        identified_row = dict(row)
        identified_row["candidate_id"] = make_candidate_id(
            perturbation_source=perturbation_source,
            base_text_id=base_text_id,
            target_layer=target_layer,
            candidate_index=candidate_index,
        )
        identified_rows.append(identified_row)

    return identified_rows
