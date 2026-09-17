"""Prepare isolated FE pilot sources, assignments, and a reviewable coverage table."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.schemas import OriginalRecord
from clumsification_code.data.io import write_json_atomic, write_jsonl_atomic
from clumsification_code.perturbations.assignment_plan import LLMAssignment
from clumsification_code.perturbations.sampling import load_edit_catalog


def prepare_pilot(source, destination, catalog_path, seed=42):
    if destination.exists():
        raise FileExistsError(f"Pilot destination already exists: {destination}")
    originals = sorted(source.read_originals(), key=lambda r: (len(r.text), r.base_text_id))
    catalog = load_edit_catalog(catalog_path)
    rng = random.Random(seed)
    records, assignments, coverage = [], [], []

    def add(record, edits, severity, method, group, stratum):
        case = f"pilot-{len(records):04d}"
        dimensions = tuple(dict.fromkeys(d for edit in edits for d in edit.target_dimensions))
        metadata = {"pilot_source_id": record.base_text_id, "pilot_source_dataset": source.dataset_name,
                    "pilot_group": group, "pilot_length": stratum, "pilot_method": method,
                    "pilot_applicability": "requires human review"}
        records.append(OriginalRecord(destination.name, case, record.text, metadata))
        assignment = LLMAssignment(case, method, len(edits), tuple(e.edit_id for e in edits), severity, dimensions)
        assignments.append(assignment.to_row())
        coverage.append({"case_id": case, **metadata, "severity": severity, "edits": list(assignment.edits), "chars": len(record.text)})

    for index, edit in enumerate(catalog):
        eligible = [r for r in originals if len(r.text.split()) >= 40 and
                    ("Coherence" not in edit.target_dimensions or r.text.count(".") >= 3)]
        if not eligible:
            raise ValueError(f"No pilot sources for {edit.edit_id}")
        for label, fraction in (("short", .1), ("medium", .5), ("long", .95)):
            record = eligible[min(len(eligible)-1, int((len(eligible)-1)*fraction) + rng.randrange(min(64,len(eligible))))]
            for severity in ("weak", "medium", "strong"):
                add(record, [edit], severity, "llm_single", "single", label)

    combination_cursor = 0
    for count in range(2,6):
        eligible = [r for r in originals if len(r.text.replace("\n"," ")) >= count*500 and r.text.count(".") >= 2]
        if not eligible:
            raise ValueError(f"No eligible source for {count} edits")
        for label, fraction in (("short", .1), ("medium", .5), ("long", .95)):
            for example in range(2):
                record = eligible[min(len(eligible)-1,int((len(eligible)-1)*fraction)+example)]
                # Reuse edits as well as source across severities for meaningful triplets.
                edits = [catalog[(combination_cursor+i)%len(catalog)] for i in range(count)]
                combination_cursor += count
                for severity in ("weak","medium","strong"):
                    add(record, edits, severity, "llm_sampled", "combined", label)

    special = [originals[0], originals[-1]]
    for predicate in (lambda r:'\n' in r.text, lambda r:'"' in r.text,
                      lambda r:'Source text:' in r.text, lambda r:'<' in r.text):
        special.append(next((r for r in originals if predicate(r)), originals[len(originals)//2]))
    special += [originals[int(len(originals)*q)] for q in (.3,.6)]
    for index, record in enumerate(special):
        edit = catalog[index % len(catalog)]
        for severity in ("weak","medium","strong"):
            add(record,[edit],severity,"llm_single","stress",f"stress-{index}")

    repository = DatasetRepository(destination)
    repository.write_originals(records)
    write_jsonl_atomic(destination / "perturbation_assignments.jsonl", assignments)
    write_jsonl_atomic(destination / "split_assignments.jsonl", [
        {"base_text_id":r.base_text_id, "split":r.metadata["pilot_method"]} for r in records])
    write_jsonl_atomic(destination / "pilot_cases.jsonl", coverage)
    report = {"source_dataset": source.dataset_name, "source_count": len(originals), "seed": seed,
              "case_count":len(records), "operation_count":len(catalog),
              "method_counts":dict(Counter(r.metadata["pilot_method"] for r in records)),
              "catalog_sha256":hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
              "notes":"Length strata are character-based at preparation; tokenizer profiling supplies exact bucket boundaries. Applicability and severity need human review."}
    write_json_atomic(destination / "pilot_manifest.json", report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="fe-dataset-final")
    parser.add_argument("--dataset-root",type=Path,default=Path("data/custom_datasets"))
    parser.add_argument("--pilot-name",default="fe-pilot-v3")
    parser.add_argument("--catalog",type=Path,default=Path("data/perturbation_prompts/english/edit_types.jsonl"))
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(prepare_pilot(DatasetRepository.from_root(args.dataset_root,args.dataset),
                                  args.dataset_root/args.pilot_name,args.catalog,args.seed),indent=2))


if __name__ == "__main__":
    main()
