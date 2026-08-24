> This document has been co-created, refactored, and cleaned using GPT 5.6.

# Repository architecture

This project builds and evaluates **FE models**: encoder-based models that assign a scalar quality score to text. The same model supports pairwise ranking and scalar regression.

## The system in one view

```text
raw documents
  └─ scripts/ud_ds_scripts/                 create or regenerate source text
       └─ data/custom_datasets/<name>/      originals, perturbation layers, optional scores
            └─ clumsification_code/data/    align → split by source ID → group examples
                 └─ data/hf_datasets/<name>/  saved Hugging Face DatasetDict
                      └─ scripts/train_fe_model.py
                           └─ clumsification_code/fe/  model + ranking/regression training
                                └─ <output>/final/     encoder + tokenizer + fe_head.pt
                                     └─ clumsification_code/evals/  inference + benchmarks
```

The critical boundary is the formatted Hugging Face dataset. Everything before it creates supervision; everything after it trains or evaluates a scorer.

## Main folders

| Path | Responsibility |
| --- | --- |
| `clumsification_code/data/` | Reads custom JSONL datasets, aligns originals/perturbations/scores, prevents source-ID leakage, and saves train/dev/test datasets. |
| `clumsification_code/fe/` | Canonical FE implementation: shared encoder, evaluation head, ranking and regression objectives, trainers, evaluation, and checkpoints. |
| `clumsification_code/evals/` | Common benchmark runner, scorer adapters, benchmark loaders, metrics, and result writing. |
| `clumsification_code/perturbations/` | LLM-based and rule-based ways to create degraded or altered texts. |
| `scripts/` | User-facing workflows: dataset creation, training, HPO, calibration, scoring, and UD-document generation. |
| `filter_scripts/` | Standalone vLLM filters for deciding which generated texts to retain. |
| `clumsification_code/compat/`, `clumsification_code/ltr/` | Removable compatibility layer for old LTR names and checkpoint files. New code must not depend on these. |

## Canonical workflows

### 1. Build training data

`scripts/create_fe_training_dataset.py` is the entry point. It calls:

1. `data/splitting.py` to split original document IDs before examples are constructed, preventing document leakage.
2. `data/format_dataset.py` to align each original with perturbation layers and external score JSONLs. Score rows are matched by canonical perturbation source, original ID, and candidate ID.
3. `data/pairing.py` when random training pairs are requested.
4. `data/hf_dataset.py` to produce and save a `DatasetDict` containing `train`, `dev`, and `test`.

A formatted row normally contains `texts` and aligned `labels`; named score lists may also be present for regression.

#### Candidate and score identity

Every original or perturbation candidate has a stable `candidate_id`. This is
distinct from the source-document identity used for leakage-safe splitting.
The source identity answers “which original document does this belong to?”;
the candidate identity answers “which exact text instance is this?”

The canonical score-record identity is:

```text
(dataset_name, base_text_id, perturbation_source, candidate_id)
```

`candidate_id` must distinguish candidates that share an original, perturbation
source, and target layer. Layer is retained as descriptive metadata and for
ordering, but it is not by itself a unique candidate key. Score records also
retain `source_layer`, `target_layer`, `score_name`, and `score_value`.

The canonical perturbation-source values are `LLM` and `trad`. New code should
use `perturbation_source` rather than directory names. The raw input directory
is used only to select which perturbation files are scored; it is not written
as the candidate's identity.

Candidate IDs must survive formatting, shuffling, pairing, train/dev/test
serialization, and score lookup. Parallel arrays such as `texts`, `labels`,
and score lists must remain aligned with the corresponding `candidate_ids`.

There is intentionally no legacy score-file reader. Existing experimental
score directories are disposable and should be removed before producing
canonical scores.

### 1a. Produce scalar supervision (before dataset formatting)

`scripts/score_custom_dataset.py` scores candidate perturbations against their
source original and writes one JSONL file per method under
`data/custom_datasets/<dataset>/scores/`. It currently supports local
token-normalized perplexity, BERTScore F1, and BLEURT. Sampling always selects
original IDs first, then scores every perturbation belonging to each selected
original.

Successful score rows use the stable fields `base_text_id`, `source_layer`,
`target_layer`, `score_name`, and `score_value`, plus canonical
`perturbation_source` and `candidate_id` fields. Both perturbation sources
share one score file per scoring method. Originals are not self-scored
and therefore have no score row.  Failures are written separately to
`<score_name>.errors.jsonl`, while `<score_name>.metadata.json` records the
model, direction/transform, seed, selected IDs, and library versions.  All
stored score values use the convention **higher is better**: BERTScore F1 is
stored directly using Hugging Face Evaluate's normal defaults, BLEURT is stored
directly, while perplexity is transformed to `-log(perplexity)`.

When only part of a custom dataset has a given score, pass `--score-names` to
`scripts/create_fe_training_dataset.py`. The split is then made only over
original IDs with at least one requested score in the selected perturbation
folder(s), ensuring regression examples are distributed across train/dev/test.

### 2. Train an FE model

`scripts/train_fe_model.py` is the only canonical training entry point. `--training-method` selects:

- `pairwise`: grouped candidates, `GroupedRankingCollator`, pairwise losses, and `PairwiseFETrainer`;
- `regression`: independent text/target rows, `RegressionCollator`, and `RegressionFETrainer`.

Both paths use `fe/modeling.py::FEModel`: a Hugging Face encoder, mean pooling, and `EvaluationHead`. `fe/checkpointing.py` saves the encoder/tokenizer plus `fe_head.pt`. Old `ltr_head.pt` files are translated only through `compat/fe_checkpoints.py`.

`scripts/run_hpo.py` repeatedly invokes this trainer using trials from `scripts/configs/hpo/hps_to_test.py`. `scripts/inspired_calibration_ft.py` performs optional CoheSentia calibration on an existing FE checkpoint.

### 3. Evaluate

`python -m clumsification_code.evals.run_benchmark --scorer ...` is the shared entry point. It builds one adapter with a common `score_texts(...)` interface:

- `evals/inference/fe.py` for local FE checkpoints;
- `evals/inference/gptscore.py` for GPTScore;
- `evals/inference/metricx.py` and `evals/metricx24/` for MetricX;
- `evals/geval/` for G-Eval.

`evals/benchmark_runner.py` runs the datasets loaded by `benchmark_data.py`; `metrics.py` calculates correlations/preferences; `result_writer.py` records results. `evaluate_model_on_tdt_regens.py` is the larger specialized workflow for regenerated TDT/UD texts.

## Raw-data and perturbation conventions

Code expects project-relative data under `data/` (usually untracked):

```text
data/custom_datasets/<dataset>/
  original.jsonl              stable custom_id + text
  <perturbation outputs>      head_id links back to an original

data/hf_datasets/<dataset>/   saved train/dev/test DatasetDict
```

`perturbations/perturbation_methods.py` orchestrates generated perturbations. `evaleval_perturbations.py` contains traditional rules, while `rule_based_multilingual.py` contains the current UniMorph-based multilingual implementation. Files explicitly named `legacy` and old-name wrappers are not extension points.

## Where changes belong

| Change | Start here | Usually also check |
| --- | --- | --- |
| New dataset field or input format | `data/format_dataset.py` | `data/hf_dataset.py`, collators |
| New custom-dataset scoring method | `scoring/custom_dataset.py` | `scripts/score_custom_dataset.py`, score metadata |
| New split/pairing behavior | `data/splitting.py` or `data/pairing.py` | leakage assertions and metadata |
| New training objective | `fe/losses.py` or `fe/regression.py` | `FEModel.forward`, trainer, CLI args |
| Model architecture change | `fe/modeling.py` | checkpoint save/load and inference adapter |
| New evaluation backend | `evals/inference/` | `run_benchmark.build_scorer`, common scorer interface |
| New benchmark | `evals/benchmark_data.py` | `benchmark_runner.py`, metrics, result metadata |
| New perturbation | `perturbations/` | output schema expected by `format_dataset.py` |

## Maintenance rules

- Use **FE** for the shared scorer; use “pairwise ranking” only for that training objective.
- Keep reusable behavior under `clumsification_code/`; scripts should mainly parse arguments and orchestrate it.
- Preserve alignment among `texts`, `labels`, and every score list.
- Split by original/source ID before pairing or grouping to prevent leakage.
- Keep backward compatibility in a wrapper or `compat/`, never in the canonical implementation.
- When architecture or data flow changes, update this file in the same change.
