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
                                └─ <output>/final/     encoder + tokenizer + FE metadata/state
                                     └─ clumsification_code/evals/  inference + benchmarks

Retired FE training and chain-evaluation implementations are kept locally in
the ignored `clumsification_code/.archive/fe_legacy/` directory. They are not
part of production imports or training.
```

The formatted Hugging Face dataset is the key boundary: upstream code creates
supervision, while downstream code trains or evaluates scorers.

## FE contracts (implementation target)

These are the non-negotiable contracts for the simplified FE implementation:

1. **Candidate-only scoring.** A trained evaluator accepts one candidate text
   and returns one higher-is-better scalar. Sources, references, perturbation
   layers, teacher names, and human labels are never model inputs at inference.
2. **One scorer, two objectives.** Both training methods use the same function
   `s(x) = w^T pool_backbone(encoder(x), attention_mask) + b`. The head is
   exactly one linear projection from the pooled representation to one scalar.
   Pooling is resolved from the backbone name: E5 uses masked mean pooling;
   Jina v5, Qwen3 Embedding, and Harrier use the last non-padding token. The
   resolved policy is stored in each checkpoint.
3. **Source-safe flattening.** Chains are an upstream provenance and auditing
   representation. Splits are made by source/original ID first; only then are
   rows flattened for training.
4. **Explicit training rows.** Regression consumes one candidate and one
   scalar target per row. Pairwise ranking consumes two independently scored
   candidates from the same source per row. The trainer does not construct
   pairs from variable-length chains.
5. **Standard training engine.** The canonical implementation uses the
   Hugging Face `Trainer`; objective-specific behavior belongs in the model,
   loss functions, collators, and metric functions. Custom trainer subclasses
   are not part of the target architecture.
6. **Training-only supervision.** Automatic teacher scores may create targets,
   but human evaluation data remains evaluation-only and cannot affect splits,
   transforms, checkpoint selection, or hyperparameters.

## Main folders

| Path | Responsibility |
| --- | --- |
| `clumsification_code/data/` | Reads custom JSONL datasets, aligns originals/perturbations/scores, prevents source-ID leakage, and saves train/dev/test datasets. |
| `clumsification_code/fe/` | Canonical FE implementation: backbone profiles and pooling, shared encoder, linear scalar head, objectives, collators, metrics, diagnostics, and checkpoints. |
| `clumsification_code/evals/` | Common benchmark runner, scorer adapters, benchmark loaders, metrics, and result writing. |
| `clumsification_code/perturbations/` | LLM-based and rule-based ways to create degraded or altered texts. |
| `clumsification_code/prompts/`, `data/prompts/` | Validates and renders versioned prompt specifications stored independently from model transports and scorer code. |
| `scripts/` | User-facing workflows: dataset creation, training, HPO, calibration, scoring, and UD-document generation. |
| `filter_scripts/` | Standalone vLLM filters for deciding which generated texts to retain. |
| `clumsification_code/compat/` | Compatibility helpers for legacy checkpoint files. Production code does not depend on LTR names. |

## Canonical workflows

### 0. Audit UniEval pseudo-data

The released UniEval JSONL files are kept immutable under
`data/unprocessed_datasets/unieval_pseudo_data/`.  Before training, run:

```bash
/home/tenojo/miniconda3/envs/genAI/bin/python scripts/audit_unieval_dataset.py
```

This streams and validates every row, records SHA-256 hashes, label counts,
duplicate inputs, contradictory labels, and source-text length statistics.  It
writes `unieval_pseudo_data.audit.json` beside (not inside) the source files.
The audit deliberately does not clean or deduplicate data: those operations
must be explicit experimental ablations.  The current imported release
contains 665,523 rows when both dimension files and merged `train_all` files
are counted, with 2,899 inputs appearing under both labels.  The merged files
overlap their component dimension files and must not be summed as additional
training data.

The reusable implementation is in `clumsification_code/unieval/data.py` and
can be used by later training and SEScore2 adapters.

### 1. Build training data

`scripts/create_fe_training_dataset.py` is the entry point. It calls:

1. `data/splitting.py` to split original document IDs before examples are constructed, preventing document leakage.
2. `data/format_dataset.py` to align each original with perturbation layers and external score JSONLs. Score rows are matched by canonical perturbation source, original ID, and candidate ID.
3. `data/pairing.py` when random training pairs are requested.
4. `data/hf_dataset.py` to produce and save a `DatasetDict` containing `train`, `dev`, and `test`.

At training time, `data/flattening.py` validates source isolation and converts
the grouped dataset into explicit regression rows or chosen/rejected pair
rows. The grouped dataset is retained for provenance; the flattened dataset is
what the trainer consumes.

A formatted row normally contains `texts` and aligned `labels`; named score lists may also be present for regression.

#### Candidate and score identity

Every original or perturbation candidate has a stable `candidate_id`, separate
from the source-document identity used for leakage-safe splitting. The source
ID identifies the document; the candidate ID identifies the exact text.

The canonical score-record identity is:

```text
(dataset_name, base_text_id, perturbation_source, candidate_id)
```

`candidate_id` distinguishes candidates that share an original, perturbation
source, or target layer. Layer remains descriptive metadata and is not a
unique key. Score records also retain `source_layer`, `target_layer`,
`score_name`, and `score_value`.

The canonical perturbation-source values are `LLM` and `trad`. New code should
use `perturbation_source` rather than directory names. The raw input directory
is used only to select which perturbation files are scored; it is not written
as the candidate's identity.

Candidate IDs must survive formatting, shuffling, pairing, train/dev/test
serialization, and score lookup. `texts`, `labels`, and score lists must stay
aligned with their corresponding candidate IDs.

There is intentionally no legacy score-file reader. Existing experimental
score directories are disposable and should be removed before producing
canonical scores.

### 1a. Produce scalar supervision (before dataset formatting)

`scripts/score_custom_dataset.py` writes one JSONL file per scoring method
under `data/custom_datasets/<dataset>/scores/`. It first samples original IDs,
then scores every candidate belonging to each selected original. Originals are
self-scored by default.

The scoring methods use two deliberate input protocols:

| Method | Input used for scoring | Stored direction |
| --- | --- | --- |
| Token-normalized perplexity | Candidate only | Negative mean token NLL; higher is better |
| BERTScore F1 | Original as reference, candidate as prediction | Raw F1; higher is better |
| BLEURT | Original as reference, candidate as prediction | Raw score; higher is better |
| MetricX-24 QE | Original source plus candidate | Negated QE error; higher is better |
| GPTScore fluency | Original in the prompt plus candidate tokens | Negative candidate-token NLL; higher is better |

The source-aware MetricX and GPTScore methods are supervision teachers for
custom datasets. Their `score_pairs(...)` entry points are separate from the
candidate-only `score_texts(...)` paths used by direct benchmark evaluation.
Neither source-aware method uses a reference. For self-scored originals,
source and candidate are the same text, and the computed teacher score is
stored rather than replaced with a perfect score.

Successful score rows use `base_text_id`, `perturbation_source`,
`candidate_id`, `source_layer`, `target_layer`, `score_name`, and
`score_value`. Both perturbation sources share one score file per method.
Failures are written to `<score_name>.errors.jsonl`; the accompanying metadata
file records the model, input mode, score transformation, sampling settings,
and library versions. Every stored score is higher-is-better. Perplexity is
stored as `-log(perplexity)`; the other methods use the transformations shown
above.

When only some scores are available, pass `--score-names` to
`scripts/create_fe_training_dataset.py`. Splitting then considers only
original IDs with at least one requested score, so regression examples remain
distributed across train/dev/test.

### 2. Train an FE model

`scripts/train_fe_model.py` is the only canonical training entry point. It
loads a source-safe formatted `DatasetDict` and flattens it into the selected
training schema before constructing the trainer.

The backbone registry resolves pooling from the model name. Use `--pooling`
only for an intentional override or ablation; new checkpoints store the
resolved value and use it during reload.

`--training-method` selects the row schema and loss only:

- `pairwise`: one chosen/rejected candidate pair per row, with logistic or
  hinge ranking loss;
- `regression`: one candidate and one numeric target per row, with a selected
  regression loss.

Both paths use the same candidate-only scalar model: a pretrained Hugging Face
encoder, backbone-specific pooling, and one `Linear(hidden_size, 1)` head.
Training uses the standard Hugging Face `Trainer`; there are no
objective-specific trainer subclasses. New checkpoints use
`save_pretrained`/`from_pretrained` serialization and include
`fe_model_state.pt` and `fe_model_config.json`, recording the resolved pooling,
objective, loss, target transformation, architecture, and tokenizer metadata.
Legacy `fe_head.pt` checkpoints remain loadable for evaluation; their archived
MLP head is not used for new training.

`scripts/run_hpo.py` repeatedly invokes this trainer using a user-supplied JSON
trial file. `scripts/inspired_calibration_ft.py` performs optional CoheSentia
calibration on an existing FE checkpoint.

### 3. Evaluate

`python -m clumsification_code.evals.run_benchmark --scorer ...` is the shared entry point. It builds one adapter with a common `score_texts(...)` interface:

- `evals/inference/fe.py` for local FE checkpoints;
- `evals/inference/gptscore.py` for GPTScore;
- `evals/inference/metricx.py` and `evals/metricx24/` for MetricX;
- `evals/geval/` for G-Eval.
- `evals/inference/vllm_scorer.py` for generic vLLM-backed judges.

`evals/benchmark_runner.py` runs the datasets loaded by `benchmark_data.py`; `multilingual_benchmarks.py` provides normalized BASSE and Norwegian human-evaluation records; `metrics.py` calculates correlations/preferences; `result_writer.py` records results. `evaluate_model_on_tdt_regens.py` is the larger specialized workflow for regenerated TDT/UD texts.

For flat FE training-set diagnostics, use
`python -m scripts.analyze_fe_predictions ...`. It writes per-row predictions
and can generate length diagnostics without affecting training or checkpoint
selection.

Direct benchmark evaluation is candidate-only unless a benchmark adapter
explicitly defines another protocol. The source-aware `score_pairs(...)`
methods described above are used for custom-dataset supervision and are not
called by the direct benchmark runner.

### vLLM-backed judges

The vLLM scorer separates the model, evaluation protocol, and rubric:

```text
--scorer vllm
  --vllm-model-name-or-path <model>
  --vllm-protocol <file under data/prompts/evaluation/protocols/>
  --vllm-rubric <file under data/prompts/evaluation/rubrics/>
```

The current examples combine Qwen3-32B or M-Prometheus with either the
Prometheus absolute-assessment protocol and MENLO fluency rubric, or the
adapted G-Eval JSON protocol and corrected G-Eval fluency rubric. Protocol
metadata selects the output parser, while rubric files provide the evaluation
criteria. Tensor parallelism is controlled with `--vllm-tensor-parallel-size`.
Thinking is disabled by default; `--vllm-enable-thinking` can be used for
models or experiments that require it. Keep `--vllm-max-model-len` large
enough for the combined prompt and output, and use `--vllm-max-tokens` to
control the generation budget.

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
| New training objective | `fe/losses.py` or `fe/regression_data.py` | `FEModel.forward`, metrics, CLI args |
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
