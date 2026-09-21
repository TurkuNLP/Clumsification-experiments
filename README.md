# Clumsification experiments

This repository contains the central methodology for generating controlled
fluency perturbations, constructing evaluator-training datasets, training text
quality evaluators, and evaluating them.

The perturbation workflow supports:

- single-edit and sampled-edit LLM perturbations;
- canonical multilingual traditional perturbations;
- generation from an original or any canonical perturbation layer;
- method- and run-separated outputs with exact candidate ancestry;
- automatic scalar supervision attached to exact candidate identities,
  including G-Eval and Themis/MENLO fluency judgments;
- leakage-safe Hugging Face datasets with configurable mixtures and pairs.

The fixed English source corpus, `nemotron-cc-high-propella-custom-eng`, has
84,554 custom-vLLM-filtered documents. Its shared source-level split manifest
is created only after the four layer-1 perturbation workflows finish, with
74,554 train, 5,000 development, and 5,000 test sources by default.

## Canonical commands

Create the frozen LLM assignment manifest, then generate all four independent
layer-1 workflows. Pass the assignment file to both LLM commands.

```bash
python scripts/plan_llm_assignments.py --dataset <dataset> --seed 42

python scripts/generate_perturbations.py \
  --dataset <dataset> --source-layer 0 \
  --method llm_sampled --run-id <run-id> \
  --model-path Qwen/Qwen3.5-27B \
  --assignment-file data/custom_datasets/<dataset>/perturbation_assignments.jsonl
```

This is a required workflow boundary: do not build an HF dataset after only a
subset of the four workflows (for example, after `trad_single` alone). After
all four layer-1 runs complete successfully, generate and review the shared
split manifest:

```bash
python scripts/assign_workflow_splits.py \
  --dataset <dataset> \
  --llm-single-run-id <run-id> \
  --llm-sampled-run-id <run-id> \
  --trad-single-run-id <run-id> \
  --trad-sampled-run-id <run-id>
```

`perturbation_assignments.jsonl` and `split_assignments.jsonl` have different
roles. The former freezes only LLM edit requests; the latter assigns every
source to `train`, `dev`, or `test` and is required by the HF builder.

Score candidates whenever the desired supervision is available, then build a
Hugging Face dataset. For example, this creates original--`trad_single` pairs
with both BERTScore F1 and BLEURT arrays:

```bash
python scripts/build_hf_dataset.py \
  --datasets <dataset> --output-name <name> \
  --include-methods trad_single --include-runs <trad-single-run-id> \
  --include-layers 1 --pair-policy original_only \
  --score-names bertscore_f1 bleurt \
  --score-run-ids <bertscore-run-id> <bleurt-run-id>
```

LLM generation derives context and output limits automatically from source
length. Each context bucket is committed in batches of at most 512 items by
default (`--batch-size` changes this). Re-submitting the same command resumes
unattempted batches; `--retry-failed` attempts only recorded failures.

Score selected candidates for regression supervision:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1
```

Two candidate-only LLM-judge supervision conditions are available alongside
the metric-based scorers:

```bash
# G-Eval with the pinned GPT-5.4-mini judge, using OpenAI Batch API
python scripts/score_custom_dataset.py \
  --dataset-name fe-dataset-final \
  --scoring-type geval_gpt54mini_fluency \
  --scoring-run-id geval-gpt54mini-trad-sampled-v1 \
  --methods trad_sampled \
  --perturbation-run-ids trad-sampled-balanced-v2 \
  --target-layers 1 \
  --exclude-originals \
  --geval-batch-size 10000 \
  --geval-batch-action prepare

# Themis with the MENLO fluency rubric (requires vLLM/GPU)
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type menlo_themis_fluency \
  --scoring-run-id menlo-themis-v1 \
  --themis-tensor-parallel-size 1
```

The G-Eval command prepares five 10,000-request JSONL files with full candidate
texts. Review the files and repeat the command with `--geval-batch-action submit`
to upload and submit them; the runner uses `OpenAI_lib.get_client_local()` to
read the local OpenAI credential. Repeat with
`--geval-batch-action collect` after the jobs finish. Submission and collection
resume from saved Batch IDs. The final scores, errors, and metadata use the
canonical `scores/` format. Batch queue limits may require submitting the
remaining files later. If one 10,000-request file exceeds your account's queue
limit, prepare smaller files with `--geval-batch-size` and `--overwrite` before
uploading any file; use a new scoring run ID if an upload already happened.
The Themis method writes canonical scores directly.

For a completed scoring run with errors, repeat the original selection and run
ID with `--retry-failed`. This selects only candidates in that run's error file
and keeps existing successful scores. The Batch scorer creates a new request
set for those failures; repeat the submit and collect actions for it. Other
scorers retry unresolved candidates for up to 100 additional rounds within
the same job; `--retry-failed-max-retries` changes that limit. An interrupted
run resumes its saved progress without this flag.

### Evaluate a formatted dataset split

The same scorer can evaluate a saved formatted Hugging Face dataset directly.
The `test` split is selected by default; use `--formatted-dataset-split` to
evaluate another split. Regression datasets require the score field used to
construct them:

```bash
python -m clumsification_code.evals.run_benchmark \
  --evaluation-role formatted-dataset \
  --formatted-dataset-path data/hf_datasets/<name> \
  --training-method regression \
  --score-name <score-name> \
  --scorer fe \
  --model-name <model-name> \
  --model-dir <model-dir>
```

Pairwise and binary formatted datasets are also supported with
`--training-method pairwise` or `--training-method binary`.

### Evaluate the English benchmark suite

Use the shared benchmark runner for the audited English suite. G-Eval uses the
OpenAI Batch API by default. Preparation is local-only: it materializes the
deduplicated requests and a resumable manifest without making paid API calls.

```bash
python -m clumsification_code.evals.run_benchmark \
  --scorer geval \
  --model-name gpt54mini-geval \
  --geval-model gpt-5.4-mini-2026-03-17 \
  --geval-processing batch \
  --geval-batch-run-id gpt54mini-geval-english-v1 \
  --geval-batch-action prepare \
  --geval-batch-size 5000 \
  --max-output-tokens 64 \
  --skip-multilingual
```

Inspect `data/evals/geval_batches/gpt54mini-geval-english-v1/`, then repeat the
same command with `--geval-batch-action submit`. Submission saves every upload
and Batch ID before continuing. Batch submission and collection obtain their
client from `OpenAI_lib.get_client_local()`, matching the other project Batch
workflow; no API-key argument or environment variable is needed. If the
account's active Batch queue fills, wait
for submitted chunks to finish and repeat `submit`; already submitted chunks
are not duplicated. Run the same command with `--geval-batch-action collect`
until all chunks finish. Collection downloads and preserves the raw output
files, parses scores, computes the normal benchmark metrics, and writes the
final result to `data/evals/final/gpt54mini-geval.jsonl` exactly once.

If collection reports failed or missing requests, use
`--geval-batch-action retry`. Only unresolved requests are submitted again;
successful responses remain in the run directory and are reused. Keep the
model, prompt, output-token limit, suite flags, and run ID identical at every
stage. Story Cloze is not part of the final runner; leave `--skip-preferences`
off to retain the JFLEG and English MultiBLiMP evaluations. The legacy direct
request implementation remains available with `--geval-processing direct`.

Themis uses the existing vLLM benchmark path with the Themis-native protocol
and MENLO rubric:

```bash
python -m clumsification_code.evals.run_benchmark \
  --scorer vllm \
  --model-name themis-menlo \
  --vllm-model-name-or-path PKU-ONELab/Themis \
  --vllm-protocol themis_direct_assessment.json \
  --vllm-rubric menlo_fluency.json \
  --vllm-tensor-parallel-size 1 \
  --skip-multilingual
```

The final benchmark command writes results to `data/evals/final/`. Use
`--max-records-per-dimension` with `--skip-preferences` for a small pilot;
the record limit applies to scalar dimensions, while `--skip-preferences`
avoids running all JFLEG and MultiBLiMP pairs during that pilot.

For a full LUMI-G node, vLLM can run independent scoring replicas. The total
device count is `--vllm-data-parallel-size` multiplied by
`--vllm-tensor-parallel-size`; each replica receives its own share of candidates,
and `--batch-size` applies within each replica. For example, four replicas with
two devices each:

```bash
sbatch --gpus-per-node=8 --cpus-per-task=32 --time=03:00:00 \
  updated_sbatch_jobs/evaluate.sh vllm Qwen3.5-9B-geval \
  --vllm-model-name-or-path Qwen/Qwen3.5-9B \
  --vllm-protocol geval_json.json \
  --vllm-rubric geval_no_reference.json \
  --vllm-data-parallel-size 4 --vllm-tensor-parallel-size 2 \
  --vllm-max-model-len 32768 --batch-size 64
```

If the model fits on one device, `--vllm-data-parallel-size 8` with
`--vllm-tensor-parallel-size 1` provides eight replicas. The default DP size is
one, preserving previous single-engine behavior. Validate memory and throughput
on LUMI before selecting a production layout.

### Evaluate checkpoints on the external development panel

Checkpoint and hyperparameter decisions use a separate human-labeled panel:
ELLIPSE train, JFLEG validation, and CoheSentia train. The command below never
scores the final English suite and writes under `data/evals/external_dev/`:

```bash
python -m clumsification_code.evals.run_benchmark \
  --evaluation-role external-dev \
  --scorer fe \
  --model-name <unique-checkpoint-and-seed-name> \
  --model-dir <checkpoint-directory> \
  --batch-size 32 \
  --max-length 32768
```

Add `--include-dev-story-cloze-diagnostic` only for the secondary narrative
diagnostic; it has zero checkpoint-selection weight. After all checkpoints
have been evaluated, rank them with equal dataset weight:

```bash
python scripts/rank_external_dev_checkpoints.py \
  data/evals/external_dev/*.jsonl \
  --output data/evals/external_dev/checkpoint_ranking.csv
```

On the cluster, evaluate every `checkpoint-*` directory plus `final/` with a
four-GPU worker queue:

```bash
sbatch updated_sbatch_jobs/evaluate_all_fe_checkpoints.sh \
  <training-output-directory> \
  <MODEL_LANGUAGE_TRAINING_DATASET run name> \
  dev
```

Each GPU takes one FE checkpoint and then the next until the folder is
exhausted. Each log is written directly into the training output directory.
Submitting the same command again skips completed checkpoints and retries the
rest. Use `full` instead of `dev` for the final suite. The existing
`evaluate.sh` remains the single-model launcher for FE and baseline scorers.

For Hugging Face PPL, allocate multiple GPUs and use independent model replicas
to score text shards in parallel. `--batch-size` is per replica. For example:

```bash
sbatch --gpus-per-node=8 --cpus-per-task=32 --mem=400G --time=12:00:00 \
  updated_sbatch_jobs/evaluate.sh ppl Qwen3.5-9B-Base-ppl \
  --hf-model-name-or-path Qwen/Qwen3.5-9B-Base \
  --ppl-data-parallel-size 8 --batch-size 1 --max-length 512
```

The PPL scorer also accepts Qwen3.5's multimodal model class for text-only
likelihood evaluation. Leave `--ppl-data-parallel-size` at 1 for one GPU.

The job requests four LUMI GPU devices by default. For a full eight-device
LUMI-G node, override the embedded allocation at submission time:

```bash
sbatch --gpus-per-node=8 --cpus-per-task=32 \
  updated_sbatch_jobs/evaluate_all_fe_checkpoints.sh \
  <training-output-directory> \
  <MODEL_LANGUAGE_TRAINING_DATASET run name> \
  dev
```

The audited split identities, revisions, checksums, and selection rule are
frozen in `configs/english_external_dev.json`. Final evaluation writes under
`data/evals/final/` and should be run only after the chosen configuration is
frozen.

Generated datasets, results, tests, notebooks, figures, local archives, and
cluster batch jobs are intentionally not repository sources.

See [the perturbation workflow guide](docs/PERTURBATION_CONFIGS.md) for exact
schemas and examples, and [the architecture overview](ARCHITECTURE.md) for the
end-to-end data flow. The [English evaluation-suite documentation](docs/ENGLISH_EVAL_SUITE.md)
records every included label dimension, its fluency-category mapping, annotation
criteria, agreement evidence, and corpus profile.
