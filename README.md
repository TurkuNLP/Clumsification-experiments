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
# G-Eval with the pinned GPT-5.4-mini judge
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type geval_gpt54mini_fluency \
  --scoring-run-id geval-gpt54mini-v1 \
  --geval-cache-path data/evals/<dataset>_geval_cache.json

# Themis with the MENLO fluency rubric (requires vLLM/GPU)
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type menlo_themis_fluency \
  --scoring-run-id menlo-themis-v1 \
  --themis-tensor-parallel-size 1
```

Both methods score candidates only and write canonical score, error, and
metadata files under the dataset's `scores/` directory. The G-Eval cache keeps
raw API responses; set `OPENAI_API_KEY` before running it.

### Evaluate the English benchmark suite

Use the shared benchmark runner for direct evaluation of the audited English
suite. G-Eval uses the existing JSON protocol and can be run with GPT-5.4-mini:

```bash
python -m clumsification_code.evals.run_benchmark \
  --scorer geval \
  --model-name gpt54mini-geval \
  --geval-model gpt-5.4-mini-2026-03-17 \
  --geval-task fluency \
  --geval-aspect fluency \
  --skip-multilingual
```

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
`--max-records-per-dimension` for a pilot and omit `--skip-preferences` if the
JFLEG, MultiBLiMP, and Story Cloze diagnostics are desired.

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
