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
is created only after the four layer-1 perturbation workflows finish.

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

After all four runs complete, generate and review the shared split manifest:

```bash
python scripts/assign_workflow_splits.py \
  --dataset <dataset> \
  --llm-single-run-id <run-id> \
  --llm-sampled-run-id <run-id> \
  --trad-single-run-id <run-id> \
  --trad-sampled-run-id <run-id>
```

Then build a Hugging Face dataset if needed:

```bash
python scripts/build_hf_dataset.py \
  --datasets <dataset> --output-name <name> \
  --include-methods llm_sampled trad_sampled \
  --include-layers 1 2
```

LLM generation derives context and output limits automatically from source
length.

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

The benchmark command writes results to `data/evals/`. Use
`--max-records-per-dimension` for a pilot and omit `--skip-preferences` if the
JFLEG, MultiBLiMP, and Story Cloze diagnostics are desired.

Generated datasets, results, tests, notebooks, figures, local archives, and
cluster batch jobs are intentionally not repository sources.

See [the perturbation workflow guide](docs/PERTURBATION_CONFIGS.md) for exact
schemas and examples, and [the architecture overview](ARCHITECTURE.md) for the
end-to-end data flow. The [English evaluation-suite documentation](docs/ENGLISH_EVAL_SUITE.md)
records every included label dimension, its fluency-category mapping, annotation
criteria, agreement evidence, and corpus profile.
