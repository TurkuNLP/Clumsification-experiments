# Clumsification experiments

This repository contains the central methodology for generating controlled
fluency perturbations, constructing evaluator-training datasets, training text
quality evaluators, and evaluating them.

The perturbation workflow supports:

- zero-shot and sampled-operation LLM perturbations;
- UniEval-style and multilingual traditional perturbations;
- generation from an original or any canonical perturbation layer;
- method- and run-separated outputs with exact candidate ancestry;
- scalar supervision attached to exact candidate identities;
- leakage-safe Hugging Face datasets with configurable mixtures and pairs.

## Canonical commands

Generate one perturbation layer:

```bash
python scripts/generate_perturbations.py \
  --dataset <dataset> --source-layer 0 \
  --method llm_sampled --run-id sampled-v1
```

Build a Hugging Face dataset:

```bash
python scripts/build_hf_dataset.py \
  --datasets <dataset> --output-name <name> \
  --include-methods llm_sampled trad_multi \
  --include-layers 1 2
```

Run generation and HF construction from one configuration:

```bash
python scripts/prepare_dataset.py run-all \
  --config configs/workflow.example.json
```

Score selected candidates for regression supervision:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1
```

Generated datasets, results, tests, notebooks, figures, local archives, and
cluster batch jobs are intentionally not repository sources.

See [the perturbation workflow guide](docs/PERTURBATION_CONFIGS.md) for exact
schemas and examples, and [the architecture overview](ARCHITECTURE.md) for the
end-to-end data flow.
