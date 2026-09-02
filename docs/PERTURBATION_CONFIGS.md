# Perturbation and dataset workflow

This document describes the canonical local workflow. It does not depend on
cluster batch jobs. Paths are relative to the repository root.

## Custom-dataset layout

Each custom dataset starts with `original.jsonl`:

```text
data/custom_datasets/<dataset>/
  original.jsonl
  perturbations/
    perturbation_manifest.json
    <method>/<run_id>/<target_layer>.jsonl
  scores/
    <scoring_method>/<scoring_run_id>.jsonl
    <scoring_method>/<scoring_run_id>.errors.jsonl
    <scoring_method>/<scoring_run_id>.metadata.json
```

An original row requires `custom_id` and `text`. String and integer source IDs
are accepted and normalized to strings. The source ID identifies a document;
`candidate_id` identifies one exact original or perturbation candidate.

The manifest is the authoritative layer index. Directory scanning is not used
to infer layers. Every candidate records its method, run, source and target
layers, and exact `parent_candidate_id`. A new layer may therefore start from
an original or any existing layer, including one produced by another method.

## Generate one layer

Use `scripts/generate_perturbations.py` for a single generation:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_sampled \
  --run-id sampled-medium-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B \
  --target-dimensions Clarity Naturalness \
  --severity medium \
  --n-edits 3
```

To perturb an existing layer, identify its method and run:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 1 \
  --source-method llm_sampled \
  --source-run-id sampled-medium-v1 \
  --method trad_multi \
  --run-id trad-after-llm-v1 \
  --target-layer 2 \
  --language en \
  --n-edits 3
```

`target_layer` defaults to `source_layer + 1`. A perturbed source requires
both `source_method` and `source_run_id`. Existing outputs are protected; use
`--overwrite` only when replacement is intentional. A reusable method config
can be passed with `--method-config`; explicit CLI values take precedence.

### Available methods

| Method | Meaning |
| --- | --- |
| `llm_zero_shot` | Fixed-prompt LLM perturbation retained for the ablation. |
| `llm_sampled` | Samples edit types, fluency dimensions, and severity for the LLM prompt. |
| `unieval` | UniEval-style insertion, deletion, and shuffle noise. |
| `trad_single` | One applicable traditional perturbation. |
| `trad_multi` | Multiple applicable traditional perturbations. |
| `unieval_trad` | UniEval noise followed by traditional edits. |

The traditional implementation is multilingual. English inflection uses
Lemminflect; other supported languages use UniMorph. Language-specific
operations are excluded where they are not applicable. The former
`unieval_summinflect` variant is represented by `unieval_trad`.

### Sampled LLM options

```json
{
  "model": "Qwen/Qwen3.5-27B",
  "language": "english",
  "edit_catalog": "data/perturbation_prompts/english/edit_types.jsonl",
  "target_dimensions": ["Clarity", "Naturalness"],
  "n_edits": 3,
  "severity": "medium",
  "require_dimension_coverage": true,
  "weights": {
    "unnecessary_circumlocution": 1.0,
    "odd_collocation": 1.0
  },
  "seed": 42
}
```

`severity` accepts `weak`, `medium`, `strong`, or an array from which a value
is sampled deterministically. vLLM and model dependencies load only when an
LLM method is executed.

### Traditional options

```json
{
  "language": "en",
  "n_noise": 1,
  "n_edits": 3,
  "operations": ["jumble", "subject_verb_dis", "typos"],
  "seed": 42
}
```

`operation` selects the operation for `trad_single`. `operations` restricts
the pool for `trad_multi` and `unieval_trad`.

## Score canonical candidates

Scores attach to exact candidates and are separated by method and score run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1 \
  --methods llm_sampled trad_multi \
  --perturbation-run-ids sampled-medium-v1 trad-after-llm-v1 \
  --target-layers 1 2 \
  --reference-policy parent
```

`reference-policy original` uses the source original; `parent` uses the exact
parent candidate. Score files retain both candidate and reference identities.
All stored scores are higher-is-better.

The custom-dataset scorer also supports `geval_gpt54mini_fluency`, which uses
the existing G-Eval scorer with the pinned GPT-5.4-mini snapshot, and
`menlo_themis_fluency`, which uses the Themis vLLM scorer with the MENLO
five-point fluency rubric. Both score candidates only; their prompt, rubric,
model, parser, and decoding settings are retained in score-run metadata.

For example, G-Eval scoring uses the pinned GPT-5.4-mini judge and an optional
response cache:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type geval_gpt54mini_fluency \
  --scoring-run-id geval-gpt54mini-v1 \
  --geval-cache-path data/evals/my_dataset_geval_cache.json
```

The Themis/MENLO scorer runs locally through vLLM:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type menlo_themis_fluency \
  --scoring-run-id menlo-themis-v1 \
  --themis-model-name PKU-ONELab/Themis \
  --themis-tensor-parallel-size 1
```

These custom-dataset commands score candidate text only. They do not pass the
original or parent text to either judge, even when `--reference-policy` is
used for score provenance. The reference policy controls stored candidate
identity only.

## Build a Hugging Face dataset

The standalone builder accepts canonical CLI fields or an `HFBuildSpec` JSON
object such as `configs/hf_build.example.json`.

```bash
python scripts/build_hf_dataset.py \
  --datasets my_dataset \
  --output-name my_dataset_hf \
  --include-methods llm_sampled trad_multi \
  --include-runs sampled-medium-v1 trad-after-llm-v1 \
  --include-layers 1 2 \
  --composition balanced \
  --pair-policy parent_child \
  --score-names bertscore_f1 \
  --score-run-ids bertscore-v1
```

Equivalent config-based use:

```bash
python scripts/build_hf_dataset.py --config configs/hf_build.example.json
```

The original is included automatically. Method, run, and layer filters are
independent; omitting one includes all values for that field.

Composition is performed separately for each source document:

| Policy | Selection |
| --- | --- |
| `all` | Every selected candidate. |
| `balanced` | The same candidate count from each available method. |
| `weighted` | Sampling without replacement using `method_weights`. |
| `source_exclusive` | Assign each source document to one method. |
| `fixed_per_source` | Keep up to `samples_per_source` candidates per method. |

Pair policies are applied only after source-safe splitting:

| Policy | Result |
| --- | --- |
| `none` | One aligned candidate group per source document. |
| `parent_child` | One pair for each selected exact graph edge. |
| `original_only` | Original versus each selected perturbation. |
| `all_unequal_layers` | Every same-source pair with different target layers. |
| `cross_source_unmatched` | Different-source, unequal-layer pairs within one dataset, with reuse bounded by `reuse_limit`. |

`score_names` selects scoring methods. If multiple score runs exist for a
selected candidate and method, specify `score_run_ids`; ambiguity is rejected.

The output is a `DatasetDict` with `train`, `dev`, and `test`. Sources are
split before composition or pairing. Rows preserve aligned text, target layer,
candidate ID, method, run, parent, source-layer, and score arrays.

## Run a complete configured workflow

`scripts/prepare_dataset.py` uses the same generation and HF contracts. Copy
`configs/workflow.example.json`, adjust it, then run:

```bash
python scripts/prepare_dataset.py generate --config configs/workflow.example.json
python scripts/prepare_dataset.py build-hf --config configs/workflow.example.json
python scripts/prepare_dataset.py run-all --config configs/workflow.example.json
```

Generation options belong inside each entry's `config` object:

```json
{
  "schema_version": 1,
  "dataset": "my_dataset",
  "dataset_root": "data/custom_datasets",
  "seed": 42,
  "generations": [
    {
      "method": "llm_sampled",
      "run_id": "sampled-medium-v1",
      "source_layer": 0,
      "target_layer": 1,
      "config": {
        "model": "Qwen/Qwen3.5-27B",
        "target_dimensions": ["Clarity", "Naturalness"],
        "severity": "medium",
        "n_edits": 3
      }
    }
  ],
  "hf": {
    "output_name": "my_dataset_sampled",
    "include_methods": ["llm_sampled"],
    "include_runs": ["sampled-medium-v1"],
    "include_layers": [1],
    "composition": "all",
    "pair_policy": "none"
  }
}
```

Within a workflow, `hf.datasets` defaults to the top-level dataset and
`hf.seed` defaults to the workflow seed. Presets are available for
`zero_shot_ablation`, `sampled_llm_ablation`, and `traditional_comparison`.

## Legacy import

Historical folders are accepted only through explicit migration:

```bash
python scripts/import_legacy_dataset.py \
  --dataset my_dataset \
  --source-directory perturbed_layers \
  --method llm_zero_shot \
  --run-id legacy-import
```

New generation, scoring, and HF construction use only canonical repositories
and manifests. Deprecated scripts are not extension points.
