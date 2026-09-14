# Perturbation and dataset workflow

This document describes the canonical local workflow. It does not depend on
cluster batch jobs. Paths are relative to the repository root.

## Custom-dataset layout

Each custom dataset starts with `original.jsonl`:

```text
data/custom_datasets/<dataset>/
  original.jsonl
  perturbation_assignments.jsonl
  split_assignments.jsonl
  perturbations/
    perturbation_manifest.json
    <method>/<run_id>/<target_layer>.jsonl
  scores/
    <scoring_method>/<scoring_run_id>.jsonl
    <scoring_method>/<scoring_run_id>.errors.jsonl
    <scoring_method>/<scoring_run_id>.metadata.json
```

### Import the English corpus after vLLM filtering

The fixed English source corpus is
`nemotron-cc-high-propella-custom-eng`. Its provenance is:

1. `nemotron-cc-high-actual` was filtered across 21 source crawls to roughly
   5.5 million documents. A document was retained only when it had 200--20,000
   characters and passed the Propella conditions `content_ratio=complete_content`,
   `content_integrity=complete`, and `content_quality` equal to `excellent` or
   `high`.
2. Every tenth retained document was sampled, yielding exactly **557,017**
   source rows in `nemotron-cc-high-propella-eng`.
3. The custom vLLM filter was run over all 557,017 rows. It evaluates each
   document relative to its genre and rejects incoherent, boilerplate-heavy,
   spammy, templated, code-like, list-dominated, metadata-dominated, or
   otherwise low-quality text. It permits minor defects and short-form genres
   only when there is a substantial excellent section. The raw assessment is
   stored in `passes_filters`.
4. Only a valid assessment with `decision="PASS"` and
   `contains_substantial_high_quality_section=true` was retained. This yielded
   exactly **84,554** documents: 451,780 explicit FAIL rows, 20,679 malformed
   assessments, and 4 internally inconsistent assessments were excluded.
5. The four perturbation workflows are generated from the same unsplit source
   corpus. A shared source-level split manifest is created afterward.

The mass filter writes one output row for every input row and stores its JSON
assessment in `passes_filters`. The importer reconstructs the pass-only
canonical corpus from that output:

```bash
# Validate the full input and preview the accepted/rejected counts.
python scripts/import_filtered_custom_dataset.py \
  --input /path/to/completed-filter-output.jsonl \
  --dataset nemotron-cc-high-propella-custom-eng \
  --dry-run

# Write original.jsonl and filter_import_manifest.json atomically.
python scripts/import_filtered_custom_dataset.py \
  --input /path/to/completed-filter-output.jsonl \
  --dataset nemotron-cc-high-propella-custom-eng \
  --overwrite
```

The importer fails closed: null, malformed, FAIL, or internally inconsistent
assessments are excluded. Passing rows retain their original metadata plus the
parsed filter assessment and input line number under `filter_provenance`. The
manifest records the input/output paths, the exact acceptance rule, and counts
for every outcome. Duplicate passing source IDs, invalid texts, and existing
derived perturbation/score/split artifacts are hard errors.

An original row requires `custom_id` and `text`. String and integer source IDs
are accepted and normalized to strings. The source ID identifies a document;
`candidate_id` identifies one exact original or perturbation candidate.

The manifest is the authoritative layer index. Directory scanning is not used
to infer layers. Every candidate records its method, run, source and target
layers, and exact `parent_candidate_id`. A new layer may therefore start from
an original or any existing layer, including one produced by another method.

### LLM assignments and workflow splits

Plan the two LLM workflows before generation. The assignment file freezes
their edit types, edit counts, severities, and derived dimensions; retries reuse
that assignment and change only the model-generation seed.

```bash
python scripts/plan_llm_assignments.py \
  --dataset nemotron-cc-high-propella-custom-eng --seed 42
```

Pass the resulting file when generating either LLM layer:

```bash
python scripts/generate_perturbations.py \
  --dataset nemotron-cc-high-propella-custom-eng \
  --source-layer 0 --method llm_sampled --run-id sampled-balanced-v2 \
  --model-path Qwen/Qwen3.5-27B \
  --assignment-file data/custom_datasets/nemotron-cc-high-propella-custom-eng/perturbation_assignments.jsonl
```

Create one source-level split file after the four independent workflows have
written their outputs. It keeps only sources present in every workflow and
defaults to 50,000 training, 5,000 development, and 5,000 test sources. It
balances source length and realized LLM characteristics while treating
traditional edit types as a lighter signal.

```bash
python scripts/assign_workflow_splits.py \
  --dataset nemotron-cc-high-propella-custom-eng \
  --llm-single-run-id single-balanced-v2 \
  --llm-sampled-run-id sampled-balanced-v2 \
  --trad-single-run-id trad-single-v1 \
  --trad-sampled-run-id trad-sampled-v1
```

The result is `split_assignments.jsonl`, with one `base_text_id` and one of
`train`, `dev`, or `test` per row. It is the only supported source of split
membership. It applies to the original and to all workflow outputs for that
source. It is also a hard prerequisite for `scripts/build_hf_dataset.py`.

`perturbation_assignments.jsonl` is not a split file: it records only the
preplanned LLM edit requests. In particular, having completed and scored
`trad_single` (or either traditional workflow) does not yet permit an HF build.
The current split planner requires the completed `llm_single`, `llm_sampled`,
`trad_single`, and `trad_sampled` layer-1 outputs named in the command above.

## Generate one layer

Run one method per layer with `scripts/generate_perturbations.py`. The command
records the effective configuration, source selection, candidate ancestry, and
method-specific edit evidence in the perturbation manifest. LLM assignment
choices are recorded separately in `perturbation_assignments.jsonl`.

### Canonical methods

The only generative methods are:

| Method | Number of edits | Generation procedure |
| --- | --- | --- |
| `llm_single` | Exactly 1 | Use the preplanned catalog operation, target dimension, and severity, then ask the LLM to apply it. |
| `llm_sampled` | 1--5 | Use the preplanned length-conditioned operations, derived dimensions, and severity, then ask the LLM to apply them. |
| `trad_single` | Exactly 1 | Sample one applicable operation from the five-operation traditional mix. |
| `trad_sampled` | 1--5 | Sample a length-conditioned number of traditional edits. |

For `llm_sampled`, the planner assigns the requested edit count uniformly from
`1..min(5, floor(character_length / 500))`. A text shorter than 500 characters
therefore receives one edit. `trad_sampled` applies the same rule during its
CPU generation. Both are deterministic for a source candidate and seed.

Run the two LLM workflows from originals as follows. `--model-path` identifies
the local or Hugging Face model served by the LLM runner.

```bash
# One LLM edit per original source.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_single \
  --run-id llm-single-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B \
  --assignment-file data/custom_datasets/my_dataset/perturbation_assignments.jsonl

# One to five LLM edits per original source, conditional on text length.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_sampled \
  --run-id llm-sampled-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B \
  --assignment-file data/custom_datasets/my_dataset/perturbation_assignments.jsonl
```

Run the two traditional workflows from originals as follows. `--language en`
uses Lemminflect; other supported languages use UniMorph.

```bash
# Exactly one traditional edit per original source.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method trad_single \
  --run-id trad-single-v1 \
  --target-layer 1 \
  --language en

# One to five traditional edits per original source, conditional on text length.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method trad_sampled \
  --run-id trad-sampled-v1 \
  --target-layer 1 \
  --language en
```

To generate from an existing perturbation layer, provide the exact source
method and run ID. For example:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 1 \
  --source-method llm_sampled \
  --source-run-id llm-sampled-v1 \
  --method trad_sampled \
  --run-id trad-sampled-after-llm-v1 \
  --target-layer 2 \
  --language en
```

`target_layer` defaults to `source_layer + 1`. A perturbed source requires
both `source_method` and `source_run_id`. LLM outputs are checkpointed after
each batch, and the same command resumes unattempted inputs automatically. Use
`--overwrite` only when replacement from the beginning is intentional. A
reusable method config can be passed with `--method-config`; explicit CLI
values take precedence.

### Recover failed inputs without regenerating successful ones

If a partial or completed layer contains failed inputs, rerun the original command with
the same dataset, source, method, run ID, frozen assignment file, and
generation configuration, adding `--retry-failed`:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset --source-layer 0 --method llm_sampled \
  --run-id sampled-dynamic-v1 --model-path Qwen/Qwen3.5-27B \
  --assignment-file data/custom_datasets/my_dataset/perturbation_assignments.jsonl \
  --retry-failed
```

This retries only source candidates explicitly recorded as failed; unattempted
inputs are not selected in this mode. It preserves all successful candidate
rows and updates the same layer. It does not create another run. Each
retry records its effective seed, attempted and recovered counts, and any
remaining failures in that layer's manifest. For both LLM and traditional
methods, retry rounds use a distinct effective generation seed while retaining
the original `seed` as the immutable request seed. `--retry-failed` cannot be
used with `--overwrite`.

Without `--retry-failed`, re-submitting an interrupted LLM job continues only
inputs that have never been attempted. Within each context bucket, results are
committed in chunks of at most `--batch-size` items (default 512); a smaller
bucket is committed as one batch. Every selected input gets exactly one model
generation chance per submission. The adjacent `.progress.json` file records
attempted and failed parent identities, while the canonical layer and manifest
are updated at every batch boundary.

The derived output-length ceiling includes a 256-character tolerance by
default. This tolerance is included in the prompt and enforced by validation,
so small 100--200 character overruns are accepted consistently. Set
`--max-output-char-tolerance 0` for the former strict behavior or provide a
different non-negative value.
The tolerance may be increased when using `--retry-failed`, so failures from a
previous strict run can be recovered without changing its run ID.

`--n-jobs` controls only local parallelism and may be changed for a retry. All
source, method, run, seed, and content-generation settings must still match
the existing layer.

### Sampled LLM options

```json
{
  "model": "Qwen/Qwen3.5-27B",
  "language": "english",
  "edit_catalog": "data/perturbation_prompts/english/edit_types.jsonl",
  "assignment_file": "data/custom_datasets/my_dataset/perturbation_assignments.jsonl",
  "seed": 42
}
```

`assignment_file` is required for both LLM methods. It fixes the edit count,
edit types, severity, and dimensions for every source before any model call.
For `llm_sampled`, the assignment planner samples a length-conditioned count
from one through `min(5, floor(character_length / 500))`; operations and
severity are balanced over the full corpus, and dimensions are derived from
the assigned operations. The stable `seed` records the assignment identity;
retries change only the generation seed.

LLM outputs that are empty, unchanged, or longer than their requested
character limit are retried up to three times. If the final output is otherwise
valid but still over the length limit, it is retained and marked with
`length_limit_exceeded`, `output_chars`, `max_output_chars`, and
`retry_attempts` in its candidate metadata.

### Edit-count provenance

Every generated candidate row has an `edit_count` field. For both
`llm_sampled` and traditional methods, it is the number of recorded
`perturbation_edits` and must equal the length of that array. In particular,
it is not the number of target dimensions or a severity level. For
`llm_sampled`, it is the per-text planned number of required edit operations;
for traditional methods, it is the number of operations that actually made a
change.

### Traditional sampling and verification

`trad_single` always realizes one edit. `trad_sampled` uses the same
per-text length rule as `llm_sampled`: it samples uniformly from one through
`min(5, floor(character_length / 500))`. For every requested edit position,
the five operations begin equally likely. An operation that cannot make a
substantive change is removed from that position's pool and another remaining
operation is drawn uniformly. After a successful edit, the full five-operation
pool is restored, so later positions sample with replacement.

The only traditional configuration normally needed is `language` (plus the
global deterministic `seed`). Fixed edit counts and operation-restriction
flags are intentionally not exposed.

The five operations are equally likely at the start of every requested edit:

1. UniEval-style token-span repetition/insertion.
2. UniEval-style token-span deletion.
3. UniEval-style token-span shuffle.
4. Same-lemma finite-verb agreement corruption.
5. Same-lemma random morphology alteration, including a possible POS change.

Repetition and deletion always select at least one token; shuffle selects at
least two and must alter their order. Deletion that would empty a text is
inapplicable. Morphology edits require evidence that the replacement has the
same lemma but different recorded features. If an operation is inapplicable,
it is removed only from the current edit's sampling pool and another remaining
operation is sampled. A successful edit restores all five operations for the
next requested edit, so sampled edits are drawn with replacement.

## Score canonical candidates

Scores attach to exact candidates and are separated by method and score run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1 \
  --methods llm_sampled trad_sampled \
  --perturbation-run-ids sampled-dynamic-v1 trad-sampled-after-llm-v1 \
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

Scoring may occur before or after writing `split_assignments.jsonl` when all
candidates are being scored. If `--source-partitions` is used, write the split
manifest first. Regardless of scoring order, do not invoke the HF builder until
the split manifest exists and covers every source selected for the build.

## Build a Hugging Face dataset

The standalone builder accepts canonical CLI fields or an `HFBuildSpec` JSON
object such as `configs/hf_build.example.json`.

```bash
python scripts/build_hf_dataset.py \
  --datasets my_dataset \
  --output-name my_dataset_hf \
  --include-methods llm_sampled trad_sampled \
  --include-runs sampled-dynamic-v1 trad-sampled-after-llm-v1 \
  --include-layers 1 2 \
  --composition balanced \
  --pair-policy parent_child \
  --score-names bertscore_f1 \
  --score-run-ids bertscore-v1
```

The builder reads `split_assignments.jsonl` automatically and requires it to
cover every selected source. Therefore the full order is: plan LLM assignments;
generate all four independent layer-1 workflows; write the shared split
manifest; score the candidates needed for supervision (unless already scored);
then build the HF dataset. The builder does not read or derive splits from
`perturbation_assignments.jsonl`.

For a scored `trad_single` pair dataset, after the four-workflow split manifest
exists, use one score run ID for each requested scoring method:

```bash
python scripts/build_hf_dataset.py \
  --datasets nemotron-cc-high-propella-custom-eng \
  --output-name en/trad_single_pairs_bertscore_bleurt \
  --include-methods trad_single \
  --include-runs trad-single-balanced-v2 \
  --include-layers 1 \
  --pair-policy original_only \
  --score-names bertscore_f1 bleurt \
  --score-run-ids bertscore-trad-v1 bleurt-trad-v1
```

This emits one row per original--perturbation pair. Each row has two aligned
values in `texts`, `labels`, `bertscore_f1`, and `bleurt`; item order is
intentionally shuffled, so use the aligned `labels` or candidate metadata
rather than assuming a fixed left/right position.

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
