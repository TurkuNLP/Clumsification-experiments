# Short English-dev HPO

Run the same grid separately for each formatted traditional, LLM-only, and mixed
training dataset (and each single/sampled variant), using a fresh output directory:

```bash
sbatch updated_sbatch_jobs/run_hpo.sh /path/to/formatted_dataset /path/to/new_hpo_output
```

The default config has **54 trials per dataset: 27 hinge and 27 logistic**.
Each loss crosses learning rates `1e-5, 5e-5, 2e-4` and per-GPU batch sizes
`4, 8, 16` with its own parameter: hinge margins `0.05, 0.2, 0.8` or
logistic scalars `0.5, 5, 50`. This includes the old LR/margin/scalar settings.
Context length stays fixed at the launcher's 32768; override it when running the
Python runner directly. Edit the JSON grid to narrow or widen the study.
Explicit lists of trial objects are still supported.

Every trial trains once on the same seeded selection of **4,992 training rows**
(pairwise rows are pairs, counted after flattening), with no early stopping or
intermediate evaluation. This is a sample budget, not 5,000 optimizer steps.
On the launcher's eight GPUs, global batches 32/64/128 take 156/78/39 updates.
The budget must divide the global batch size, including gradient accumulation;
the training dataset must contain at least 4,992 rows.

Only `final/` is evaluated on the English dev panel: ELLIPSE grammar/cohesion,
JFLEG correction preference, and CoheSentia holistic/incremental coherence.
Scores within each dataset are averaged, then the three dataset scores receive
equal weight. `external_dev_hpo_runs.csv` retains all three scores, their mean,
the hyperparameters, and the budget. `best_external_dev_trial.json` selects the
best endpoint; `best_external_dev_hinge.json` and `best_external_dev_logistic.json`
retain separate loss winners. Compare the score columns between dataset runs
and the existing `fe_cps_csvs` results. Held-out test is never evaluated.

Use a fresh output directory after changing the config or budget. `--resume`
skips completed training and existing evaluations; it is for continuing the
same study. To inspect commands without training:

```bash
python -m scripts.run_hpo --external_dev_hpo \
  --formatted_dataset_path /path/to/formatted_dataset \
  --cuda_visible_devices 0,1,2,3,4,5,6,7 \
  --output_root /tmp/fe_hpo_preview --dry_run
```

## BLEURT regression pilot

The runner also supports `--training_method regression --score_name bleurt`.
This automatically selects `fe_regression_pilot.json`: **15 trials**, with
per-device training batch size fixed at **8** and accumulation fixed at **1**:

| Loss | Parameter | Learning rates | Trials |
| --- | --- | --- | --- |
| Huber | delta = 0.05, 0.2, 1.0 | 1e-5, 5e-5, 2e-4 | 9 |
| MSE | none | 1e-5, 5e-5, 2e-4 | 3 |
| MAE | none | 1e-5, 5e-5, 2e-4 | 3 |

The trainer scales targets using the training split's min/max, so these deltas
refer to that normalized scale. `smooth_l1`/`smoothl1` are aliases for Huber in
this codebase, so they are not additional loss variants.

Launch on the same cluster setup and English dev panel as pairwise:

```bash
sbatch updated_sbatch_jobs/run_hpo.sh \
  /path/to/bleurt_scored_formatted_dataset /path/to/new_bleurt_hpo_output \
  regression bleurt
```

Input must contain grouped `texts`, `labels`, `candidate_ids`,
`perturbation_sources`, and aligned `bleurt` scores in train/dev/test splits.
An already flattened pairwise dataset containing only chosen/rejected texts
cannot supply regression targets. Missing/non-finite scores are filtered;
the training split needs at least 4,992 usable scored candidates. Originals
are included by default. Each trial uses the same 4,992 candidates; on eight
GPUs this is a global batch of 64 and 78 optimizer updates.

Pass `--extra_args --exclude-layer-zero-training` to exclude originals from
regression training while retaining them in the formatted dev and test splits.
The existing `--exclude-layer-zero` excludes originals from all three splits.

To preview all 15 commands without training:

```bash
python -m scripts.run_hpo --external_dev_hpo \
  --training_method regression --score_name bleurt \
  --formatted_dataset_path /path/to/bleurt_scored_formatted_dataset \
  --cuda_visible_devices 0,1,2,3,4,5,6,7 \
  --output_root /tmp/bleurt_hpo_preview --dry_run
```

Omit `--external_dev_hpo` to select by the formatted dev split's Spearman
correlation with BLEURT instead. Neither route evaluates held-out test.
Use `--trials_file` to override the default grid, and `--score_name` to reuse
the regression search for a different score field. External-dev summaries
include `huber_delta` and save separate winners for Huber, MSE, and MAE.

## Binary pilot

For a flat binary dataset with `text` and `label` columns, the default binary
grid runs **9 trials**: learning rates `1e-5`, `5e-5`, `2e-4` crossed with
per-device training batch sizes `4`, `8`, `16`. On eight GPUs with accumulation
1, these are global batches `32`, `64`, `128`, respectively. Each trial sees the
same seeded 4,992 training rows and makes `156`, `78`, or `39` optimizer updates.
Warmup ratio is 0.03 and weight decay is 0.01. Start with `5e-5` and batch 4
as the reference trial; the grid tests both lower and higher rates and batches.

```bash
sbatch updated_sbatch_jobs/run_hpo.sh \
  /path/to/binary_formatted_dataset /path/to/new_binary_hpo_output binary
```

The default config is `fe_binary_pilot.json`. The runner selects by the same
English external-dev panel as above. To select by formatted binary dev accuracy
instead, run `python -m scripts.run_hpo --training_method binary` with the dataset
path, eight GPU IDs, and a new output root, omitting `--external_dev_hpo`.
