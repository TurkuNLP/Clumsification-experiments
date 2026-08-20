# FKTQE
Work in progress on automatically evaluating the quality of Finnish texts written for children

Python files in this repository carry an attribution notice where they have
been co-created, refactored, or cleaned with GPT 5.6. Earlier GPT-version
notices are retained where they already existed.

For a two-minute overview of the data, training, evaluation, and compatibility layers, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Producing regression supervision scores

Score perturbations against their source originals before creating the Hugging
Face training dataset.  Install the local scoring dependencies in the intended
environment (`torch`, `transformers`, `evaluate`, and `bert-score`), then run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type token_normalized_perplexity \
  --sample-limit 1000 \
  --seed 42
```

or, for source-based BERTScore F1:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type bertscore_f1 \
  --language fi \
  --batch-size 8
```

For BLEURT, install BLEURT and its TensorFlow dependency as well, then run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name <dataset> \
  --scoring-type bleurt
```

BLEURT defaults to its authors' recommended `BLEURT-20` checkpoint; override
it with `--bleurt-checkpoint` when necessary.

Each command creates `scores/<perturbation-folder>/<method>.jsonl`, an
accompanying error JSONL, and a metadata file within the selected custom
dataset. Score JSONL contains only
successful candidate scores; originals are not self-scored.  Every stored value
is higher-is-better: BERTScore is the Hugging Face Evaluate metric's raw F1
using its normal defaults for the selected language, BLEURT is used directly,
and perplexity is stored as `-log(perplexity)`.

When creating a regression dataset from partial scores, identify the score that
will be trained on so its source documents are split correctly:

```bash
python scripts/create_fe_training_dataset.py \
  --custom-datasets <dataset> \
  --score-names bleurt
```
