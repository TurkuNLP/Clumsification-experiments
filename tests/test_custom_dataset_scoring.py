# This script has been co-created, refactored, and cleaned using GPT 5.6.
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clumsification_code.scoring.custom_dataset import (
    BERTScoreScorer,
    BLEURTScorer,
    ScoreTask,
    load_score_tasks,
    score_with_failure_isolation,
    select_original_ids,
)
from clumsification_code.scoring.args import parse_score_args


class CustomDatasetScoringTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as writer:
            for row in rows:
                writer.write(json.dumps(row) + "\n")

    def test_sampling_selects_source_ids_deterministically(self) -> None:
        first = select_original_ids(range(20), sample_limit=5, seed=123)
        second = select_original_ids(range(20), sample_limit=5, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 5)

    def test_loads_every_candidate_for_selected_originals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_jsonl(
                root / "original.jsonl",
                [
                    {"custom_id": 1, "text": "original one"},
                    {"custom_id": 2, "text": "original two"},
                ],
            )
            self._write_jsonl(
                root / "perturbed_layers" / "1.jsonl",
                [
                    {"head_id": 1, "text": "candidate one"},
                    {"head_id": 2, "text": "candidate two"},
                ],
            )
            self._write_jsonl(
                root / "perturbed_layers" / "2.jsonl",
                [{"head_id": 1, "text": "candidate three"}],
            )

            tasks, selected_ids = load_score_tasks(
                dataset_dir=root,
                layer_directory="perturbed_layers",
                sample_limit=1,
                seed=7,
            )

            self.assertEqual(len(selected_ids), 1)
            self.assertTrue(tasks)
            self.assertEqual({task.base_text_id for task in tasks}, set(selected_ids))
            self.assertTrue(all(task.source_layer == 0 for task in tasks))

    def test_failure_isolation_retains_successful_scores(self) -> None:
        tasks = [
            ScoreTask(1, 0, 1, "source", "good"),
            ScoreTask(2, 0, 1, "source", "bad"),
            ScoreTask(3, 0, 1, "source", "good again"),
        ]

        def scorer(batch):
            if any(task.target_text == "bad" for task in batch):
                raise ValueError("unscorable")
            return [0.25] * len(batch)

        scores, failures = score_with_failure_isolation(tasks, scorer)
        self.assertEqual(scores, [0.25, None, 0.25])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].task.base_text_id, 2)
        self.assertEqual(failures[0].error_type, "ValueError")

    def test_bertscore_uses_evaluate_defaults(self) -> None:
        class Metric:
            def __init__(self):
                self.arguments = None

            def compute(self, **kwargs):
                self.arguments = kwargs
                return {"f1": [0.75]}

        metric = Metric()
        scorer = BERTScoreScorer.__new__(BERTScoreScorer)
        scorer.metric = metric
        scorer.language = "fi"
        scorer.batch_size = 4

        self.assertEqual(
            scorer.score([ScoreTask(1, 0, 1, "original", "candidate")]),
            [0.75],
        )
        self.assertEqual(
            metric.arguments,
            {
                "predictions": ["candidate"],
                "references": ["original"],
                "lang": "fi",
                "batch_size": 4,
            },
        )

    def test_bleurt_uses_reference_and_candidate_texts(self) -> None:
        class Metric:
            def __init__(self):
                self.arguments = None

            def compute(self, **kwargs):
                self.arguments = kwargs
                return {"scores": [0.81]}

        metric = Metric()
        scorer = BLEURTScorer.__new__(BLEURTScorer)
        scorer.metric = metric

        self.assertEqual(
            scorer.score([ScoreTask(1, 0, 1, "original", "candidate")]),
            [0.81],
        )
        self.assertEqual(
            metric.arguments,
            {"predictions": ["candidate"], "references": ["original"]},
        )

    def test_score_arguments_have_project_defaults(self) -> None:
        with patch.object(sys, "argv", ["score_custom_dataset.py", "--dataset-name", "demo", "--scoring-type", "bertscore_f1"]):
            args = parse_score_args()

        self.assertEqual(args.dataset_name, "demo")
        self.assertEqual(args.language, "fi")
        self.assertEqual(args.layer_directory, "perturbed_layers")
        self.assertEqual(args.base_model, "Qwen/Qwen3-8B-Base")
        self.assertEqual(args.bleurt_checkpoint, "BLEURT-20")


if __name__ == "__main__":
    unittest.main()
