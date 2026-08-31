# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Tools for producing scalar supervision from custom text datasets."""

from .custom_dataset import (
    SUPPORTED_SCORING_TYPES,
    ScoreFailure,
    ScoreTask,
    load_score_tasks,
    score_custom_dataset,
    score_with_failure_isolation,
)

__all__ = [
    "SUPPORTED_SCORING_TYPES",
    "ScoreFailure",
    "ScoreTask",
    "load_score_tasks",
    "score_custom_dataset",
    "score_with_failure_isolation",
]
