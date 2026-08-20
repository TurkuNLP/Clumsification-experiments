# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy evaluation-name adapter."""

from clumsification_code.fe.evaluation import *  # noqa: F403
from clumsification_code.fe.evaluation import (
    baseline_pairwise_accuracies,
    evaluate_pairwise_accuracy_distributed,
)

baseline_winrates = baseline_pairwise_accuracies
evaluate_win_rate_distributed = evaluate_pairwise_accuracy_distributed
