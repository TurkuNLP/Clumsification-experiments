# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy model-name adapter."""

from clumsification_code.fe.modeling import *  # noqa: F403
from clumsification_code.fe.modeling import FEModel


class LTRModel(FEModel):
    """Compatibility class exposing the pre-FE method and attribute names."""

    @property
    def scorer(self):
        return self.evaluation_head

    def score_flat(self, input_ids, attention_mask):
        return self.score_text_batch(input_ids, attention_mask)
