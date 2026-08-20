# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy checkpoint API adapter."""

from clumsification_code.fe.checkpointing import *  # noqa: F403
from clumsification_code.fe.checkpointing import load_fe_model

load_ltr_model = load_fe_model
