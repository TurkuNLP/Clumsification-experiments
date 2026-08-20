# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy inference adapter; use :mod:`clumsification_code.evals.inference.fe`."""

from clumsification_code.evals.inference.fe import *  # noqa: F403
from clumsification_code.evals.inference.fe import FEInferenceModel, load_fe_inference_model

LTRInferenceModel = FEInferenceModel
load_ltr_inference_model = load_fe_inference_model
