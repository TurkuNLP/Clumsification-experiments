# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy module-name adapter for :mod:`gpt_batch_perturbation`."""

import sys

from clumsification_code.perturbations.gpt_batch_perturbation import *  # noqa: F403
from clumsification_code.perturbations.gpt_batch_perturbation import main


if __name__ == "__main__":
    main(sys.argv[1:])
