# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy regression entry point routing to the canonical FE trainer."""

from __future__ import annotations

import sys

from train_fe_model import main


if __name__ == "__main__":
    if "--training-method" not in sys.argv:
        sys.argv.extend(["--training-method", "regression"])
    main()
