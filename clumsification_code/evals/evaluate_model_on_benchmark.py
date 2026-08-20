# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Legacy LTR benchmark entry point routing to the canonical FE backend."""

from __future__ import annotations

import sys

from clumsification_code.evals.run_benchmark import main


if __name__ == "__main__":
    if not any(argument.startswith("--scorer") for argument in sys.argv[1:]):
        sys.argv.insert(1, "--scorer=fe")
    main()
