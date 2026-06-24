"""Deprecated compatibility wrapper.

Use:
    python -m clumsification_code.evals.run_benchmark --scorer ltr ...
"""

from __future__ import annotations

import sys

from clumsification_code.evals.run_benchmark import main


if __name__ == "__main__":
    if not any(arg.startswith("--scorer") for arg in sys.argv[1:]):
        sys.argv.insert(1, "--scorer=ltr")
    main()