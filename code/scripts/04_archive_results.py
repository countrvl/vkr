"""Archive code-benchmark artifacts under results/code/archive/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.logging_utils import configure_logging


def main() -> None:
    configure_logging(logging.INFO)
    raise NotImplementedError("Code benchmark archive flow is not implemented yet.")


if __name__ == "__main__":
    main()

