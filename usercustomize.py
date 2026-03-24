"""Project-wide interpreter tweaks for local development tools."""

from __future__ import annotations

import os
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
