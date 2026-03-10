"""Utilities for persisting experiment results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Return filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Save experiment result JSON in output directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ts = utc_timestamp()
    path = output / f"{ts}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return path
