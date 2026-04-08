"""Скачать наборы данных Spider и BIRD."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nl2sql.src.data.download import download_bird, download_spider
from shared.logging_utils import configure_logging


def main() -> None:
    """Скачать выбранные benchmark-наборы данных."""
    configure_logging(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "nl2sql")
    parser.add_argument("--benchmark", choices=["spider", "bird", "all"], default="all")
    args = parser.parse_args()

    if args.benchmark in ("spider", "all"):
        download_spider(args.data_dir)
    if args.benchmark in ("bird", "all"):
        download_bird(args.data_dir)


if __name__ == "__main__":
    main()
