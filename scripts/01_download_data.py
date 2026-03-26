"""Download Spider and BIRD datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.download import download_bird, download_spider
from src.logging_utils import configure_logging


def main() -> None:
    """Download the requested benchmark datasets."""
    configure_logging(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--benchmark", choices=["spider", "bird", "all"], default="all")
    args = parser.parse_args()

    if args.benchmark in ("spider", "all"):
        download_spider(args.data_dir)
    if args.benchmark in ("bird", "all"):
        download_bird(args.data_dir)


if __name__ == "__main__":
    main()
