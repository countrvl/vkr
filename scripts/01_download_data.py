"""Download benchmark data scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.download import download_all, download_bird, download_spider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Spider/BIRD data directories.")
    parser.add_argument("--benchmark", choices=["spider", "bird", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark == "spider":
        path = download_spider(args.data_dir)
        print(f"Prepared Spider directory: {path}")
    elif args.benchmark == "bird":
        path = download_bird(args.data_dir)
        print(f"Prepared BIRD directory: {path}")
    else:
        paths = download_all(args.data_dir)
        print(f"Prepared datasets: {paths}")


if __name__ == "__main__":
    main()
