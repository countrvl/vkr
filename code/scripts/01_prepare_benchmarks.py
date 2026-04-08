"""Подготовить code-бенчмарки, например HumanEval+ и MBPP+."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "code"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(DOMAIN_ROOT))

from code_bench.data.prepare import normalize_benchmark_name, prepare_benchmark_artifacts
from shared.config import load_yaml_config
from shared.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HumanEval+ and MBPP+ artifacts.")
    parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    parser.add_argument(
        "--benchmark",
        choices=["humaneval_plus", "mbpp_plus", "all"],
        default="all",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Prepare the HumanEval+ mini split when available.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging(logging.INFO)
    args = parse_args()
    benchmarks_cfg = load_yaml_config(args.config_dir / "benchmarks.yaml")
    data_dir = PROJECT_ROOT / benchmarks_cfg["data_dir"]
    benchmark_names = (
        list(benchmarks_cfg["benchmarks"])
        if args.benchmark == "all"
        else [normalize_benchmark_name(args.benchmark)]
    )

    for benchmark in benchmark_names:
        benchmark_cfg = benchmarks_cfg["benchmarks"][benchmark]
        benchmark_dir = PROJECT_ROOT / benchmark_cfg["local_dir"]
        mini = args.mini and bool(benchmark_cfg.get("supports_mini", False))
        if args.mini and not mini:
            LOGGER.info("%s does not support mini mode; preparing full dataset.", benchmark)
        manifest = prepare_benchmark_artifacts(
            benchmark,
            data_dir=data_dir,
            local_dir=benchmark_dir,
            mini=mini,
            noextreme=bool(benchmark_cfg.get("noextreme", False)),
        )
        LOGGER.info(
            "Prepared %s (%s samples, hash=%s) at %s",
            benchmark,
            manifest["n_samples"],
            manifest["dataset_hash"],
            benchmark_dir,
        )


if __name__ == "__main__":
    main()
