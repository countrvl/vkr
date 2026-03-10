"""Консольная обертка над раннерами экспериментов."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.batch_runner import run_batch
from src.inference.runner import run_experiment
from src.utils.config_loader import load_yaml
from src.utils.env_loader import load_dotenv_file

load_dotenv_file(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск NL2SQL-бенчмарка")
    parser.add_argument(
        "--config",
        default=os.getenv("L2SB_EXPERIMENTS_CONFIG", str(PROJECT_ROOT / "configs" / "experiments.yaml")),
        help="Путь к YAML-файлу экспериментов",
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("L2SB_EXPERIMENT"),
        help="Имя эксперимента. Если не указано, запускаются все эксперименты из конфига.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("L2SB_K", "0")) or None,
        help="Переопределить k генераций для выбранного(ых) эксперимента(ов)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    if args.experiment:
        experiments = config.get("experiments", {})
        if args.experiment not in experiments:
            raise KeyError(f"Эксперимент не найден: {args.experiment}")

        exp_cfg = dict(experiments[args.experiment])
        exp_cfg["models_config_path"] = config.get(
            "models_config_path",
            os.getenv("L2SB_MODELS_CONFIG_PATH", "configs/models.yaml"),
        )
        if args.k is not None:
            exp_cfg["k"] = args.k

        result = run_experiment(exp_cfg)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    results = run_batch(args.config, k_override=args.k)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
