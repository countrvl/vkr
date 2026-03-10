"""Batch experiment runner from YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: list[str], **_: Any) -> list[str]:
        """Fallback when tqdm is unavailable."""
        return iterable

from src.inference.runner import run_experiment
from src.utils.config_loader import load_yaml
from src.utils.env_loader import load_dotenv_file


def run_batch(
    config_path: str | Path,
    experiment_names: list[str] | None = None,
    k_override: int | None = None,
) -> list[dict[str, Any]]:
    """Run a set of experiments from config and return all results."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv_file(root / ".env")

    config_file = Path(config_path)
    config = load_yaml(config_file)

    experiments = config.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("`experiments` must be a mapping in experiments config")

    env_names = os.getenv("L2SB_EXPERIMENT_NAMES", "").strip()
    names_from_env = [name.strip() for name in env_names.split(",") if name.strip()] if env_names else None

    selected_names = experiment_names or names_from_env or list(experiments.keys())
    models_config_path = config.get(
        "models_config_path",
        os.getenv("L2SB_MODELS_CONFIG_PATH", "configs/models.yaml"),
    )

    results: list[dict[str, Any]] = []
    for name in tqdm(selected_names, desc="Running experiments"):
        if name not in experiments:
            raise KeyError(f"Experiment not found in config: {name}")

        exp_cfg = dict(experiments[name])
        exp_cfg["name"] = name
        exp_cfg["models_config_path"] = models_config_path
        if k_override is not None:
            exp_cfg["k"] = k_override
        result = run_experiment(exp_cfg)
        results.append(result)

    return results
