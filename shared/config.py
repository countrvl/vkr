"""Вспомогательные функции для YAML-конфигов экспериментов."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_CONFIG_DIR = PROJECT_ROOT / "shared" / "configs"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Загрузить YAML-файл в словарь."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}.")
    return data


def load_shared_models_config() -> dict[str, Any]:
    """Загрузить единый каталог моделей для всех доменов."""
    return load_yaml_config(SHARED_CONFIG_DIR / "models.yaml")


def _merge_model_override(base_cfg: dict[str, Any], override_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base_cfg)
    for key, value in override_cfg.items():
        if key == "parameters":
            parameters = dict(merged.get("parameters", {}))
            parameters.update(value)
            merged["parameters"] = parameters
        else:
            merged[key] = value
    return merged


def load_domain_models(capability: str) -> dict[str, Any]:
    """Вернуть модели, доступные для указанного доменного флага."""
    if capability not in {"supports_sql", "supports_code"}:
        raise ValueError(f"Unsupported model capability: {capability}")
    domain_key = "sql" if capability == "supports_sql" else "code"
    models = load_shared_models_config()["models"]
    domain_models: dict[str, Any] = {}
    for key, cfg in models.items():
        if not bool(cfg.get(capability)):
            continue
        merged_cfg = deepcopy(cfg)
        override_cfg = dict(cfg.get("domain_overrides", {}).get(domain_key, {}))
        if override_cfg:
            merged_cfg = _merge_model_override(merged_cfg, override_cfg)
        domain_models[key] = merged_cfg
    return domain_models
