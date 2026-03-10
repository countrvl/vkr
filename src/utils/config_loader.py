"""Утилиты загрузки YAML-конфигов экспериментов и моделей."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Загрузить YAML-файл и вернуть словарь.

    Поддерживается интерполяция env-переменных в строковых полях:
    - `${VAR}`: обязательная переменная окружения
    - `${VAR:-default}`: значение по умолчанию при отсутствии переменной

    Args:
        path: Путь к YAML-файлу.

    Returns:
        Содержимое YAML в виде словаря.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если корень YAML не является отображением.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл конфига не найден: {path_obj}")

    with path_obj.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Корень YAML должен быть отображением: {path_obj}")

    return _expand_env(data)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _expand_env_string(value)
    return value


def _expand_env_string(raw: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        default = match.group(2)
        resolved = os.getenv(var)
        if resolved is not None and resolved != "":
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Переменная окружения `{var}` не установлена")

    return _ENV_PATTERN.sub(repl, raw)
