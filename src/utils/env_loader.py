"""Минимальный загрузчик .env без внешних зависимостей."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: str | Path, override: bool = False) -> None:
    """Загрузить переменные окружения из .env-файла.

    Args:
        path: Путь к .env-файлу.
        override: Если True, перезаписывать уже установленные переменные.
    """
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue

        if key in os.environ and not override:
            continue
        os.environ[key] = value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value
