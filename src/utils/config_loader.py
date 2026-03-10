"""Helpers for loading YAML experiment and model configuration files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary.

    Supports environment interpolation in scalar strings:
    - `${VAR}`: requires environment variable to be set
    - `${VAR:-default}`: uses default when variable is missing or empty

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML root is not a dictionary.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with path_obj.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path_obj}")

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
        raise ValueError(f"Environment variable `{var}` is not set")

    return _ENV_PATTERN.sub(repl, raw)
