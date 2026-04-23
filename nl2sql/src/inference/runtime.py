"""Shared runtime helpers for NL2SQL inference entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nl2sql.src.inference.anthropic_backend import AnthropicBackend
from nl2sql.src.inference.api_backend import APIBackend
from nl2sql.src.inference.ollama_backend import OllamaBackend
from shared.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = load_yaml_config(PROJECT_ROOT / "nl2sql" / "configs" / "experiment.yaml")


def build_backend(model_key: str, model_cfg: dict[str, Any]):
    """Build the NL2SQL backend using the same rules as the main inference script."""
    backend = model_cfg["backend"]
    base_url = model_cfg["base_url"]
    base_url_env = model_cfg.get("base_url_env")
    model_id = model_cfg["model_id"]
    model_id_env = model_cfg.get("model_id_env")
    if base_url_env:
        base_url = os.getenv(base_url_env, base_url)
    if model_id_env:
        model_id = os.getenv(model_id_env, model_id)
    structured_output = bool(model_cfg.get("structured_output", True))

    if backend == "api":
        env_key = model_cfg["env_key"]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_key} is required for {model_key}")
        return APIBackend(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_name=model_cfg["name"],
            parameters=model_cfg.get("parameters", {}),
            pricing=model_cfg.get("pricing"),
            structured_output=structured_output,
        )
    if backend == "anthropic":
        env_key = model_cfg["env_key"]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_key} is required for {model_key}")
        return AnthropicBackend(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_name=model_cfg["name"],
            parameters=model_cfg.get("parameters", {}),
            pricing=model_cfg.get("pricing"),
            use_batch=bool(model_cfg.get("batch_support")) and model_cfg.get("dispatch_preference") == "batch",
        )
    if backend == "ollama":
        parameters = dict(model_cfg.get("parameters", {}))
        num_ctx = int(parameters.pop("num_ctx", 4096))
        return OllamaBackend(
            model_id=model_id,
            base_url=base_url,
            num_ctx=num_ctx,
            model_name=model_cfg["name"],
            parameters=parameters,
            structured_output=structured_output,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def resolve_model_runtime(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime settings used by the main NL2SQL EA pipeline."""
    return {
        "max_tokens": int(model_cfg.get("max_tokens", EXPERIMENT_CONFIG.get("max_tokens", 512))),
        "prompt_profile": str(model_cfg.get("prompt_profile", "nl2sql_json")),
        "temperature": 0.0,
        "seed": EXPERIMENT_CONFIG.get("seed"),
        "top_p": EXPERIMENT_CONFIG.get("top_p"),
    }
