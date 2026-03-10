"""Single experiment runner for NL2SQL benchmarking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.dataset.bird_loader import BirdLoader
from src.dataset.spider_loader import SpiderLoader
from src.evaluation.execution_accuracy import compute_execution_accuracy, execution_match
from src.evaluation.passk import compute_pass_at_k, pass_at_k
from src.logging.experiment_logger import save_run_result
from src.metrics.latency import average_latency, timed_call
from src.metrics.token_usage import aggregate_usage, extract_usage
from src.models.api_model import APIModel
from src.models.base_model import BaseModel
from src.models.ollama_model import OllamaModel
from src.prompts.prompt_templates import build_nl2sql_prompt
from src.utils.config_loader import load_yaml
from src.utils.env_loader import load_dotenv_file


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one NL2SQL experiment and persist results.

    Env-first overrides are supported for all key runtime parameters.
    """
    root = Path(__file__).resolve().parents[2]
    load_dotenv_file(root / ".env")
    cfg = _apply_env_overrides(config)

    models_config_path = Path(
        cfg.get(
            "models_config_path",
            os.getenv("L2SB_MODELS_CONFIG_PATH", str(root / "configs" / "models.yaml")),
        )
    )
    if not models_config_path.is_absolute():
        models_config_path = root / models_config_path

    models_config = load_yaml(models_config_path).get("models", {})

    dataset_name = str(_required(cfg, "dataset", "L2SB_DATASET")).lower()

    dataset_path_value = _required(cfg, "dataset_path", "L2SB_DATASET_PATH")
    dataset_path = Path(str(dataset_path_value))
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    model_key = str(_required(cfg, "model", "L2SB_MODEL_KEY"))
    k = int(cfg.get("k", 1))

    limit_raw = cfg.get("limit")
    limit = None if limit_raw in (None, "", "null") else int(limit_raw)

    records = _load_dataset(dataset_name, dataset_path)
    if limit is not None:
        records = records[:limit]

    model, model_info = _build_model(model_key, models_config)

    default_db_path = str(cfg.get("db_path", ""))
    output_dir = Path(cfg.get("output_dir", os.getenv("L2SB_OUTPUT_DIR", str(root / "results" / "runs"))))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    evaluated_rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    all_usages: list[dict[str, int | None]] = []

    for sample in records:
        prompt = build_nl2sql_prompt(sample["question"], sample["schema"])
        predictions: list[str] = []

        for _ in range(k):
            prediction, metadata, latency = timed_call(model.generate_with_metadata, prompt)
            predictions.append(prediction)
            all_latencies.append(latency)
            all_usages.append(extract_usage(metadata))

        db_path = _resolve_db_path(sample.get("db_path", ""), default_db_path, dataset_path)
        top_prediction = predictions[0] if predictions else ""

        is_correct = False
        passed_k = False
        if db_path:
            is_correct = execution_match(db_path, sample["gold_sql"], top_prediction)
            passed_k = pass_at_k(db_path, sample["gold_sql"], predictions, k)

        evaluated_rows.append(
            {
                "question": sample["question"],
                "gold_sql": sample["gold_sql"],
                "prediction": top_prediction,
                "predictions": predictions,
                "db_path": db_path,
                "is_correct": is_correct,
                f"pass_at_{k}": passed_k,
            }
        )

    result: dict[str, Any] = {
        "model": model_key,
        "model_name": model_info.get("model_name", ""),
        "model_backend": model_info.get("backend", ""),
        "dataset": dataset_name,
        "execution_accuracy": compute_execution_accuracy(evaluated_rows),
        "pass_at_k": compute_pass_at_k(evaluated_rows, k),
        "avg_latency": average_latency(all_latencies),
        "k": k,
        "num_samples": len(evaluated_rows),
        "token_usage": aggregate_usage(all_usages),
        "rows": evaluated_rows,
    }

    output_path = save_run_result(result, output_dir)
    result["result_path"] = str(output_path)
    return result


def _load_dataset(name: str, dataset_path: Path) -> list[dict[str, Any]]:
    loaders = {
        "spider": SpiderLoader,
        "bird": BirdLoader,
    }
    if name not in loaders:
        raise ValueError(f"Unsupported dataset: {name}")

    loader = loaders[name](dataset_path)
    return loader.load()


def _build_model(model_key: str, models_config: dict[str, Any]) -> tuple[BaseModel, dict[str, str]]:
    if model_key not in models_config:
        raise KeyError(f"Model key not found in models config: {model_key}")

    cfg = models_config[model_key]
    backend = str(cfg.get("backend", "")).lower()

    if backend == "ollama":
        model_name = _env_or_cfg("L2SB_OLLAMA_MODEL_NAME", cfg, "model_name")
        base_url = _env_or_cfg("L2SB_OLLAMA_BASE_URL", cfg, "base_url", default="http://localhost:11434")
        timeout = int(os.getenv("L2SB_MODEL_TIMEOUT", str(cfg.get("timeout", 120))))
        model = OllamaModel(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            options=cfg.get("options", {}),
        )
        return model, {"backend": backend, "model_name": model_name}

    if backend == "api":
        model_name = _env_or_cfg("L2SB_API_MODEL_NAME", cfg, "model_name")
        base_url = _env_or_cfg("L2SB_API_BASE_URL", cfg, "base_url", default="https://api.openai.com")
        temperature = float(os.getenv("L2SB_API_TEMPERATURE", str(cfg.get("temperature", 0.0))))
        max_tokens = int(os.getenv("L2SB_API_MAX_TOKENS", str(cfg.get("max_tokens", 256))))
        timeout = int(os.getenv("L2SB_MODEL_TIMEOUT", str(cfg.get("timeout", 120))))
        model = APIModel(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return model, {"backend": backend, "model_name": model_name}

    raise ValueError(f"Unsupported model backend: {backend}")


def _resolve_db_path(record_db_path: str, default_db_path: str, dataset_path: Path) -> str:
    selected = record_db_path or default_db_path
    if not selected:
        return ""

    db_path = Path(selected)
    if not db_path.is_absolute():
        db_path = dataset_path.parent / db_path
    return str(db_path)


def _env_or_cfg(env_key: str, cfg: dict[str, Any], cfg_key: str, default: str | None = None) -> str:
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    if cfg_key in cfg and str(cfg[cfg_key]) != "":
        return str(cfg[cfg_key])
    if default is not None:
        return default
    raise ValueError(f"Missing required config `{cfg_key}` and env `{env_key}`")


def _required(cfg: dict[str, Any], key: str, env_key: str) -> str:
    value = cfg.get(key)
    if value is not None and str(value) != "":
        return str(value)
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    raise ValueError(f"Missing required parameter `{key}` and env `{env_key}`")


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    mapping = {
        "dataset": "L2SB_DATASET",
        "dataset_path": "L2SB_DATASET_PATH",
        "model": "L2SB_MODEL_KEY",
        "k": "L2SB_K",
        "limit": "L2SB_LIMIT",
        "db_path": "L2SB_DB_PATH",
        "output_dir": "L2SB_OUTPUT_DIR",
        "models_config_path": "L2SB_MODELS_CONFIG_PATH",
    }
    for key, env_key in mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None and env_value != "":
            cfg[key] = env_value
    return cfg
