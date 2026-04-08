"""Запустить NL2SQL-инференс на выбранных benchmark-ах."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "nl2sql"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nl2sql.src.data.loader import load_benchmark
from nl2sql.src.inference.api_backend import APIBackend
from nl2sql.src.inference.ollama_backend import OllamaBackend
from nl2sql.src.inference.runner import ExperimentRunner
from shared.config import load_domain_models
from shared.logging_utils import configure_logging, create_progress
from nl2sql.src.prompt.template import PromptBuilder


LOGGER = logging.getLogger(__name__)


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    """Загрузить значения CLI по умолчанию из experiment.yaml."""
    config_path = config_dir / "experiment.yaml"
    if not config_path.exists():
        return {
            "data_dir": PROJECT_ROOT / "data" / "nl2sql",
            "results_dir": PROJECT_ROOT / "results" / "nl2sql" / "raw",
        }

    with config_path.open("r", encoding="utf-8") as handle:
        exp_cfg = yaml.safe_load(handle) or {}
    return {
        "data_dir": PROJECT_ROOT / exp_cfg.get("data_dir", "data/nl2sql"),
        "results_dir": PROJECT_ROOT / exp_cfg.get("results_dir", "results/nl2sql/raw"),
    }


def _load_models_config(config_dir: Path) -> dict[str, Any]:
    _ = config_dir
    return load_domain_models("supports_sql")


def _resolve_model_keys(model_arg: str, models_cfg: dict[str, Any]) -> list[str]:
    """Преобразовать selector моделей в упорядоченный список ключей.

    Поддерживаются:
    - один ключ модели, например ``m1_deepseek``
    - ``all`` для всех настроенных моделей
    - ``m1`` для всех ключей с префиксом ``m1_``
    - ``m2`` для всех ключей с префиксом ``m2_``
    - комбинации через запятую, например ``m1,m2_defog``
    """
    tokens = [token.strip() for token in model_arg.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Model selector must not be empty.")

    resolved: list[str] = []
    seen: set[str] = set()

    def add_keys(keys: list[str]) -> None:
        for key in keys:
            if key not in seen:
                seen.add(key)
                resolved.append(key)

    for token in tokens:
        if token == "all":
            add_keys(list(models_cfg))
        elif token == "m1":
            add_keys([key for key in models_cfg if key.startswith("m1_")])
        elif token == "m2":
            add_keys([key for key in models_cfg if key.startswith("m2_")])
        elif token in models_cfg:
            add_keys([token])
        else:
            available = ", ".join(sorted(models_cfg))
            raise ValueError(
                f"Unknown model selector '{token}'. Use one of: all, m1, m2, {available}"
            )

    if not resolved:
        raise ValueError(f"Model selector '{model_arg}' did not resolve to any configured models.")
    return resolved


def _build_backend(model_key: str, model_cfg: dict[str, Any]):
    """Собрать backend инференса из конфигурации модели."""
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


def _resolve_mode_params(mode: str, exp_cfg: dict[str, Any]) -> tuple[float, int, int | None]:
    """Вернуть `(temperature, n, seed)` для выбранного режима инференса."""
    if mode == "ea":
        return 0.0, 1, exp_cfg.get("seed")
    if mode == "pass_k":
        return exp_cfg["temperature_pass_k"], max(exp_cfg["k_values"]), None
    raise ValueError(f"Unsupported mode: {mode}")


async def _run(
    args: argparse.Namespace,
    exp_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    temperature: float,
    n: int,
    seed: int | None,
) -> None:
    """Создать backend-ы, загрузить данные и выполнить инференс."""
    prompt_builder = PromptBuilder()
    benchmark_names = exp_cfg["benchmarks"] if args.benchmark == "all" else [args.benchmark]
    model_keys = _resolve_model_keys(args.model, models_cfg)
    top_p: float | None = exp_cfg.get("top_p")
    total_groups = len(model_keys) * len(benchmark_names)
    with create_progress() as progress:
        groups_task = progress.add_task("Inference groups", total=total_groups, status="")
        for model_key in model_keys:
            model_cfg = models_cfg[model_key]
            max_tokens = int(model_cfg.get("max_tokens", exp_cfg.get("max_tokens", 512)))
            prompt_profile = str(model_cfg.get("prompt_profile", "nl2sql_json"))
            backend = _build_backend(model_key, model_cfg)
            runner = ExperimentRunner(
                backend=backend,
                prompt_builder=prompt_builder,
                output_dir=args.results_dir,
                data_root=args.data_dir,
            )
            for benchmark in benchmark_names:
                progress.update(groups_task, status=f"{model_key}/{benchmark}/{args.mode}")
                samples = load_benchmark(benchmark, args.data_dir)
                if args.limit is not None:
                    samples = samples[: args.limit]
                    LOGGER.info("Limiting to %d samples for %s/%s.", args.limit, model_key, benchmark)
                output_path = await runner.run(
                    samples,
                    model_name=model_cfg["name"],
                    benchmark=benchmark,
                    model_display_name=model_cfg.get("display_name"),
                    model_version=model_cfg.get("version"),
                    model_key=model_key,
                    run_label=args.mode,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    prompt_profile=prompt_profile,
                    progress=progress,
                )
                LOGGER.info("Saved raw generations to %s", output_path)
                progress.update(groups_task, advance=1)


def main() -> None:
    """Разобрать аргументы, загрузить конфиг и запустить async-инференс."""
    configure_logging(logging.INFO)
    load_dotenv()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)

    models_cfg = _load_models_config(config_args.config_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model selector: a single key from shared/configs/models.yaml "
            "with supports_sql=true, "
            "'all', 'm1', 'm2', or a comma-separated combination."
        ),
    )
    parser.add_argument("--benchmark", choices=["spider", "bird", "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["ea", "pass_k"],
        default="ea",
        help="ea: temp=0, n=1. pass_k: temp from config, n=K",
    )
    parser.add_argument("--config-dir", type=Path, default=config_args.config_dir)
    parser.add_argument("--data-dir", type=Path, default=defaults["data_dir"])
    parser.add_argument("--results-dir", type=Path, default=defaults["results_dir"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of samples per benchmark (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    with (args.config_dir / "experiment.yaml").open("r", encoding="utf-8") as handle:
        exp_cfg = yaml.safe_load(handle) or {}
    models_cfg = _load_models_config(args.config_dir)
    _resolve_model_keys(args.model, models_cfg)

    temperature, n, seed = _resolve_mode_params(args.mode, exp_cfg)

    asyncio.run(_run(args, exp_cfg, models_cfg, temperature, n, seed))


if __name__ == "__main__":
    main()
