"""Run inference for code-generation benchmarks."""

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
DOMAIN_ROOT = PROJECT_ROOT / "code"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(DOMAIN_ROOT))

from code_bench.data.loader import load_benchmark
from code_bench.inference.api_backend import APIBackend
from code_bench.inference.ollama_backend import OllamaBackend
from code_bench.inference.runner import ExperimentRunner
from code_bench.prompt.template import PromptBuilder
from shared.config import load_domain_models
from shared.logging_utils import configure_logging, create_progress


LOGGER = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    benchmarks_cfg = _load_yaml(config_dir / "benchmarks.yaml")
    return {
        "data_dir": PROJECT_ROOT / benchmarks_cfg.get("data_dir", "data/code"),
        "results_dir": PROJECT_ROOT / benchmarks_cfg.get("results_dir", "results/code/raw"),
    }


def _load_models_config(config_dir: Path) -> dict[str, Any]:
    _ = config_dir
    return load_domain_models("supports_code")


def _resolve_model_keys(model_arg: str, models_cfg: dict[str, Any]) -> list[str]:
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

    def default_keys(prefix: str | None = None) -> list[str]:
        keys = []
        for key, cfg in models_cfg.items():
            if prefix is not None and not key.startswith(prefix):
                continue
            if cfg.get("active_by_default", True):
                keys.append(key)
        return keys

    for token in tokens:
        if token == "all":
            add_keys(default_keys())
        elif token == "m1":
            add_keys(default_keys("m1_"))
        elif token == "m2":
            add_keys(default_keys("m2_"))
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


def _normalize_mode(mode: str) -> str:
    return "fc" if mode == "ea" else mode


def _build_backend(model_key: str, model_cfg: dict[str, Any]):
    backend = model_cfg["backend"]
    base_url = model_cfg["base_url"]
    base_url_env = model_cfg.get("base_url_env")
    model_id = model_cfg["model_id"]
    model_id_env = model_cfg.get("model_id_env")
    if base_url_env:
        base_url = os.getenv(base_url_env, base_url)
    if model_id_env:
        model_id = os.getenv(model_id_env, model_id)

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
        )
    if backend == "ollama":
        parameters = dict(model_cfg.get("parameters", {}))
        num_ctx = int(parameters.pop("num_ctx", 8192))
        return OllamaBackend(
            model_id=model_id,
            base_url=base_url,
            num_ctx=num_ctx,
            model_name=model_cfg["name"],
            parameters=parameters,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _resolve_mode_params(mode: str, exp_cfg: dict[str, Any]) -> tuple[str, float, int, int | None]:
    normalized = _normalize_mode(mode)
    if normalized == "fc":
        return normalized, 0.0, 1, exp_cfg.get("seed")
    if normalized == "pass_k":
        return normalized, float(exp_cfg["temperature_pass_k"]), max(exp_cfg["k_values"]), None
    raise ValueError(f"Unsupported mode: {mode}")


async def _run(
    args: argparse.Namespace,
    exp_cfg: dict[str, Any],
    benchmarks_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    run_label: str,
    temperature: float,
    n: int,
    seed: int | None,
) -> None:
    prompt_builder = PromptBuilder()
    benchmark_names = list(benchmarks_cfg["benchmarks"]) if args.benchmark == "all" else [args.benchmark]
    model_keys = _resolve_model_keys(args.model, models_cfg)
    top_p: float | None = exp_cfg.get("top_p")
    total_groups = len(model_keys) * len(benchmark_names)
    with create_progress() as progress:
        groups_task = progress.add_task("Inference groups", total=total_groups, status="")
        for model_key in model_keys:
            model_cfg = models_cfg[model_key]
            max_tokens = int(model_cfg.get("max_tokens", exp_cfg.get("max_tokens", 768)))
            prompt_profile = str(model_cfg.get("prompt_profile", "codegen_default"))
            backend = _build_backend(model_key, model_cfg)
            runner = ExperimentRunner(
                backend=backend,
                prompt_builder=prompt_builder,
                output_dir=args.results_dir,
            )
            for benchmark in benchmark_names:
                progress.update(groups_task, status=f"{model_key}/{benchmark}/{run_label}")
                benchmark_cfg = benchmarks_cfg["benchmarks"][benchmark]
                samples = load_benchmark(
                    benchmark,
                    args.data_dir,
                    mini=bool(args.mini and benchmark_cfg.get("supports_mini", False)),
                    noextreme=bool(benchmark_cfg.get("noextreme", False)),
                )
                if args.limit is not None:
                    samples = samples[: args.limit]
                    LOGGER.info("Limiting to %d samples for %s/%s.", args.limit, model_key, benchmark)
                output_path = await runner.run(
                    samples,
                    model_key=model_key,
                    model_name=model_cfg["name"],
                    model_display_name=model_cfg.get("display_name"),
                    model_version=model_cfg.get("version"),
                    benchmark=benchmark,
                    run_label=run_label,
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
    configure_logging(logging.INFO)
    load_dotenv()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)
    _load_models_config(config_args.config_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model selector: a single key from shared/configs/models.yaml "
            "with supports_code=true, "
            "'all', 'm1', 'm2', or a comma-separated combination."
        ),
    )
    parser.add_argument("--benchmark", choices=["humaneval_plus", "mbpp_plus", "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["fc", "ea", "pass_k"],
        default="fc",
        help="fc/ea: temp=0, n=1. pass_k: temp from config, n=max(k_values).",
    )
    parser.add_argument("--config-dir", type=Path, default=config_args.config_dir)
    parser.add_argument("--data-dir", type=Path, default=defaults["data_dir"])
    parser.add_argument("--results-dir", type=Path, default=defaults["results_dir"])
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Limit samples per benchmark.")
    parser.add_argument("--mini", action="store_true", help="Use HumanEval+ mini where supported.")
    args = parser.parse_args()

    benchmarks_cfg = _load_yaml(args.config_dir / "benchmarks.yaml")
    exp_cfg = _load_yaml(args.config_dir / "experiment.yaml")
    models_cfg = _load_models_config(args.config_dir)
    _resolve_model_keys(args.model, models_cfg)
    run_label, temperature, n, seed = _resolve_mode_params(args.mode, exp_cfg)
    asyncio.run(_run(args, exp_cfg, benchmarks_cfg, models_cfg, run_label, temperature, n, seed))


if __name__ == "__main__":
    main()
