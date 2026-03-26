"""Run NL2SQL inference over configured benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.data.loader import load_benchmark
from src.inference.api_backend import APIBackend
from src.inference.ollama_backend import OllamaBackend
from src.inference.runner import ExperimentRunner
from src.prompt.template import PromptBuilder


LOGGER = logging.getLogger(__name__)


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    """Load CLI defaults sourced from experiment.yaml."""
    config_path = config_dir / "experiment.yaml"
    if not config_path.exists():
        return {"data_dir": Path("data"), "results_dir": Path("results/raw")}

    with config_path.open("r", encoding="utf-8") as handle:
        exp_cfg = yaml.safe_load(handle) or {}
    return {
        "data_dir": Path(exp_cfg.get("data_dir", "data")),
        "results_dir": Path(exp_cfg.get("results_dir", "results/raw")),
    }


def _load_models_config(config_dir: Path) -> dict[str, Any]:
    with (config_dir / "models.yaml").open("r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {})["models"]


def _build_backend(model_key: str, model_cfg: dict[str, Any]):
    """Build an inference backend from model configuration."""
    backend = model_cfg["backend"]
    if backend == "api":
        env_key = model_cfg["env_key"]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_key} is required for {model_key}")
        return APIBackend(
            model_id=model_cfg["model_id"],
            base_url=model_cfg["base_url"],
            api_key=api_key,
            model_name=model_cfg["name"],
            parameters=model_cfg.get("parameters", {}),
            pricing=model_cfg.get("pricing"),
        )
    if backend == "ollama":
        parameters = dict(model_cfg.get("parameters", {}))
        num_ctx = int(parameters.pop("num_ctx", 4096))
        return OllamaBackend(
            model_id=model_cfg["model_id"],
            base_url=model_cfg["base_url"],
            num_ctx=num_ctx,
            model_name=model_cfg["name"],
            parameters=parameters,
        )
    raise ValueError(f"Unsupported backend: {backend}")


async def _run(
    args: argparse.Namespace,
    exp_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    temperature: float,
    n: int,
) -> None:
    """Create backends, load data, and run inference."""
    prompt_builder = PromptBuilder()
    benchmark_names = exp_cfg["benchmarks"] if args.benchmark == "all" else [args.benchmark]
    model_keys = list(models_cfg) if args.model == "all" else [args.model]
    max_tokens = int(exp_cfg.get("max_tokens", 512))
    seed: int | None = exp_cfg.get("seed")
    top_p: float | None = exp_cfg.get("top_p")
    total_groups = len(model_keys) * len(benchmark_names)
    groups_progress = tqdm(total=total_groups, desc="Inference groups", unit="group")

    try:
        for model_key in model_keys:
            model_cfg = models_cfg[model_key]
            backend = _build_backend(model_key, model_cfg)
            runner = ExperimentRunner(
                backend=backend,
                prompt_builder=prompt_builder,
                output_dir=args.results_dir,
                data_root=args.data_dir,
            )
            for benchmark in benchmark_names:
                groups_progress.set_postfix(model=model_key, benchmark=benchmark, mode=args.mode)
                samples = load_benchmark(benchmark, args.data_dir)
                if args.limit is not None:
                    samples = samples[: args.limit]
                    LOGGER.info("Limiting to %d samples for %s/%s.", args.limit, model_key, benchmark)
                output_path = await runner.run(
                    samples,
                    model_name=model_cfg["name"],
                    benchmark=benchmark,
                    run_label=args.mode,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )
                LOGGER.info("Saved raw generations to %s", output_path)
                groups_progress.update(1)
    finally:
        groups_progress.close()


def main() -> None:
    """Parse args, load config, and start the async inference workflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)

    models_cfg = _load_models_config(config_args.config_dir)
    model_choices = sorted(models_cfg.keys()) + ["all"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=model_choices, required=True)
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

    if args.mode == "ea":
        temperature = 0.0
        n = 1
    else:
        temperature = exp_cfg["temperature_pass_k"]
        n = max(exp_cfg["k_values"])

    asyncio.run(_run(args, exp_cfg, models_cfg, temperature, n))


if __name__ == "__main__":
    main()
