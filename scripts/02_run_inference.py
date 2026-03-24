"""Run NL2SQL inference over configured benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config import load_yaml_config
from src.data.loader import load_benchmark
from src.inference.api_backend import ApiInferenceBackend
from src.inference.ollama_backend import OllamaInferenceBackend
from src.inference.runner import ExperimentRunner
from src.prompt.template import PromptBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured NL2SQL inference.")
    parser.add_argument("--model", choices=["m1_frontier", "m2_compact", "all"], default="all")
    parser.add_argument("--benchmark", choices=["spider", "bird", "all"], default="all")
    parser.add_argument("--mode", choices=["ea", "pass_k"], default="ea")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw"))
    return parser.parse_args()


def build_backend(model_key: str, model_cfg: dict[str, Any]):
    backend = model_cfg["backend"]
    if backend == "api":
        env_key = model_cfg["env_key"]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_key} is required for {model_key}")
        return ApiInferenceBackend(
            api_key=api_key,
            base_url=model_cfg["base_url"],
            model_id=model_cfg["model_id"],
            model_name=model_cfg["name"],
            parameters=model_cfg.get("parameters", {}),
        )
    if backend == "ollama":
        return OllamaInferenceBackend(
            base_url=model_cfg["base_url"],
            model_id=model_cfg["model_id"],
            model_name=model_cfg["name"],
            parameters=model_cfg.get("parameters", {}),
        )
    raise ValueError(f"Unsupported backend: {backend}")


async def run() -> None:
    load_dotenv()
    args = parse_args()
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    models_cfg = load_yaml_config(args.config_dir / "models.yaml")["models"]

    model_keys = list(models_cfg) if args.model == "all" else [args.model]
    benchmarks = experiment_cfg["benchmarks"] if args.benchmark == "all" else [args.benchmark]
    n = 1 if args.mode == "ea" else max(experiment_cfg["k_values"])
    temperature = experiment_cfg["temperature"] if args.mode == "ea" else experiment_cfg["temperature_pass_k"]

    for model_key in model_keys:
        model_cfg = models_cfg[model_key]
        backend = build_backend(model_key, model_cfg)
        runner = ExperimentRunner(
            experiment_config=experiment_cfg,
            backend=backend,
            prompt_builder=PromptBuilder(),
            output_dir=args.results_dir,
        )
        for benchmark in benchmarks:
            samples = load_benchmark(benchmark, args.data_dir)
            output_path = await runner.run(
                samples,
                benchmark=benchmark,
                model_name=model_key,
                n=n,
                temperature=temperature,
            )
            print(f"Saved raw generations to {output_path}")


if __name__ == "__main__":
    asyncio.run(run())
