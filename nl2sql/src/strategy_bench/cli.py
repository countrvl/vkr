"""CLI entrypoint for the NL2SQL strategy bench."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from nl2sql.src.inference.anthropic_backend import AnthropicBackend
from nl2sql.src.inference.api_backend import APIBackend
from nl2sql.src.inference.ollama_backend import OllamaBackend
from nl2sql.src.strategy_bench.dataset import DatasetLoader
from nl2sql.src.strategy_bench.executor import PostgresExecutor, SQLiteExecutor
from nl2sql.src.strategy_bench.model import BackendModelAdapter
from nl2sql.src.strategy_bench.retry import RetryPolicy
from nl2sql.src.strategy_bench.routing import RoutingCatalog, RuleBasedRouter
from nl2sql.src.strategy_bench.runner import ExperimentRunner
from shared.config import load_domain_models
from shared.logging_utils import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOMAIN_ROOT = PROJECT_ROOT / "nl2sql"


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NL2SQL strategy bench.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--strategy",
        choices=["generate_only", "generate_validate_retry", "routing", "all"],
        default="all",
    )
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "nl2sql" / "strategy_bench",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    parser.add_argument("--db-dsn-env", default="NL2SQL_STRATEGY_DB_DSN")
    parser.add_argument("--db-kind", choices=["postgres", "sqlite"], default="postgres")
    parser.add_argument("--sqlite-path", type=Path)
    return parser.parse_args()


def _build_backend(model_key: str, models_cfg: dict[str, Any]):
    model_cfg = models_cfg[model_key]
    backend = model_cfg["backend"]
    base_url = model_cfg["base_url"]
    model_id = model_cfg["model_id"]
    base_url_env = model_cfg.get("base_url_env")
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


def main() -> None:
    args = parse_args()
    load_dotenv()
    configure_logging()
    models_cfg = load_domain_models("supports_sql")
    if args.model not in models_cfg:
        available = ", ".join(sorted(models_cfg))
        raise ValueError(f"Unknown model {args.model!r}. Available: {available}")

    loader = DatasetLoader()
    cases = loader.load(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
        LOGGER.info("Limiting strategy bench to %d cases", args.limit)

    backend = _build_backend(args.model, models_cfg)
    model = BackendModelAdapter(backend)
    if args.db_kind == "sqlite":
        if args.sqlite_path is None:
            raise ValueError("--sqlite-path is required when --db-kind sqlite")
        executor = SQLiteExecutor(args.sqlite_path)
    else:
        dsn = os.getenv(args.db_dsn_env)
        if not dsn:
            raise RuntimeError(f"Environment variable {args.db_dsn_env} is required")
        executor = PostgresExecutor(dsn)

    router = None
    if args.catalog_path is not None:
        router = RuleBasedRouter(RoutingCatalog.from_yaml(args.catalog_path))

    runner = ExperimentRunner(
        model=model,
        executor=executor,
        retry_policy=RetryPolicy(max_attempts=args.max_attempts),
        router=router,
        output_dir=args.output_dir,
    )
    runner.run(cases, strategy=args.strategy)


if __name__ == "__main__":
    main()
