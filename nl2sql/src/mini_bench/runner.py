"""Execution runner for the local NL2SQL mini-benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from nl2sql.src.data.loader import DataSample
from nl2sql.src.data.schema import serialize_schema
from nl2sql.src.inference.runtime import build_backend, resolve_model_runtime
from nl2sql.src.prompt.template import PromptBuilder
from nl2sql.src.mini_bench.prepare import DATASET_PATH, DB_PATH, build_snapshot_db, load_cases
from nl2sql.src.strategy_bench.executor import SQLiteExecutor
from nl2sql.src.strategy_bench.model import BackendModelAdapter
from nl2sql.src.strategy_bench.validation import ValidationModule
from shared.config import load_domain_models
from shared.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "nl2sql" / "mini_bench"


def _build_backend(model_key: str, model_cfg: dict[str, Any]):
    """Compatibility wrapper around the shared NL2SQL backend builder."""
    return build_backend(model_key, model_cfg)


def mini_case_to_sample(case: Any, *, db_path: Path) -> DataSample:
    """Adapt a local mini-benchmark case to the main NL2SQL prompt contract."""
    db_id = case.db_target or "mini_bench_snapshot"
    return DataSample(
        id=case.id,
        benchmark="mini_bench",
        question=case.natural_language_query,
        gold_sql=case.expected_sql or "",
        db_id=db_id,
        db_path=db_path,
        schema=serialize_schema(db_path),
        difficulty=str(case.metadata.get("difficulty", "")) or None,
        evidence=str(case.metadata.get("evidence", "")) or None,
    )


def aggregate_case_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [bool(row.get("accuracy")) for row in results if row.get("accuracy") is not None]
    execution_successes = [bool(row.get("execution_success")) for row in results]
    latencies = [float(row.get("model_latency_ms", 0.0)) for row in results]
    summary = {
        "n_cases": len(results),
        "execution_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "execution_success_rate": sum(execution_successes) / len(execution_successes) if execution_successes else 0.0,
        "model_latency_ms": mean(latencies) if latencies else 0.0,
    }
    return summary


def aggregate_breakdown(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    breakdown: list[dict[str, Any]] = []
    for value, rows in sorted(grouped.items()):
        accuracies = [bool(row.get("accuracy")) for row in rows if row.get("accuracy") is not None]
        execution_successes = [bool(row.get("execution_success")) for row in rows]
        breakdown.append(
            {
                key: value,
                "n_cases": len(rows),
                "execution_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
                "execution_success_rate": (
                    sum(execution_successes) / len(execution_successes) if execution_successes else 0.0
                ),
            }
        )
    return breakdown


def collect_failure_examples(results: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    failures = [row for row in results if row.get("accuracy") is False or not row.get("execution_success")]
    examples: list[dict[str, Any]] = []
    for row in failures[:limit]:
        examples.append(
            {
                "case_id": row["case_id"],
                "question": row["question"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "generated_sql": row["generated_sql"],
                "expected_sql": row["expected_sql"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            }
        )
    return examples


def error_type_counts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        row["error_type"]
        for row in results
        if row.get("error_type")
    )
    return [{"error_type": key, "count": value} for key, value in counts.most_common()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_mini_bench(
    *,
    model_key: str,
    dataset_path: Path = DATASET_PATH,
    db_path: Path = DB_PATH,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    models_cfg = load_domain_models("supports_sql")
    if model_key not in models_cfg:
        available = ", ".join(sorted(models_cfg))
        raise ValueError(f"Unknown model {model_key!r}. Available: {available}")

    build_snapshot_db(output_path=db_path)
    cases = load_cases(dataset_path)
    if limit is not None:
        cases = cases[:limit]
        LOGGER.info("Limiting mini-benchmark to %d cases", limit)

    model_cfg = models_cfg[model_key]
    backend = _build_backend(model_key, model_cfg)
    runtime = resolve_model_runtime(model_cfg)
    model = BackendModelAdapter(
        backend,
        temperature=float(runtime["temperature"]),
        max_tokens=int(runtime["max_tokens"]),
        seed=runtime["seed"],
        top_p=runtime["top_p"],
    )
    prompt_builder = PromptBuilder()
    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()

    resolved_output_dir = output_dir or RESULTS_ROOT / model_key
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for case in cases:
        sample = mini_case_to_sample(case, db_path=db_path)
        prompt = prompt_builder.build(sample, str(runtime["prompt_profile"]))
        response = model.generate_sql(prompt)
        validation = validator.validate(case, response.sql, executor)
        results.append(
            {
                "case_id": case.id,
                "question": case.natural_language_query,
                "category": str(case.metadata.get("category", "")),
                "difficulty": str(case.metadata.get("difficulty", "")),
                "expected_sql": case.expected_sql,
                "generated_sql": response.sql,
                "execution_success": bool(validation.success),
                "accuracy": validation.accuracy,
                "comparison_mode": validation.comparison_mode,
                "error_type": validation.error_type,
                "error_message": validation.error_message,
                "model_latency_ms": response.latency_ms,
                "model_key": model_key,
                "model_display_name": model_cfg["display_name"],
                "model_family": model_cfg["family"],
                "prompt_profile": runtime["prompt_profile"],
                "benchmark": sample.benchmark,
            }
        )

    summary = aggregate_case_results(results)
    by_category = aggregate_breakdown(results, "category")
    by_difficulty = aggregate_breakdown(results, "difficulty")
    failures = collect_failure_examples(results)
    errors = error_type_counts(results)

    write_json(resolved_output_dir / "per_case.json", results)
    write_csv(resolved_output_dir / "per_case.csv", results)
    write_json(
        resolved_output_dir / "summary_metrics.json",
        {
            "model_key": model_key,
            "model_display_name": model_cfg["display_name"],
            "model_family": model_cfg["family"],
            "prompt_profile": runtime["prompt_profile"],
            "summary": summary,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "failure_examples": failures,
            "error_types": errors,
        },
    )
    write_csv(resolved_output_dir / "summary_by_category.csv", by_category)
    write_csv(resolved_output_dir / "summary_by_difficulty.csv", by_difficulty)
    write_json(resolved_output_dir / "failure_examples.json", failures)
    write_json(resolved_output_dir / "error_types.json", errors)
    return {
        "summary": summary,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "failure_examples": failures,
        "error_types": errors,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local NL2SQL mini-benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    configure_logging()
    outcome = run_mini_bench(
        model_key=args.model,
        dataset_path=args.dataset,
        db_path=args.db_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    LOGGER.info(
        "Mini-benchmark finished: EA=%s over %d cases",
        outcome["summary"]["execution_accuracy"],
        outcome["summary"]["n_cases"],
    )


if __name__ == "__main__":
    main()
