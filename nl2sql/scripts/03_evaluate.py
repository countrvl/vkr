"""Aggregate metrics from raw generation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "nl2sql"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import load_domain_models, load_yaml_config
from nl2sql.src.evaluation.ea import evaluate_candidate_predictions
from nl2sql.src.inference.base import normalize_sql_text
from nl2sql.src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from nl2sql.src.evaluation.pass_at_k import pass_at_k
from nl2sql.src.inference.base import GenerationResult
from shared.logging_utils import ProgressType, configure_logging, create_progress


LOGGER = logging.getLogger(__name__)
_LEGACY_RUN_LABEL = "legacy"
_MODEL_DISPLAY_LOOKUP = {
    cfg.get("name"): {
        "display_name": cfg.get("display_name") or cfg.get("name"),
        "version": cfg.get("version"),
        "key": key,
    }
    for key, cfg in load_domain_models("supports_sql").items()
}


def _model_display_name(record: dict[str, Any]) -> str:
    model_name = record.get("model_name")
    return (
        record.get("model_display_name")
        or _MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("display_name")
        or model_name
        or ""
    )


def _model_version(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_version") or _MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("version")


def _model_key(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_key") or _MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("key")


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    """Load CLI defaults sourced from experiment.yaml."""
    experiment_cfg = load_yaml_config(config_dir / "experiment.yaml")
    return {
        "raw_dir": PROJECT_ROOT / experiment_cfg.get("results_dir", "results/nl2sql/raw"),
        "data_dir": PROJECT_ROOT / experiment_cfg.get("data_dir", "data/nl2sql"),
    }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)

    parser = argparse.ArgumentParser(description="Evaluate raw NL2SQL generation results.")
    parser.add_argument("--config-dir", type=Path, default=config_args.config_dir)
    parser.add_argument("--raw-dir", type=Path, default=defaults["raw_dir"])
    parser.add_argument("--data-dir", type=Path, default=defaults["data_dir"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "nl2sql" / "metrics")
    parser.add_argument(
        "--run-label",
        choices=["ea", "pass_k", "all"],
        default="all",
        help="Which run_label to evaluate. Results are written into output-dir/<run_label>/.",
    )
    return parser.parse_args()


def _resolve_db_path(raw_db_path: str, data_dir: Path) -> Path:
    """Resolve a DB path stored in JSONL, supporting old and new formats."""
    db_path = Path(raw_db_path)
    if db_path.is_absolute():
        return db_path

    candidates = [db_path]
    if str(db_path).startswith(f"{data_dir.name}/"):
        candidates.append(data_dir.parent / db_path)
    else:
        candidates.append(data_dir / db_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Keep the most likely new-format path for error reporting downstream.
    return data_dir / db_path


def _normalize_run_label(record: dict[str, Any]) -> str:
    """Normalize run labels for new and legacy raw records."""
    run_label = record.get("run_label")
    if isinstance(run_label, str) and run_label.strip():
        return run_label.strip()
    return _LEGACY_RUN_LABEL


def _expected_generation_count(run_label: str, experiment_cfg: dict[str, Any]) -> int | None:
    """Return the expected generations per sample for a known run label."""
    if run_label == "ea":
        return 1
    if run_label == "pass_k":
        return max(int(k) for k in experiment_cfg["k_values"])
    return None


def _load_records(raw_dir: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load and group JSONL records by (model_name, benchmark, run_label).

    Each record is one sample with a ``generations`` list, as written by
    :class:`src.inference.runner.ExperimentRunner`.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed JSONL line in %s", path)
                    continue
                record["_source_path"] = str(path)
                key = (record["model_name"], record["benchmark"], _normalize_run_label(record))
                grouped[key].append(record)
    return grouped


def _validate_records(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    experiment_cfg: dict[str, Any],
    data_dir: Path,
) -> None:
    """Raise a clear error before evaluation when raw inputs are incompatible."""
    errors: list[str] = []
    grouped_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)

    for (model_name, benchmark, run_label), records in grouped.items():
        grouped_by_pair[(model_name, benchmark)].add(run_label)
        expected_generations = _expected_generation_count(run_label, experiment_cfg)

        for record in records:
            sample_id = record.get("sample_id", "<unknown>")
            source_path = record.get("_source_path", "<unknown>")
            generations = record.get("generations", [])
            if not generations:
                errors.append(
                    f"{source_path}: sample {sample_id} has no generations "
                    f"for {model_name}/{benchmark}/{run_label}"
                )
                continue

            if expected_generations is not None and len(generations) != expected_generations:
                errors.append(
                    f"{source_path}: sample {sample_id} has {len(generations)} generations, "
                    f"expected {expected_generations} for run_label={run_label}"
                )

            raw_db_path = record.get("db_path")
            if not isinstance(raw_db_path, str) or not raw_db_path.strip():
                errors.append(f"{source_path}: sample {sample_id} is missing db_path")
                continue
            resolved_db_path = _resolve_db_path(raw_db_path, data_dir)
            if not resolved_db_path.exists():
                errors.append(
                    f"{source_path}: sample {sample_id} db_path does not exist: "
                    f"raw={raw_db_path} resolved={resolved_db_path}"
                )

    for (model_name, benchmark), run_labels in grouped_by_pair.items():
        if len(run_labels) > 1 and _LEGACY_RUN_LABEL in run_labels:
            errors.append(
                f"Mixed legacy and labeled raw files for {model_name}/{benchmark}: {sorted(run_labels)}. "
                "Re-run inference into a clean raw directory or separate files by mode."
            )

    if errors:
        preview = "\n".join(errors[:10])
        remaining = len(errors) - 10
        if remaining > 0:
            preview += f"\n... and {remaining} more"
        raise ValueError(f"Raw evaluation input validation failed:\n{preview}")


def _evaluate_record(record: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Evaluate one raw record into sample-level outcomes."""
    gold_sql = record.get("gold_sql", "")
    db_path = _resolve_db_path(record["db_path"], data_dir)
    generations = record.get("generations", [])
    candidate_sql = [normalize_sql_text(gen["sql"]) for gen in generations]
    evaluation = evaluate_candidate_predictions(candidate_sql, gold_sql, db_path)
    candidate_hits = evaluation["candidate_hits"]

    return {
        "candidate_hits": candidate_hits,
        "sample_row": {
            "sample_id": record.get("sample_id"),
            "model_key": _model_key(record),
            "model_name": record.get("model_name"),
            "model_display_name": _model_display_name(record),
            "model_version": _model_version(record),
            "benchmark": record.get("benchmark"),
            "run_label": _normalize_run_label(record),
            "question": record.get("question", ""),
            "gold_sql": gold_sql,
            "db_id": record.get("db_id"),
            "db_path": str(db_path),
            "difficulty": record.get("difficulty"),
            "evidence": record.get("evidence"),
            "question_len": len(record.get("question", "")),
            "gold_sql_len": len(gold_sql),
            "n_generations": len(generations),
            "source_path": record.get("_source_path", ""),
            "gold_success": evaluation["gold_success"],
            "gold_error": evaluation["gold_error"],
            "candidate_hits": json.dumps(candidate_hits),
            "first_hit": bool(candidate_hits[0]),
            "any_hit": any(candidate_hits),
            "n_candidates": len(candidate_hits),
            "first_pred_sql": candidate_sql[0],
            "first_pred_success": evaluation["first_pred_success"],
            "first_pred_error": evaluation["first_pred_error"],
            "empty_sql": not str(candidate_sql[0]).strip(),
        },
    }


def _evaluate_records(
    records: list[dict[str, Any]],
    data_dir: Path,
    *,
    progress_label: str,
    progress: ProgressType,
) -> list[dict[str, Any]]:
    """Evaluate raw records, using processes for larger groups."""
    task_id = progress.add_task(progress_label, total=len(records), status="")
    if len(records) < 2:
        results: list[dict[str, Any]] = []
        for record in records:
            results.append(_evaluate_record(record, data_dir))
            progress.update(task_id, advance=1)
        progress.remove_task(task_id)
        return results

    max_workers = min(os.cpu_count() or 1, len(records))
    if max_workers <= 1:
        results = []
        for record in records:
            results.append(_evaluate_record(record, data_dir))
            progress.update(task_id, advance=1)
        progress.remove_task(task_id)
        return results

    results: list[dict[str, Any]] = [None] * len(records)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_evaluate_record, record, data_dir): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            results[future_to_idx[future]] = future.result()
            progress.update(task_id, advance=1)
    progress.remove_task(task_id)
    return results


def main() -> None:
    configure_logging(logging.INFO)
    args = parse_args()
    metrics_cfg = load_yaml_config(args.config_dir / "metrics.yaml")
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped = _load_records(args.raw_dir)
    _validate_records(grouped, experiment_cfg=experiment_cfg, data_dir=args.data_dir)
    with create_progress() as progress:
        selected_items = [
            item for item in grouped.items() if args.run_label == "all" or item[0][2] == args.run_label
        ]
        groups_task = progress.add_task("Evaluation groups", total=len(selected_items), status="")
        for (model_name, benchmark, run_label), records in selected_items:
            pass_results: list[list[bool]] = []
            generation_results: list[GenerationResult] = []
            evaluable_records = [record for record in records if record.get("generations")]
            progress_label = f"{model_name} / {benchmark} / {run_label}"
            progress.update(groups_task, status=progress_label)

            for record in records:
                if not record.get("generations"):
                    LOGGER.warning("Sample %s has no generations, skipping.", record.get("sample_id"))
                    continue
                for gen in record["generations"]:
                    generation_results.append(
                        GenerationResult(
                            sql=gen["sql"],
                            raw_response=gen["raw_response"],
                            tokens_input=gen["tokens_input"],
                            tokens_output=gen["tokens_output"],
                            latency_ms=gen["latency_ms"],
                            model_name=gen.get("model_name", model_name),
                            metadata=gen.get("metadata", {}),
                        )
                    )

            for evaluated in _evaluate_records(
                evaluable_records,
                args.data_dir,
                progress_label=progress_label,
                progress=progress,
            ):
                pass_results.append(evaluated["candidate_hits"])
                sample_rows_by_label[run_label].append(evaluated["sample_row"])

            if not pass_results:
                LOGGER.warning("No usable samples for %s / %s.", model_name, benchmark)
                progress.update(groups_task, advance=1)
                continue

            gold_failures = sum(
                1 for row in sample_rows_by_label[run_label]
                if row.get("model_name") == model_name
                and row.get("benchmark") == benchmark
                and not row.get("gold_success", True)
            )
            if gold_failures:
                LOGGER.warning(
                    "%s / %s / %s: %d sample(s) have failing gold SQL — "
                    "these are counted as model misses but may indicate data issues.",
                    model_name, benchmark, run_label, gold_failures,
                )

            row: dict[str, Any] = {
                "model_name": model_name,
                "model_display_name": next(
                    (
                        row.get("model_display_name")
                        for row in sample_rows_by_label[run_label]
                        if row.get("model_name") == model_name and row.get("benchmark") == benchmark
                    ),
                    model_name,
                ),
                "model_version": next(
                    (
                        row.get("model_version")
                        for row in sample_rows_by_label[run_label]
                        if row.get("model_name") == model_name and row.get("benchmark") == benchmark
                    ),
                    None,
                ),
                "benchmark": benchmark,
                "run_label": run_label,
                "n_samples": len(pass_results),
                "execution_accuracy": sum(hits[0] for hits in pass_results) / len(pass_results),
            }
            for k in experiment_cfg["k_values"]:
                row[f"pass@{k}"] = pass_at_k(pass_results, k)
            eff_metrics = compute_efficiency(generation_results, metrics_cfg)
            eff_metrics["_weights"] = metrics_cfg["efficiency_weights"]
            row.update(eff_metrics)
            rows_by_label[run_label].append(row)
            progress.update(groups_task, advance=1)

    if not rows_by_label:
        LOGGER.warning("No raw JSONL files found in %s", args.raw_dir)
        return

    for run_label, rows in rows_by_label.items():
        label_output_dir = args.output_dir / run_label
        label_output_dir.mkdir(parents=True, exist_ok=True)

        normalized_rows = normalize_efficiency_rows(rows)
        for row in normalized_rows:
            row.pop("_weights", None)

        output_path = label_output_dir / "summary_metrics.csv"
        fieldnames = sorted({key for row in normalized_rows for key in row})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_rows)
        LOGGER.info("Saved %s metrics to %s", run_label, output_path)

        sample_rows = sample_rows_by_label.get(run_label, [])
        if sample_rows:
            sample_output_path = label_output_dir / "sample_metrics.csv"
            sample_fieldnames = sorted({key for row in sample_rows for key in row})
            with sample_output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
                writer.writeheader()
                writer.writerows(sample_rows)
            LOGGER.info("Saved %s sample metrics to %s", run_label, sample_output_path)


if __name__ == "__main__":
    main()
