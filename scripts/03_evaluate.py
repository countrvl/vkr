"""Aggregate metrics from raw generation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import load_yaml_config
from src.evaluation.ea import evaluate_candidate_predictions
from src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from src.evaluation.pass_at_k import pass_at_k
from src.inference.base import GenerationResult


LOGGER = logging.getLogger(__name__)
_LEGACY_RUN_LABEL = "legacy"


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    """Load CLI defaults sourced from experiment.yaml."""
    experiment_cfg = load_yaml_config(config_dir / "experiment.yaml")
    return {
        "raw_dir": Path(experiment_cfg.get("results_dir", "results/raw")),
        "data_dir": Path(experiment_cfg.get("data_dir", "data")),
    }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)

    parser = argparse.ArgumentParser(description="Evaluate raw NL2SQL generation results.")
    parser.add_argument("--config-dir", type=Path, default=config_args.config_dir)
    parser.add_argument("--raw-dir", type=Path, default=defaults["raw_dir"])
    parser.add_argument("--data-dir", type=Path, default=defaults["data_dir"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
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
    candidate_sql = [gen["sql"] for gen in generations]
    evaluation = evaluate_candidate_predictions(candidate_sql, gold_sql, db_path)
    candidate_hits = evaluation["candidate_hits"]

    return {
        "candidate_hits": candidate_hits,
        "sample_row": {
            "sample_id": record.get("sample_id"),
            "model_name": record.get("model_name"),
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
) -> list[dict[str, Any]]:
    """Evaluate raw records, using processes for larger groups."""
    if len(records) < 2:
        results: list[dict[str, Any]] = []
        with tqdm(total=len(records), desc=progress_label, unit="sample") as progress:
            for record in records:
                results.append(_evaluate_record(record, data_dir))
                progress.update(1)
        return results

    max_workers = min(os.cpu_count() or 1, len(records))
    if max_workers <= 1:
        results = []
        with tqdm(total=len(records), desc=progress_label, unit="sample") as progress:
            for record in records:
                results.append(_evaluate_record(record, data_dir))
                progress.update(1)
        return results

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_evaluate_record, record, data_dir) for record in records]
        with tqdm(total=len(records), desc=progress_label, unit="sample") as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    metrics_cfg = load_yaml_config(args.config_dir / "metrics.yaml")
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_rows: list[dict[str, Any]] = []
    grouped = _load_records(args.raw_dir)
    _validate_records(grouped, experiment_cfg=experiment_cfg, data_dir=args.data_dir)
    for (model_name, benchmark, run_label), records in grouped.items():
        pass_results: list[list[bool]] = []
        generation_results: list[GenerationResult] = []
        evaluable_records = [record for record in records if record.get("generations")]
        progress_label = f"{model_name} / {benchmark} / {run_label}"

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
        ):
            pass_results.append(evaluated["candidate_hits"])
            sample_rows.append(evaluated["sample_row"])

        if not pass_results:
            LOGGER.warning("No usable samples for %s / %s.", model_name, benchmark)
            continue

        row: dict[str, Any] = {
            "model_name": model_name,
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
        rows.append(row)

    if not rows:
        LOGGER.warning("No raw JSONL files found in %s", args.raw_dir)
        return

    rows = normalize_efficiency_rows(rows)
    # Remove internal helper key before writing CSV.
    for row in rows:
        row.pop("_weights", None)

    output_path = args.output_dir / "summary_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved metrics to %s", output_path)

    sample_output_path = args.output_dir / "sample_metrics.csv"
    if sample_rows:
        sample_fieldnames = sorted({key for row in sample_rows for key in row})
        with sample_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
            writer.writeheader()
            writer.writerows(sample_rows)
        LOGGER.info("Saved sample metrics to %s", sample_output_path)


if __name__ == "__main__":
    main()
