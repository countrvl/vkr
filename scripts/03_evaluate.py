"""Aggregate metrics from raw generation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import load_yaml_config
from src.evaluation.ea import execution_accuracy
from src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from src.evaluation.pass_at_k import pass_at_k
from src.inference.base import GenerationResult


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw NL2SQL generation results.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
    return parser.parse_args()


def _load_records(raw_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load and group JSONL records by (model_name, benchmark).

    Each record is one sample with a ``generations`` list, as written by
    :class:`src.inference.runner.ExperimentRunner`.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
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
                key = (record["model_name"], record["benchmark"])
                grouped[key].append(record)
    return grouped


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    metrics_cfg = load_yaml_config(args.config_dir / "metrics.yaml")
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    grouped = _load_records(args.raw_dir)
    for (model_name, benchmark), records in grouped.items():
        predictions: list[str] = []
        gold: list[str] = []
        db_paths: list[Path] = []
        pass_results: list[list[bool]] = []
        generation_results: list[GenerationResult] = []

        for record in records:
            gold_sql = record.get("gold_sql", "")
            db_path = Path(record["db_path"])
            generations = record.get("generations", [])

            if not generations:
                LOGGER.warning("Sample %s has no generations, skipping.", record.get("sample_id"))
                continue

            candidate_sql = [gen["sql"] for gen in generations]
            candidate_hits: list[bool] = []
            for sql in candidate_sql:
                ea_score = execution_accuracy([sql], [gold_sql], [db_path])
                candidate_hits.append(ea_score == 1.0)

            pass_results.append(candidate_hits)
            predictions.append(candidate_sql[0])
            gold.append(gold_sql)
            db_paths.append(db_path)

            for gen in generations:
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

        if not predictions:
            LOGGER.warning("No usable samples for %s / %s.", model_name, benchmark)
            continue

        row: dict[str, Any] = {
            "model_name": model_name,
            "benchmark": benchmark,
            "n_samples": len(predictions),
            "execution_accuracy": execution_accuracy(predictions, gold, db_paths),
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


if __name__ == "__main__":
    main()
