"""Aggregate metrics from raw generation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import load_yaml_config
from src.evaluation.ea import execution_accuracy
from src.evaluation.efficiency import compute_efficiency
from src.evaluation.pass_at_k import pass_at_k
from src.inference.base import GenerationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw NL2SQL generation results.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
    return parser.parse_args()


def _load_records(raw_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (record["model_name"], record["benchmark"])
                grouped[key].append(record)
    return grouped


def main() -> None:
    args = parse_args()
    metrics_cfg = load_yaml_config(args.config_dir / "metrics.yaml")
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    grouped = _load_records(args.raw_dir)
    for (model_name, benchmark), records in grouped.items():
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_sample[record["sample"]["id"]].append(record)

        predictions = []
        gold = []
        db_paths = []
        pass_results = []
        generation_results = []
        for sample_records in by_sample.values():
            sample_records.sort(key=lambda item: item["generation_index"])
            sample = sample_records[0]["sample"]
            candidate_sql = [item["generation"]["sql"] for item in sample_records]
            candidate_hits = []
            for sql in candidate_sql:
                score = execution_accuracy([sql], [sample["gold_sql"]], [Path(sample["db_path"])])
                candidate_hits.append(score == 1.0)
            pass_results.append(candidate_hits)
            predictions.append(candidate_sql[0])
            gold.append(sample["gold_sql"])
            db_paths.append(Path(sample["db_path"]))
            for item in sample_records:
                generation_results.append(
                    GenerationResult(
                        sql=item["generation"]["sql"],
                        raw_response=item["generation"]["raw_response"],
                        tokens_input=item["generation"]["tokens_input"],
                        tokens_output=item["generation"]["tokens_output"],
                        latency_ms=item["generation"]["latency_ms"],
                        model_name=item["generation"]["model_name"],
                        metadata=item["generation"].get("metadata", {}),
                    )
                )

        row = {
            "model_name": model_name,
            "benchmark": benchmark,
            "execution_accuracy": execution_accuracy(predictions, gold, db_paths),
        }
        for k in experiment_cfg["k_values"]:
            row[f"pass@{k}"] = pass_at_k(pass_results, k)
        row.update(compute_efficiency(generation_results, metrics_cfg))
        rows.append(row)

    if not rows:
        print(f"No raw JSONL files found in {args.raw_dir}")
        return

    output_path = args.output_dir / "summary_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
