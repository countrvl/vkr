"""Evaluate code-generation benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "code"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from code.src.evaluation.functional_correctness import evaluate_code_candidate
from code.src.evaluation.pass_at_k import compute_all_pass_at_k
from code.src.inference.base import GenerationResult
from shared.config import load_yaml_config
from shared.logging_utils import configure_logging, create_progress


LOGGER = logging.getLogger(__name__)


def _config_defaults(config_dir: Path) -> dict[str, Path]:
    benchmarks_cfg = load_yaml_config(config_dir / "benchmarks.yaml")
    return {
        "raw_dir": PROJECT_ROOT / benchmarks_cfg.get("results_dir", "results/code/raw"),
    }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-dir", type=Path, default=DOMAIN_ROOT / "configs")
    config_args, _ = config_parser.parse_known_args()
    defaults = _config_defaults(config_args.config_dir)

    parser = argparse.ArgumentParser(description="Evaluate raw code-generation results.")
    parser.add_argument("--config-dir", type=Path, default=config_args.config_dir)
    parser.add_argument("--raw-dir", type=Path, default=defaults["raw_dir"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "code" / "metrics")
    parser.add_argument(
        "--run-label",
        choices=["fc", "ea", "pass_k", "all"],
        default="all",
        help="Which run_label to evaluate. Results are written into output-dir/<run_label>/.",
    )
    return parser.parse_args()


def _normalize_run_label(run_label: str | None) -> str:
    if run_label in (None, "", "ea"):
        return "fc"
    return run_label


def _expected_generation_count(run_label: str, experiment_cfg: dict[str, Any]) -> int | None:
    if run_label == "fc":
        return 1
    if run_label == "pass_k":
        return max(int(k) for k in experiment_cfg["k_values"])
    return None


def _load_records(raw_dir: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
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
                record["run_label"] = _normalize_run_label(record.get("run_label"))
                key = (record["model_name"], record["benchmark"], record["run_label"])
                grouped[key].append(record)
    return grouped


def _validate_records(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    experiment_cfg: dict[str, Any],
) -> None:
    errors: list[str] = []
    for (_, _, run_label), records in grouped.items():
        expected_generations = _expected_generation_count(run_label, experiment_cfg)
        for record in records:
            sample_id = record.get("sample_id", "<unknown>")
            source_path = record.get("_source_path", "<unknown>")
            generations = record.get("generations", [])
            if not generations:
                errors.append(f"{source_path}: sample {sample_id} has no generations")
                continue
            if expected_generations is not None and len(generations) != expected_generations:
                errors.append(
                    f"{source_path}: sample {sample_id} has {len(generations)} generations, "
                    f"expected {expected_generations} for run_label={run_label}"
                )
    if errors:
        preview = "\n".join(errors[:10])
        remaining = len(errors) - 10
        if remaining > 0:
            preview += f"\n... and {remaining} more"
        raise ValueError(f"Raw evaluation input validation failed:\n{preview}")


def _evaluate_record(record: dict[str, Any], execution_cfg: dict[str, Any]) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    candidate_hits: list[bool] = []
    generations = record.get("generations", [])
    for idx, generation in enumerate(generations, start=1):
        candidate = evaluate_code_candidate(
            benchmark=record["benchmark"],
            task_id=record["sample_id"],
            candidate_index=idx - 1,
            code=generation.get("code", ""),
            execution_cfg=execution_cfg,
        )
        candidate_hit = bool(candidate["functional_correctness"])
        candidate_hits.append(candidate_hit)
        candidate_rows.append(
            {
                "sample_id": record["sample_id"],
                "model_name": record["model_name"],
                "benchmark": record["benchmark"],
                "run_label": record["run_label"],
                "candidate_index": idx,
                "entry_point": record.get("entry_point"),
                "code": candidate["normalized_code"],
                "compiled_ok": candidate["compiled_ok"],
                "tests_passed": candidate["tests_passed"],
                "functional_correctness": candidate["functional_correctness"],
                "error_type": candidate["error_type"],
                "base_status": candidate["base_status"],
                "plus_status": candidate["plus_status"],
                "base_passed": candidate["base_passed"],
                "base_total": candidate["base_total"],
                "plus_passed": candidate["plus_passed"],
                "plus_total": candidate["plus_total"],
            }
        )

    first_candidate = candidate_rows[0]
    sample_row = {
        "sample_id": record["sample_id"],
        "model_name": record["model_name"],
        "benchmark": record["benchmark"],
        "run_label": record["run_label"],
        "entry_point": record.get("entry_point"),
        "prompt_len": len(record.get("prompt", "")),
        "n_generations": len(generations),
        "source_path": record.get("_source_path", ""),
        "candidate_hits": json.dumps(candidate_hits),
        "first_hit": bool(candidate_hits[0]),
        "any_hit": any(candidate_hits),
        "n_candidates": len(candidate_hits),
        "first_code": first_candidate["code"],
        "first_compiled_ok": first_candidate["compiled_ok"],
        "first_tests_passed": first_candidate["tests_passed"],
        "first_error_type": first_candidate["error_type"],
    }

    return {
        "candidate_hits": candidate_hits,
        "sample_row": sample_row,
        "candidate_rows": candidate_rows,
    }


def main() -> None:
    configure_logging(logging.INFO)
    args = parse_args()
    metrics_cfg = load_yaml_config(args.config_dir / "metrics.yaml")
    experiment_cfg = load_yaml_config(args.config_dir / "experiment.yaml")
    execution_cfg = metrics_cfg.get("execution", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped = _load_records(args.raw_dir)
    _validate_records(grouped, experiment_cfg=experiment_cfg)

    selected_items = [
        item
        for item in grouped.items()
        if args.run_label == "all" or item[0][2] == _normalize_run_label(args.run_label)
    ]
    with create_progress() as progress:
        groups_task = progress.add_task("Evaluation groups", total=len(selected_items), status="")
        for (model_name, benchmark, run_label), records in selected_items:
            pass_results: list[list[bool]] = []
            generation_results: list[GenerationResult] = []
            progress_label = f"{model_name} / {benchmark} / {run_label}"
            progress.update(groups_task, status=progress_label)
            eval_task = progress.add_task(progress_label, total=len(records), status="")
            for record in records:
                evaluated = _evaluate_record(record, execution_cfg)
                pass_results.append(evaluated["candidate_hits"])
                sample_rows_by_label[run_label].append(evaluated["sample_row"])
                candidate_rows_by_label[run_label].extend(evaluated["candidate_rows"])
                for generation in record.get("generations", []):
                    generation_results.append(
                        GenerationResult(
                            code=generation.get("code", ""),
                            raw_response=generation.get("raw_response", ""),
                            tokens_input=generation.get("tokens_input", 0),
                            tokens_output=generation.get("tokens_output", 0),
                            latency_ms=generation.get("latency_ms", 0.0),
                            model_name=generation.get("model_name", model_name),
                            metadata=generation.get("metadata", {}),
                        )
                    )
                progress.update(eval_task, advance=1)
            progress.remove_task(eval_task)

            if not pass_results:
                LOGGER.warning("No usable samples for %s / %s.", model_name, benchmark)
                progress.update(groups_task, advance=1)
                continue

            row: dict[str, Any] = {
                "model_name": model_name,
                "benchmark": benchmark,
                "run_label": run_label,
                "n_samples": len(pass_results),
                "functional_correctness": sum(hits[0] for hits in pass_results) / len(pass_results),
            }
            for k, value in compute_all_pass_at_k(pass_results, experiment_cfg["k_values"]).items():
                row[f"pass@{k}"] = value
            eff_metrics = compute_efficiency(generation_results, metrics_cfg)
            eff_metrics["_weights"] = metrics_cfg["efficiency_weights"]
            row.update(eff_metrics)
            rows_by_label[run_label].append(row)
            progress.update(groups_task, advance=1)

    for run_label, rows in rows_by_label.items():
        run_output_dir = args.output_dir / run_label
        run_output_dir.mkdir(parents=True, exist_ok=True)
        normalized_rows = normalize_efficiency_rows(rows)
        summary_fieldnames = list(normalized_rows[0].keys())
        with (run_output_dir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(normalized_rows)

        sample_rows = sample_rows_by_label[run_label]
        if sample_rows:
            sample_fieldnames = list(sample_rows[0].keys())
            with (run_output_dir / "sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
                writer.writeheader()
                writer.writerows(sample_rows)

        candidate_rows = candidate_rows_by_label[run_label]
        if candidate_rows:
            candidate_fieldnames = list(candidate_rows[0].keys())
            with (run_output_dir / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=candidate_fieldnames)
                writer.writeheader()
                writer.writerows(candidate_rows)

        LOGGER.info("Wrote code metrics for %s to %s", run_label, run_output_dir)


if __name__ == "__main__":
    main()
