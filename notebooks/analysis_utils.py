from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_yaml_config
from src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from src.evaluation.executor import ExecutionResult, execute_sql
from src.evaluation.pass_at_k import compute_all_pass_at_k
from src.inference.base import GenerationResult


def detect_project_root() -> Path:
    """Return the repository root when running from the repo or notebooks directory."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
    return cwd


PROJECT_ROOT = detect_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENT_CFG = load_yaml_config(CONFIG_DIR / "experiment.yaml")
METRICS_CFG = load_yaml_config(CONFIG_DIR / "metrics.yaml")
DATA_DIR = PROJECT_ROOT / EXPERIMENT_CFG.get("data_dir", "data")


def candidate_results_dirs() -> list[Path]:
    """Return likely results directories, ordered by usefulness."""
    candidates = [PROJECT_ROOT / "results"]
    tmp_root = Path(tempfile.gettempdir())
    smoke_dirs = sorted(tmp_root.glob("vkr_smoke*"), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates.extend(smoke_dirs)
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _results_dir_score(results_dir: Path) -> tuple[int, int]:
    metrics_path = results_dir / "metrics" / "summary_metrics.csv"
    metric_score = 1 if metrics_path.exists() else 0
    raw_score = 0
    raw_dir = results_dir / "raw"
    if raw_dir.exists():
        for path in raw_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as handle:
                raw_score += sum(1 for line in handle if line.strip())
    return metric_score, raw_score


def select_results_dir() -> Path:
    """Pick the most useful available results directory."""
    explicit_results_dir = os.getenv("NL2SQL_RESULTS_DIR")
    if explicit_results_dir:
        return Path(explicit_results_dir).expanduser().resolve()

    candidates = [path for path in candidate_results_dirs() if path.exists()]
    if not candidates:
        return PROJECT_ROOT / "results"
    return max(candidates, key=_results_dir_score)


RESULTS_DIR = select_results_dir()
SAMPLE_METRICS_PATH: Path | None = None


def get_results_dir() -> Path:
    """Return the active results directory, honoring env overrides at call time."""
    return select_results_dir()


def get_figures_dir(run_label: str | None = None) -> Path:
    """Return the figures directory for a run label."""
    figures_dir = get_results_dir() / "figures"
    if run_label:
        figures_dir = figures_dir / run_label
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_metrics_dir(run_label: str | None = None) -> Path:
    """Return the metrics directory for a run label."""
    metrics_dir = get_results_dir() / "metrics"
    if run_label:
        metrics_dir = metrics_dir / run_label
    return metrics_dir


def get_sample_metrics_path(run_label: str | None = None) -> Path:
    """Return the sample metrics path for the active results directory."""
    if SAMPLE_METRICS_PATH is not None:
        return SAMPLE_METRICS_PATH
    return get_metrics_dir(run_label) / "sample_metrics.csv"


def infer_run_label(record: dict[str, Any], source_path: str) -> str:
    run_label = record.get("run_label")
    if isinstance(run_label, str) and run_label.strip():
        return run_label.strip()
    if "_pass_k_" in source_path:
        return "pass_k"
    if "_ea_" in source_path:
        return "ea"
    return "legacy"


def resolve_db_path(raw_db_path: str) -> Path:
    db_path = Path(raw_db_path)
    if db_path.is_absolute():
        return db_path
    candidates = [db_path]
    if str(db_path).startswith(f"{DATA_DIR.name}/"):
        candidates.append(DATA_DIR.parent / db_path)
    else:
        candidates.append(DATA_DIR / db_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _iter_raw_records(results_dir: Path, run_label: str | None = None) -> list[dict[str, Any]]:
    raw_dir = results_dir / "raw"
    records: list[dict[str, Any]] = []
    if not raw_dir.exists():
        return records
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_path"] = str(path)
                record["run_label"] = infer_run_label(record, str(path))
                if run_label is not None and record["run_label"] != run_label:
                    continue
                records.append(record)
    return records


@lru_cache(maxsize=8)
def load_records(run_label: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample-level and generation-level DataFrames."""
    sample_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for record in _iter_raw_records(get_results_dir(), run_label):
        db_path = resolve_db_path(record["db_path"])
        sample_row = {
            "sample_id": record.get("sample_id"),
            "model_name": record.get("model_name"),
            "benchmark": record.get("benchmark"),
            "run_label": record.get("run_label"),
            "question": record.get("question", ""),
            "gold_sql": record.get("gold_sql", ""),
            "db_id": record.get("db_id"),
            "db_path": str(db_path),
            "difficulty": record.get("difficulty"),
            "evidence": record.get("evidence"),
            "question_len": len(record.get("question", "")),
            "gold_sql_len": len(record.get("gold_sql", "")),
            "n_generations": len(record.get("generations", [])),
            "source_path": record["_source_path"],
        }
        sample_rows.append(sample_row)

        for idx, generation in enumerate(record.get("generations", []), start=1):
            generation_rows.append(
                {
                    "sample_id": record.get("sample_id"),
                    "model_name": record.get("model_name"),
                    "benchmark": record.get("benchmark"),
                    "run_label": record.get("run_label"),
                    "generation_index": idx,
                    "sql": generation.get("sql", ""),
                    "raw_response": generation.get("raw_response", ""),
                    "tokens_input": generation.get("tokens_input", 0),
                    "tokens_output": generation.get("tokens_output", 0),
                    "latency_ms": generation.get("latency_ms", 0.0),
                    "metadata": generation.get("metadata", {}),
                    "timestamp": generation.get("timestamp"),
                    "db_path": str(db_path),
                    "gold_sql": record.get("gold_sql", ""),
                    "question": record.get("question", ""),
                }
            )

    samples_df = pd.DataFrame(sample_rows)
    generations_df = pd.DataFrame(generation_rows)
    return samples_df, generations_df


def reset_analysis_caches() -> None:
    """Clear cached notebook data after new inference/evaluation runs."""
    load_records.cache_clear()
    load_summary_metrics.cache_clear()
    load_sample_metrics.cache_clear()
    compute_summary_metrics.cache_clear()
    compute_sample_outcomes.cache_clear()


@lru_cache(maxsize=8)
def load_summary_metrics(run_label: str | None = None) -> pd.DataFrame:
    metrics_path = get_metrics_dir(run_label) / "summary_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return compute_summary_metrics(run_label)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() == "true"


def _parse_candidate_hits(value: Any) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    if pd.isna(value):
        return []
    parsed = json.loads(value)
    return [bool(item) for item in parsed]


@lru_cache(maxsize=8)
def load_sample_metrics(run_label: str | None = None) -> pd.DataFrame:
    sample_metrics_path = get_sample_metrics_path(run_label)
    if not sample_metrics_path.exists():
        return pd.DataFrame()

    outcomes_df = pd.read_csv(sample_metrics_path)
    if outcomes_df.empty:
        return outcomes_df

    if "candidate_hits" in outcomes_df.columns:
        outcomes_df["candidate_hits"] = outcomes_df["candidate_hits"].map(_parse_candidate_hits)

    for column in ["gold_success", "first_hit", "any_hit", "first_pred_success", "empty_sql"]:
        if column in outcomes_df.columns:
            outcomes_df[column] = outcomes_df[column].map(_parse_bool)

    return outcomes_df


@lru_cache(maxsize=8)
def compute_summary_metrics(run_label: str | None = None) -> pd.DataFrame:
    samples_df, generations_df = load_records(run_label)
    if samples_df.empty or generations_df.empty:
        return pd.DataFrame()

    outcomes_df = compute_sample_outcomes(run_label)
    rows: list[dict[str, Any]] = []

    for (model_name, benchmark, run_label), group in outcomes_df.groupby(
        ["model_name", "benchmark", "run_label"], dropna=False
    ):
        gen_group = generations_df[
            (generations_df["model_name"] == model_name)
            & (generations_df["benchmark"] == benchmark)
            & (generations_df["run_label"] == run_label)
        ]

        generation_results = [
            GenerationResult(
                sql=row.sql,
                raw_response=row.raw_response,
                tokens_input=int(row.tokens_input),
                tokens_output=int(row.tokens_output),
                latency_ms=float(row.latency_ms),
                model_name=row.model_name,
                metadata=row.metadata if isinstance(row.metadata, dict) else {},
            )
            for row in gen_group.itertuples(index=False)
        ]
        eff_metrics = compute_efficiency(generation_results, METRICS_CFG)
        row = {
            "model_name": model_name,
            "benchmark": benchmark,
            "run_label": run_label,
            "n_samples": int(len(group)),
            "execution_accuracy": float(group["first_hit"].mean()),
        }
        pass_results = group["candidate_hits"].tolist()
        row.update({f"pass@{k}": v for k, v in compute_all_pass_at_k(pass_results, EXPERIMENT_CFG["k_values"]).items()})
        row.update(eff_metrics)
        row["_weights"] = METRICS_CFG["efficiency_weights"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    normalized = normalize_efficiency_rows(rows)
    for row in normalized:
        row.pop("_weights", None)
    return pd.DataFrame(normalized)


@lru_cache(maxsize=8)
def compute_sample_outcomes(run_label: str | None = None) -> pd.DataFrame:
    """Evaluate every sample and generation against the gold SQL."""
    persisted_outcomes = load_sample_metrics(run_label)
    if not persisted_outcomes.empty:
        return persisted_outcomes.copy()

    samples_df, generations_df = load_records(run_label)
    if samples_df.empty or generations_df.empty:
        return pd.DataFrame()

    generation_lookup: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in generations_df.to_dict(orient="records"):
        key = (row["model_name"], row["benchmark"], row["run_label"], row["sample_id"])
        generation_lookup[key].append(row)

    outcome_rows: list[dict[str, Any]] = []
    for sample in samples_df.to_dict(orient="records"):
        key = (sample["model_name"], sample["benchmark"], sample["run_label"], sample["sample_id"])
        generations = sorted(generation_lookup.get(key, []), key=lambda row: row["generation_index"])
        if not generations:
            continue

        db_path = Path(sample["db_path"])
        gold_sql = sample["gold_sql"]
        gold_result = execute_sql(gold_sql, db_path)
        candidate_hits: list[bool] = []
        first_pred_error = None
        first_pred_sql = generations[0]["sql"]
        first_pred_success = False

        for idx, generation in enumerate(generations):
            pred_result = execute_sql(generation["sql"], db_path)
            hit = pred_result.success and gold_result.success and pred_result.rows == gold_result.rows
            candidate_hits.append(hit)
            if idx == 0:
                first_pred_error = pred_result.error
                first_pred_success = pred_result.success

        outcome_rows.append(
            {
                **sample,
                "gold_success": gold_result.success,
                "gold_error": gold_result.error,
                "candidate_hits": candidate_hits,
                "first_hit": bool(candidate_hits[0]),
                "any_hit": any(candidate_hits),
                "n_candidates": len(candidate_hits),
                "first_pred_sql": first_pred_sql,
                "first_pred_success": first_pred_success,
                "first_pred_error": first_pred_error,
                "empty_sql": not str(first_pred_sql).strip(),
            }
        )

    return pd.DataFrame(outcome_rows)


def ensure_expert_template(run_label: str | None = None) -> Path:
    """Create an expert score template if a populated file is absent."""
    metrics_dir = get_metrics_dir(run_label)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    expert_scores_path = metrics_dir / "expert_scores.csv"
    template_path = metrics_dir / "expert_scores_template.csv"

    if expert_scores_path.exists():
        return expert_scores_path

    outcomes_df = compute_sample_outcomes(run_label)
    if outcomes_df.empty:
        return template_path

    template_df = outcomes_df[["sample_id", "model_name", "benchmark"]].copy()
    template_df["completeness"] = pd.NA
    template_df["efficiency"] = pd.NA
    template_df["readability"] = pd.NA
    template_df.to_csv(template_path, index=False)
    return template_path
