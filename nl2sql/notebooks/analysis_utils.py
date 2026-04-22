from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from shared.config import load_domain_models, load_yaml_config
from shared.evaluation.statistics import (
    bootstrap_interval,
    bootstrap_quantile_fields,
    quantile_fields,
    wilson_interval,
)
from nl2sql.src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from nl2sql.src.evaluation.executor import execute_sql
from nl2sql.src.evaluation.pass_at_k import compute_all_pass_at_k
from nl2sql.src.inference.base import GenerationResult


def detect_project_root() -> Path:
    """Return the repository root when running from the repo or notebook directories."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if (candidate / "nl2sql").exists() and (candidate / "shared").exists():
            return candidate
    return cwd


PROJECT_ROOT = detect_project_root()
DOMAIN_ROOT = PROJECT_ROOT / "nl2sql"
CONFIG_DIR = DOMAIN_ROOT / "configs"
EXPERIMENT_CFG = load_yaml_config(CONFIG_DIR / "experiment.yaml")
METRICS_CFG = load_yaml_config(CONFIG_DIR / "metrics.yaml")
DATA_DIR = PROJECT_ROOT / EXPERIMENT_CFG.get("data_dir", "data/nl2sql")
MODEL_DISPLAY_LOOKUP = {
    cfg.get("name"): {
        "display_name": cfg.get("display_name") or cfg.get("name"),
        "version": cfg.get("version"),
        "key": key,
        "family": cfg.get("family"),
        "active_by_default": bool(cfg.get("active_by_default", True)),
    }
    for key, cfg in load_domain_models("supports_sql").items()
}
PRIMARY_REPORT_MODEL_KEYS = (
    "m1_deepseek",
    "m1_chatgpt",
    "m2_defog",
    "m2_hrida",
    "m2_arctic",
)
EXPECTED_BENCHMARK_SAMPLE_COUNTS = {
    "spider": 1034,
    "bird": 1534,
}


def _statistics_config() -> dict[str, Any]:
    return {
        "confidence_level": float(METRICS_CFG.get("statistics", {}).get("confidence_level", 0.95)),
        "bootstrap_resamples": int(METRICS_CFG.get("statistics", {}).get("bootstrap_resamples", 1000)),
        "bootstrap_seed": int(METRICS_CFG.get("statistics", {}).get("bootstrap_seed", 42)),
        "quantiles": tuple(float(q) for q in METRICS_CFG.get("statistics", {}).get("quantiles", (0.05, 0.5, 0.95))),
    }


STATS_CFG = _statistics_config()


def first_non_null(values: Any, default: Any = None) -> Any:
    """Return the first non-null item from a Series-like value or a scalar."""
    if isinstance(values, pd.Series):
        non_null = values.dropna().tolist()
        return non_null[0] if non_null else default
    return default if pd.isna(values) else values


def model_display_name(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_display_name") or MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("display_name") or model_name


def model_version(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_version") or MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("version")


def model_key(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_key") or MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("key")


def model_family(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_family") or MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("family")


def active_by_default(record: dict[str, Any]) -> bool:
    model_name = record.get("model_name")
    value = record.get("active_by_default")
    if value is not None and not pd.isna(value):
        return bool(value)
    return bool(MODEL_DISPLAY_LOOKUP.get(model_name, {}).get("active_by_default", True))


def candidate_results_dirs() -> list[Path]:
    """Return likely results directories, ordered by usefulness."""
    candidates = [PROJECT_ROOT / "results" / "nl2sql"]
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
    metrics_dir = results_dir / "metrics"
    metric_score = 1 if any(
        (metrics_dir / run_label / "summary_metrics.csv").exists() for run_label in ("ea", "pass_k")
    ) else 0
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
        return PROJECT_ROOT / "results" / "nl2sql"
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


def get_mini_bench_dir() -> Path:
    """Return the mini-benchmark results directory."""
    return get_results_dir() / "mini_bench"


def get_sample_metrics_path(run_label: str | None = None) -> Path:
    """Return the sample metrics path for the active results directory."""
    if SAMPLE_METRICS_PATH is not None:
        return SAMPLE_METRICS_PATH
    return get_metrics_dir(run_label) / "sample_metrics.csv"


def _raise_missing_run_artifacts(run_label: str, artifact_name: str) -> None:
    """Raise a clear error when notebook artifacts for a run label are missing."""
    metrics_dir = get_metrics_dir(run_label)
    raise FileNotFoundError(
        f"No {artifact_name} for run_label={run_label!r} in {metrics_dir}. "
        f"Run `uv run python nl2sql/scripts/03_evaluate.py --run-label {run_label}` first."
    )


def infer_run_label(record: dict[str, Any], source_path: str) -> str:
    run_label = record.get("run_label")
    if isinstance(run_label, str) and run_label.strip():
        return run_label.strip()
    if "_pass_k_" in source_path:
        return "pass_k"
    if "_ea_" in source_path:
        return "ea"
    return "legacy"


def _summary_metrics_require_refresh(summary_df: pd.DataFrame) -> bool:
    if summary_df.empty:
        return False
    required_columns = {
        "execution_accuracy_ci_low",
        "execution_accuracy_ci_high",
        "execution_accuracy_q05",
        "execution_accuracy_q50",
        "execution_accuracy_q95",
    }
    for k in EXPERIMENT_CFG.get("k_values", []):
        required_columns.update(
            {
                f"pass@{k}_ci_low",
                f"pass@{k}_ci_high",
                f"pass@{k}_q05",
                f"pass@{k}_q50",
                f"pass@{k}_q95",
            }
        )
    return not required_columns.issubset(summary_df.columns)


def _with_model_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    enriched = df.copy()
    if "model_key" not in enriched.columns and "model_name" in enriched.columns:
        enriched["model_key"] = enriched["model_name"].map(
            lambda name: MODEL_DISPLAY_LOOKUP.get(name, {}).get("key")
        )
    if "model_family" not in enriched.columns and "model_name" in enriched.columns:
        enriched["model_family"] = enriched["model_name"].map(
            lambda name: MODEL_DISPLAY_LOOKUP.get(name, {}).get("family")
        )
    if "active_by_default" not in enriched.columns and "model_name" in enriched.columns:
        enriched["active_by_default"] = enriched["model_name"].map(
            lambda name: bool(MODEL_DISPLAY_LOOKUP.get(name, {}).get("active_by_default", True))
        )
    enriched["is_primary_model"] = enriched["model_key"].isin(PRIMARY_REPORT_MODEL_KEYS)
    return enriched


def _mean_bool_hits(pass_results: list[list[bool]]) -> float:
    if not pass_results:
        return 0.0
    return float(sum(bool(hits[0]) for hits in pass_results) / len(pass_results))


def _mean_numeric(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_generation_values(results: list[GenerationResult], field: str) -> list[float]:
    values: list[float] = []
    for result in results:
        if field == "Tinf":
            values.append(float(result.latency_ms))
        elif field == "Tok":
            values.append(float(result.tokens_input + result.tokens_output))
        elif field == "Cost":
            backend = result.metadata.get("backend")
            if backend == "ollama":
                values.append(0.0)
            else:
                cost_usd = result.metadata.get("cost_usd")
                if cost_usd is not None:
                    values.append(float(cost_usd))
                else:
                    pricing = result.metadata.get("pricing")
                    if pricing:
                        input_cost = (result.tokens_input / 1_000_000.0) * float(
                            pricing.get("input_per_mtok", 0.0)
                        )
                        output_cost = (result.tokens_output / 1_000_000.0) * float(
                            pricing.get("output_per_mtok", 0.0)
                        )
                        values.append(input_cost + output_cost)
        elif field == "Mem":
            memory = result.metadata.get("memory_mb")
            if memory is not None:
                values.append(float(memory))
    return values


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


def load_mini_bench_artifacts(
    *,
    results_dir: Path | None = None,
    model_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load mini-benchmark summary, category, difficulty and failure tables."""
    base_dir = (results_dir or get_results_dir()) / "mini_bench"
    if not base_dir.exists():
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    selected_keys = set(model_keys or [])
    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for model_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        model_key = model_dir.name
        if selected_keys and model_key not in selected_keys:
            continue
        summary_path = model_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = dict(payload.get("summary", {}))
        if not summary:
            continue
        base_row = {
            "model_key": payload.get("model_key", model_key),
            "model_display_name": payload.get("model_display_name"),
            "model_family": payload.get("model_family"),
            "prompt_profile": payload.get("prompt_profile"),
        }
        summary_rows.append({**base_row, **summary})
        for row in payload.get("by_category", []):
            category_rows.append({**base_row, **row})
        for row in payload.get("by_difficulty", []):
            difficulty_rows.append({**base_row, **row})
        for row in payload.get("failure_examples", []):
            failure_rows.append({**base_row, **row})

    return (
        _with_model_metadata(pd.DataFrame(summary_rows)),
        _with_model_metadata(pd.DataFrame(category_rows)),
        _with_model_metadata(pd.DataFrame(difficulty_rows)),
        _with_model_metadata(pd.DataFrame(failure_rows)),
    )


@lru_cache(maxsize=8)
def load_records(run_label: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample-level and generation-level DataFrames."""
    sample_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for record in _iter_raw_records(get_results_dir(), run_label):
        db_path = resolve_db_path(record["db_path"])
        sample_row = {
            "sample_id": record.get("sample_id"),
            "model_key": model_key(record),
            "model_name": record.get("model_name"),
            "model_display_name": model_display_name(record),
            "model_version": model_version(record),
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
                    "model_display_name": model_display_name(record),
                    "model_version": model_version(record),
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
        summary_df = _with_model_metadata(pd.read_csv(metrics_path))
        if run_label is not None and _summary_metrics_require_refresh(summary_df) and _iter_raw_records(get_results_dir(), run_label):
            computed = compute_summary_metrics(run_label)
            if not computed.empty:
                return computed
        return summary_df
    if run_label is not None and not _iter_raw_records(get_results_dir(), run_label):
        _raise_missing_run_artifacts(run_label, "summary metrics or raw records")
    return _with_model_metadata(compute_summary_metrics(run_label))


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
        if run_label is not None and not _iter_raw_records(get_results_dir(), run_label):
            _raise_missing_run_artifacts(run_label, "sample metrics or raw records")
        return pd.DataFrame()

    outcomes_df = pd.read_csv(sample_metrics_path)
    if outcomes_df.empty:
        return outcomes_df

    if "candidate_hits" in outcomes_df.columns:
        outcomes_df["candidate_hits"] = outcomes_df["candidate_hits"].map(_parse_candidate_hits)

    for column in ["gold_success", "first_hit", "any_hit", "first_pred_success", "empty_sql"]:
        if column in outcomes_df.columns:
            outcomes_df[column] = outcomes_df[column].map(_parse_bool)

    return _with_model_metadata(outcomes_df)


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
            "model_display_name": (
                first_non_null(group["model_display_name"], model_name)
                if "model_display_name" in group.columns
                else model_name
            ),
            "model_version": (
                first_non_null(group["model_version"])
                if "model_version" in group.columns
                else None
            ),
            "benchmark": benchmark,
            "run_label": run_label,
            "n_samples": int(len(group)),
            "execution_accuracy": float(group["first_hit"].mean()),
        }
        pass_results = group["candidate_hits"].tolist()
        successes = int(group["first_hit"].sum())
        ea_ci_low, ea_ci_high = wilson_interval(
            successes,
            int(len(group)),
            confidence_level=STATS_CFG["confidence_level"],
        )
        row["execution_accuracy_ci_low"] = ea_ci_low
        row["execution_accuracy_ci_high"] = ea_ci_high
        row.update(
            bootstrap_quantile_fields(
                pass_results,
                _mean_bool_hits,
                prefix="execution_accuracy",
                quantiles=STATS_CFG["quantiles"],
                n_resamples=STATS_CFG["bootstrap_resamples"],
                seed=STATS_CFG["bootstrap_seed"],
            )
        )
        row.update({f"pass@{k}": v for k, v in compute_all_pass_at_k(pass_results, EXPERIMENT_CFG["k_values"]).items()})
        for k in EXPERIMENT_CFG["k_values"]:
            ci_low, ci_high = bootstrap_interval(
                pass_results,
                lambda sample, k=k: compute_all_pass_at_k(list(sample), [k])[f"pass@{k}"],
                confidence_level=STATS_CFG["confidence_level"],
                n_resamples=STATS_CFG["bootstrap_resamples"],
                seed=STATS_CFG["bootstrap_seed"] + k,
            )
            row[f"pass@{k}_ci_low"] = ci_low
            row[f"pass@{k}_ci_high"] = ci_high
            row.update(
                bootstrap_quantile_fields(
                    pass_results,
                    lambda sample, k=k: compute_all_pass_at_k(list(sample), [k])[f"pass@{k}"],
                    prefix=f"pass@{k}",
                    quantiles=STATS_CFG["quantiles"],
                    n_resamples=STATS_CFG["bootstrap_resamples"],
                    seed=STATS_CFG["bootstrap_seed"] + k,
                )
            )
        row.update(eff_metrics)
        for component in ("Tinf", "Tok", "Cost", "Mem"):
            row.update(
                quantile_fields(
                    _extract_generation_values(generation_results, component),
                    prefix=component,
                    quantiles=STATS_CFG["quantiles"],
                )
            )
        row["_weights"] = METRICS_CFG["efficiency_weights"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    normalized = normalize_efficiency_rows(rows)
    for row in normalized:
        row.pop("_weights", None)
    return _with_model_metadata(pd.DataFrame(normalized))


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

    return _with_model_metadata(pd.DataFrame(outcome_rows))


def get_expected_sample_counts(run_label: str | None = "ea") -> dict[str, int]:
    if run_label in (None, "ea", "pass_k"):
        return EXPECTED_BENCHMARK_SAMPLE_COUNTS.copy()
    return {}


def build_completeness_audit(
    run_label: str = "ea",
    *,
    summary_df: pd.DataFrame | None = None,
    expected_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    summary = _with_model_metadata(load_summary_metrics(run_label) if summary_df is None else summary_df)
    if summary.empty:
        return pd.DataFrame()

    expected = expected_counts or get_expected_sample_counts(run_label)
    model_rows = (
        summary[["model_name", "model_display_name", "model_key", "model_family", "active_by_default", "is_primary_model"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    rows: list[dict[str, Any]] = []
    for record in model_rows:
        row: dict[str, Any] = {
            "run_label": run_label,
            **record,
        }
        is_complete = True
        for benchmark, expected_count in expected.items():
            subset = summary[
                (summary["model_name"] == record["model_name"]) & (summary["benchmark"] == benchmark)
            ]
            actual_count = int(first_non_null(subset["n_samples"], 0)) if not subset.empty else 0
            complete = actual_count == expected_count
            row[f"{benchmark}_n_samples"] = actual_count
            row[f"{benchmark}_expected_samples"] = expected_count
            row[f"{benchmark}_complete"] = complete
            is_complete &= complete
        row["is_complete"] = is_complete
        if row["is_primary_model"] and is_complete:
            row["report_bucket"] = "main"
            row["exclusion_reason"] = ""
        elif is_complete:
            row["report_bucket"] = "appendix"
            row["exclusion_reason"] = "complete_but_non_core"
        else:
            row["report_bucket"] = "excluded"
            row["exclusion_reason"] = "incomplete_coverage"
        rows.append(row)

    audit_df = pd.DataFrame(rows).sort_values(
        ["report_bucket", "model_family", "model_display_name"],
        kind="stable",
    )
    return audit_df.reset_index(drop=True)


def filter_main_report_rows(
    summary_df: pd.DataFrame,
    *,
    run_label: str = "ea",
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    summary = _with_model_metadata(summary_df)
    audit = build_completeness_audit(run_label, summary_df=summary) if audit_df is None else audit_df
    allowed = set(audit.loc[audit["report_bucket"] == "main", "model_name"])
    filtered = summary[summary["model_name"].isin(allowed)].copy()
    return filtered.sort_values(["benchmark", "model_family", "model_display_name"], kind="stable").reset_index(drop=True)


def filter_appendix_rows(
    summary_df: pd.DataFrame,
    *,
    run_label: str = "ea",
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    summary = _with_model_metadata(summary_df)
    audit = build_completeness_audit(run_label, summary_df=summary) if audit_df is None else audit_df
    allowed = set(audit.loc[audit["report_bucket"] == "appendix", "model_name"])
    filtered = summary[summary["model_name"].isin(allowed)].copy()
    return filtered.sort_values(["benchmark", "model_family", "model_display_name"], kind="stable").reset_index(drop=True)


def compute_pairwise_ea_deltas(
    run_label: str = "ea",
    *,
    outcomes_df: pd.DataFrame | None = None,
    allowed_models: list[str] | None = None,
) -> pd.DataFrame:
    outcomes = _with_model_metadata(compute_sample_outcomes(run_label) if outcomes_df is None else outcomes_df)
    if outcomes.empty:
        return pd.DataFrame()
    if allowed_models is not None:
        outcomes = outcomes[outcomes["model_name"].isin(allowed_models)].copy()
    rows: list[dict[str, Any]] = []
    for benchmark, group in outcomes.groupby("benchmark", dropna=False):
        pivot = group.pivot_table(
            index="sample_id",
            columns="model_display_name",
            values="first_hit",
            aggfunc="first",
        )
        model_lookup = (
            group[["model_display_name", "model_name", "model_key", "model_family"]]
            .drop_duplicates()
            .set_index("model_display_name")
            .to_dict(orient="index")
        )
        for left_name, right_name in combinations(sorted(pivot.columns.tolist()), 2):
            paired = pivot[[left_name, right_name]].dropna()
            if paired.empty:
                continue
            deltas = (
                paired[left_name].astype(float).reset_index(drop=True)
                - paired[right_name].astype(float).reset_index(drop=True)
            ).tolist()
            ci_low, ci_high = bootstrap_interval(
                deltas,
                _mean_numeric,
                confidence_level=STATS_CFG["confidence_level"],
                n_resamples=STATS_CFG["bootstrap_resamples"],
                seed=STATS_CFG["bootstrap_seed"],
            )
            row = {
                "benchmark": benchmark,
                "left_model": left_name,
                "right_model": right_name,
                "left_model_name": model_lookup[left_name]["model_name"],
                "right_model_name": model_lookup[right_name]["model_name"],
                "left_model_key": model_lookup[left_name]["model_key"],
                "right_model_key": model_lookup[right_name]["model_key"],
                "left_family": model_lookup[left_name]["model_family"],
                "right_family": model_lookup[right_name]["model_family"],
                "comparison": f"{left_name} vs {right_name}",
                "matched_samples": len(deltas),
                "left_ea": float(paired[left_name].mean()),
                "right_ea": float(paired[right_name].mean()),
                "delta_ea": float(sum(deltas) / len(deltas)),
                "delta_ea_ci_low": ci_low,
                "delta_ea_ci_high": ci_high,
            }
            row.update(
                bootstrap_quantile_fields(
                    deltas,
                    _mean_numeric,
                    prefix="delta_ea",
                    quantiles=STATS_CFG["quantiles"],
                    n_resamples=STATS_CFG["bootstrap_resamples"],
                    seed=STATS_CFG["bootstrap_seed"],
                )
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["benchmark", "delta_ea"], ascending=[True, False], kind="stable").reset_index(drop=True)


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
