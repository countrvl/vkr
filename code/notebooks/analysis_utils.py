from __future__ import annotations

import json
import os
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from shared.config import load_domain_models, load_yaml_config
from shared.evaluation.statistics import bootstrap_quantile_fields


def detect_project_root() -> Path:
    """Return the repository root when running from the repo or notebook directories."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if (candidate / "code").exists() and (candidate / "shared").exists():
            return candidate
    return cwd


PROJECT_ROOT = detect_project_root()
DOMAIN_ROOT = PROJECT_ROOT / "code"
CONFIG_DIR = DOMAIN_ROOT / "configs"
BENCHMARKS_CFG = load_yaml_config(CONFIG_DIR / "benchmarks.yaml")
EXPERIMENT_CFG = load_yaml_config(CONFIG_DIR / "experiment.yaml")
METRICS_CFG = load_yaml_config(CONFIG_DIR / "metrics.yaml")
MODEL_DISPLAY_LOOKUP: dict[str, dict[str, Any]] = {}
for key, cfg in load_domain_models("supports_code").items():
    lookup_row = {
        "display_name": cfg.get("display_name") or cfg.get("name") or key,
        "version": cfg.get("version"),
        "key": key,
        "family": str(cfg.get("family", "")).upper() or ("M1" if key.startswith("m1_") else "M2"),
        "active_by_default": bool(cfg.get("active_by_default", True)),
    }
    for alias in {key, cfg.get("name"), cfg.get("display_name")}:
        if alias:
            MODEL_DISPLAY_LOOKUP[str(alias)] = lookup_row


def first_non_null(values: Any, default: Any = None) -> Any:
    """Return the first non-null item from a Series-like value or a scalar."""
    if isinstance(values, pd.Series):
        non_null = values.dropna().tolist()
        return non_null[0] if non_null else default
    return default if pd.isna(values) else values


def model_display_name(record: dict[str, Any]) -> Any:
    model_key = record.get("model_key")
    model_name = record.get("model_name")
    lookup = MODEL_DISPLAY_LOOKUP.get(str(model_key)) or MODEL_DISPLAY_LOOKUP.get(str(model_name))
    return record.get("model_display_name") or (lookup or {}).get("display_name") or model_name


def model_version(record: dict[str, Any]) -> Any:
    model_key = record.get("model_key")
    model_name = record.get("model_name")
    lookup = MODEL_DISPLAY_LOOKUP.get(str(model_key)) or MODEL_DISPLAY_LOOKUP.get(str(model_name))
    return record.get("model_version") or (lookup or {}).get("version")


def model_key(record: dict[str, Any]) -> Any:
    model_name = record.get("model_name")
    return record.get("model_key") or MODEL_DISPLAY_LOOKUP.get(str(model_name), {}).get("key")


def model_family(name_or_key: Any) -> str:
    """Classify a model label into M1 (large API) or M2 (specialized/local)."""
    lookup = MODEL_DISPLAY_LOOKUP.get(str(name_or_key))
    if lookup and lookup.get("family"):
        return str(lookup["family"]).upper()
    return "M1" if name_or_key in {"DeepSeek", "ChatGPT"} else "M2"


def ensure_model_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add display/version/family columns for old metrics files when needed."""
    if df.empty or "model_name" not in df.columns:
        return df
    enriched = df.copy()
    if "model_key" not in enriched.columns:
        enriched["model_key"] = enriched.apply(lambda row: model_key(row.to_dict()), axis=1)
    if "model_display_name" not in enriched.columns:
        enriched["model_display_name"] = enriched.apply(
            lambda row: model_display_name(row.to_dict()),
            axis=1,
        )
    if "model_version" not in enriched.columns:
        enriched["model_version"] = enriched.apply(lambda row: model_version(row.to_dict()), axis=1)
    if "family" not in enriched.columns:
        enriched["family"] = enriched["model_key"].map(model_family)
    if "active_by_default" not in enriched.columns:
        enriched["active_by_default"] = enriched["model_key"].map(
            lambda key: bool(MODEL_DISPLAY_LOOKUP.get(str(key), {}).get("active_by_default", True))
        )
    return enriched


EXPECTED_SAMPLE_COUNTS = {
    "humaneval_plus": 164,
    "mbpp_plus": 378,
}

PRIMARY_METRIC_COLUMNS = (
    "functional_correctness_q05",
    "functional_correctness_q50",
    "functional_correctness_q95",
)


def get_results_dir() -> Path:
    explicit_results_dir = os.getenv("CODE_RESULTS_DIR")
    if explicit_results_dir:
        return Path(explicit_results_dir).expanduser().resolve()
    return PROJECT_ROOT / "results" / "code"


def get_metrics_dir(run_label: str | None = None) -> Path:
    metrics_dir = get_results_dir() / "metrics"
    if run_label:
        metrics_dir = metrics_dir / run_label
    return metrics_dir


def get_figures_dir(run_label: str | None = None) -> Path:
    figures_dir = get_results_dir() / "figures"
    if run_label:
        figures_dir = figures_dir / run_label
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _normalize_run_label(run_label: str) -> str:
    return "fc" if run_label == "ea" else run_label


def _load_csv(path: Path, required: bool) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Missing metrics artifact: {path}")
    return pd.DataFrame()


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
def load_summary_metrics(run_label: str, *, required: bool = False) -> pd.DataFrame:
    run_label = _normalize_run_label(run_label)
    return ensure_model_label_columns(_load_csv(get_metrics_dir(run_label) / "summary_metrics.csv", required))


@lru_cache(maxsize=8)
def load_sample_metrics(run_label: str, *, required: bool = False) -> pd.DataFrame:
    run_label = _normalize_run_label(run_label)
    df = _load_csv(get_metrics_dir(run_label) / "sample_metrics.csv", required)
    if df.empty:
        return df
    df = ensure_model_label_columns(df)
    if "candidate_hits" in df.columns:
        df["candidate_hits"] = df["candidate_hits"].map(_parse_candidate_hits)
    for col in ["first_hit", "any_hit", "first_compiled_ok", "first_tests_passed"]:
        if col in df.columns:
            df[col] = df[col].map(_parse_bool)
    return df


@lru_cache(maxsize=8)
def load_candidate_metrics(run_label: str, *, required: bool = False) -> pd.DataFrame:
    run_label = _normalize_run_label(run_label)
    df = _load_csv(get_metrics_dir(run_label) / "candidate_metrics.csv", required)
    if df.empty:
        return df
    df = ensure_model_label_columns(df)
    for col in ["compiled_ok", "tests_passed", "functional_correctness"]:
        if col in df.columns:
            df[col] = df[col].map(_parse_bool)
    return df


def _infer_run_label(record: dict[str, Any], source_path: str) -> str:
    run_label = record.get("run_label")
    if isinstance(run_label, str) and run_label.strip():
        return _normalize_run_label(run_label.strip())
    if "_pass_k_" in source_path:
        return "pass_k"
    return "fc"


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
                record["run_label"] = _infer_run_label(record, str(path))
                if run_label is not None and record["run_label"] != run_label:
                    continue
                records.append(record)
    return records


@lru_cache(maxsize=8)
def load_records(run_label: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample-level and generation-level DataFrames from raw JSONL."""
    sample_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for record in _iter_raw_records(get_results_dir(), run_label):
        sample_row = {
            "sample_id": record.get("sample_id"),
            "model_key": model_key(record),
            "model_name": record.get("model_name"),
            "model_display_name": model_display_name(record),
            "model_version": model_version(record),
            "benchmark": record.get("benchmark"),
            "run_label": record.get("run_label"),
            "entry_point": record.get("entry_point"),
            "prompt": record.get("prompt", ""),
            "contract": record.get("contract", ""),
            "prompt_len": len(record.get("prompt", "")),
            "n_generations": len(record.get("generations", [])),
            "source_path": record["_source_path"],
        }
        sample_rows.append(sample_row)

        for idx, gen in enumerate(record.get("generations", []), start=1):
            generation_rows.append(
                {
                    "sample_id": record.get("sample_id"),
                    "model_name": record.get("model_name"),
                    "model_display_name": model_display_name(record),
                    "model_version": model_version(record),
                    "benchmark": record.get("benchmark"),
                    "run_label": record.get("run_label"),
                    "generation_index": idx,
                    "code": gen.get("code", ""),
                    "raw_response": gen.get("raw_response", ""),
                    "tokens_input": gen.get("tokens_input", 0),
                    "tokens_output": gen.get("tokens_output", 0),
                    "latency_ms": gen.get("latency_ms", 0.0),
                    "cost_usd": gen.get("cost_usd"),
                    "metadata": gen.get("metadata", {}),
                    "timestamp": gen.get("timestamp"),
                }
            )

    return pd.DataFrame(sample_rows), pd.DataFrame(generation_rows)


def available_run_labels() -> list[str]:
    labels: list[str] = []
    metrics_root = get_results_dir() / "metrics"
    for run_label in ("fc", "pass_k"):
        if (metrics_root / run_label / "summary_metrics.csv").exists():
            labels.append(run_label)
    return labels


def get_expected_sample_counts(run_label: str = "fc") -> dict[str, int]:
    run_label = _normalize_run_label(run_label)
    if run_label != "fc":
        return {}
    return EXPECTED_SAMPLE_COUNTS.copy()


def build_completeness_audit(
    summary_df: pd.DataFrame,
    *,
    run_label: str = "fc",
    expected_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    run_label = _normalize_run_label(run_label)
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "model_name",
                "model_display_name",
                "family",
                "active_by_default",
                "benchmarks_expected",
                "benchmarks_complete",
                "is_complete_main",
                "missing_benchmarks",
                "missing_details",
            ]
        )
    expected_counts = expected_counts or get_expected_sample_counts(run_label)
    df = ensure_model_label_columns(summary_df)
    df = df[df["run_label"].map(_normalize_run_label) == run_label].copy()
    rows: list[dict[str, Any]] = []
    for model_key_value, group in df.groupby("model_key", dropna=False):
        benchmark_map = {str(row["benchmark"]): int(row["n_samples"]) for _, row in group.iterrows()}
        missing_benchmarks: list[str] = []
        missing_details: list[str] = []
        complete_count = 0
        for benchmark, expected in expected_counts.items():
            actual = benchmark_map.get(benchmark, 0)
            if actual == expected:
                complete_count += 1
            else:
                missing_benchmarks.append(benchmark)
                missing_details.append(f"{benchmark}: {actual}/{expected}")
        record = group.iloc[0]
        rows.append(
            {
                "model_key": model_key_value,
                "model_name": first_non_null(record.get("model_name")),
                "model_display_name": first_non_null(record.get("model_display_name")),
                "family": first_non_null(record.get("family")),
                "active_by_default": bool(first_non_null(record.get("active_by_default"), True)),
                "benchmarks_expected": len(expected_counts),
                "benchmarks_complete": complete_count,
                "is_complete_main": complete_count == len(expected_counts),
                "missing_benchmarks": ", ".join(missing_benchmarks),
                "missing_details": "; ".join(missing_details),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "model_display_name"]).reset_index(drop=True)


def filter_main_report_rows(
    summary_df: pd.DataFrame,
    *,
    run_label: str = "fc",
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    audit_df = audit_df if audit_df is not None else build_completeness_audit(summary_df, run_label=run_label)
    allowed = set(
        audit_df.loc[audit_df["is_complete_main"] & audit_df["active_by_default"], "model_key"].astype(str)
    )
    df = ensure_model_label_columns(summary_df)
    return df[df["model_key"].astype(str).isin(allowed)].copy()


def filter_appendix_rows(
    summary_df: pd.DataFrame,
    *,
    run_label: str = "fc",
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    audit_df = audit_df if audit_df is not None else build_completeness_audit(summary_df, run_label=run_label)
    appendix = audit_df[
        (~audit_df["is_complete_main"]) | (~audit_df["active_by_default"])
    ]["model_key"].astype(str)
    df = ensure_model_label_columns(summary_df)
    return df[df["model_key"].astype(str).isin(set(appendix))].copy()


def compute_pairwise_fc_deltas(
    sample_df: pd.DataFrame,
    *,
    allowed_models: list[str] | None = None,
    n_resamples: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    if sample_df.empty or "first_hit" not in sample_df.columns:
        return pd.DataFrame()
    stats_cfg = METRICS_CFG.get("statistics", {})
    n_resamples = int(n_resamples or stats_cfg.get("bootstrap_resamples", 1000))
    seed = int(seed or stats_cfg.get("bootstrap_seed", 42))

    df = ensure_model_label_columns(sample_df)
    df = df[df["run_label"].map(_normalize_run_label) == "fc"].copy()
    if allowed_models is not None:
        df = df[df["model_display_name"].isin(allowed_models)].copy()
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for benchmark, benchmark_df in df.groupby("benchmark"):
        model_names = sorted(benchmark_df["model_display_name"].dropna().unique().tolist())
        for left_model, right_model in combinations(model_names, 2):
            left = benchmark_df[benchmark_df["model_display_name"] == left_model][
                ["sample_id", "first_hit"]
            ].rename(columns={"first_hit": "left_hit"})
            right = benchmark_df[benchmark_df["model_display_name"] == right_model][
                ["sample_id", "first_hit"]
            ].rename(columns={"first_hit": "right_hit"})
            paired = left.merge(right, on="sample_id", how="inner")
            if paired.empty:
                continue
            deltas = (paired["left_hit"].astype(float) - paired["right_hit"].astype(float)).tolist()
            point_estimate = sum(deltas) / len(deltas)
            quantiles = bootstrap_quantile_fields(
                deltas,
                lambda sample: sum(float(x) for x in sample) / len(sample),
                prefix="delta",
                quantiles=stats_cfg.get("quantiles", (0.05, 0.5, 0.95)),
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "left_model": left_model,
                    "right_model": right_model,
                    "n_pairs": len(deltas),
                    "delta": point_estimate,
                    **quantiles,
                }
            )
    return pd.DataFrame(rows).sort_values(["benchmark", "delta"], ascending=[True, False]).reset_index(drop=True)


def pairwise_metric_deltas(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty or metric not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for benchmark, group in summary_df.groupby("benchmark"):
        model_col = "model_display_name" if "model_display_name" in group.columns else "model_name"
        values = group[[model_col, metric]].dropna()
        records = values.to_dict("records")
        for left in records:
            for right in records:
                if left[model_col] >= right[model_col]:
                    continue
                rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "left_model": left[model_col],
                        "right_model": right[model_col],
                        "delta": float(left[metric]) - float(right[metric]),
                    }
                )
    return pd.DataFrame(rows)


def ensure_expert_template(run_label: str | None = None) -> Path:
    """Create an expert score template CSV if a populated file is absent."""
    metrics_dir = get_metrics_dir(run_label)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    expert_scores_path = metrics_dir / "expert_scores.csv"
    template_path = metrics_dir / "expert_scores_template.csv"

    if expert_scores_path.exists():
        return expert_scores_path

    sample_df = load_sample_metrics(run_label, required=False)
    if sample_df.empty:
        return template_path

    template_cols = ["sample_id", "model_name", "model_display_name", "benchmark"]
    template_df = sample_df[[col for col in template_cols if col in sample_df.columns]].copy()
    template_df["completeness"] = pd.NA
    template_df["efficiency"] = pd.NA
    template_df["readability"] = pd.NA
    template_df.to_csv(template_path, index=False)
    return template_path


def reset_analysis_caches() -> None:
    load_summary_metrics.cache_clear()
    load_sample_metrics.cache_clear()
    load_candidate_metrics.cache_clear()
    load_records.cache_clear()
