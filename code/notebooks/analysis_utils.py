from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from shared.config import load_yaml_config


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
    return _load_csv(get_metrics_dir(run_label) / "summary_metrics.csv", required)


@lru_cache(maxsize=8)
def load_sample_metrics(run_label: str, *, required: bool = False) -> pd.DataFrame:
    run_label = _normalize_run_label(run_label)
    df = _load_csv(get_metrics_dir(run_label) / "sample_metrics.csv", required)
    if df.empty:
        return df
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
            "model_name": record.get("model_name"),
            "benchmark": record.get("benchmark"),
            "run_label": record.get("run_label"),
            "model_key": record.get("model_key"),
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


def pairwise_metric_deltas(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty or metric not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for benchmark, group in summary_df.groupby("benchmark"):
        values = group[["model_name", metric]].dropna()
        records = values.to_dict("records")
        for left in records:
            for right in records:
                if left["model_name"] >= right["model_name"]:
                    continue
                rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "left_model": left["model_name"],
                        "right_model": right["model_name"],
                        "delta": float(left[metric]) - float(right[metric]),
                    }
                )
    return pd.DataFrame(rows)


def model_family(name: str) -> str:
    """Classify a model name into M1 (large API) or M2 (compact local)."""
    m1_names = {"DeepSeek", "ChatGPT"}
    return "M1" if name in m1_names else "M2"


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

    template_df = sample_df[["sample_id", "model_name", "benchmark"]].copy()
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
