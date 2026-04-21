from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "code" / "notebooks" / "analysis_utils.py"
_SPEC = importlib.util.spec_from_file_location("_code_notebook_analysis_utils", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load analysis utils from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PROJECT_ROOT = _MODULE.PROJECT_ROOT
DOMAIN_ROOT = _MODULE.DOMAIN_ROOT
CONFIG_DIR = _MODULE.CONFIG_DIR
BENCHMARKS_CFG = _MODULE.BENCHMARKS_CFG
EXPERIMENT_CFG = _MODULE.EXPERIMENT_CFG
METRICS_CFG = _MODULE.METRICS_CFG
MODEL_DISPLAY_LOOKUP = _MODULE.MODEL_DISPLAY_LOOKUP
EXPECTED_SAMPLE_COUNTS = _MODULE.EXPECTED_SAMPLE_COUNTS

available_run_labels = _MODULE.available_run_labels
build_completeness_audit = _MODULE.build_completeness_audit
compute_pairwise_fc_deltas = _MODULE.compute_pairwise_fc_deltas
detect_project_root = _MODULE.detect_project_root
ensure_expert_template = _MODULE.ensure_expert_template
ensure_model_label_columns = _MODULE.ensure_model_label_columns
filter_appendix_rows = _MODULE.filter_appendix_rows
filter_main_report_rows = _MODULE.filter_main_report_rows
get_expected_sample_counts = _MODULE.get_expected_sample_counts
get_figures_dir = _MODULE.get_figures_dir
get_metrics_dir = _MODULE.get_metrics_dir
get_results_dir = _MODULE.get_results_dir
load_candidate_metrics = _MODULE.load_candidate_metrics
load_records = _MODULE.load_records
load_sample_metrics = _MODULE.load_sample_metrics
load_summary_metrics = _MODULE.load_summary_metrics
model_display_name = _MODULE.model_display_name
model_family = _MODULE.model_family
model_key = _MODULE.model_key
model_version = _MODULE.model_version
pairwise_metric_deltas = _MODULE.pairwise_metric_deltas
reset_analysis_caches = _MODULE.reset_analysis_caches

__all__ = [
    "BENCHMARKS_CFG",
    "CONFIG_DIR",
    "DOMAIN_ROOT",
    "EXPECTED_SAMPLE_COUNTS",
    "EXPERIMENT_CFG",
    "METRICS_CFG",
    "MODEL_DISPLAY_LOOKUP",
    "PROJECT_ROOT",
    "available_run_labels",
    "build_completeness_audit",
    "compute_pairwise_fc_deltas",
    "detect_project_root",
    "ensure_expert_template",
    "ensure_model_label_columns",
    "filter_appendix_rows",
    "filter_main_report_rows",
    "get_expected_sample_counts",
    "get_figures_dir",
    "get_metrics_dir",
    "get_results_dir",
    "load_candidate_metrics",
    "load_records",
    "load_sample_metrics",
    "load_summary_metrics",
    "model_display_name",
    "model_family",
    "model_key",
    "model_version",
    "pairwise_metric_deltas",
    "reset_analysis_caches",
]
