"""Benchmark preparation and EvalPlus integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)


_BENCHMARK_KEYS = ("humaneval_plus", "mbpp_plus")


def normalize_benchmark_name(name: str) -> str:
    if name not in _BENCHMARK_KEYS:
        raise ValueError(f"Unsupported code benchmark: {name}")
    return name


def load_evalplus_tasks(benchmark: str, *, mini: bool = False, noextreme: bool = False) -> dict[str, dict[str, Any]]:
    """Load EvalPlus tasks for a benchmark."""
    benchmark = normalize_benchmark_name(benchmark)
    if benchmark == "humaneval_plus":
        return get_human_eval_plus(err_incomplete=False, mini=mini, noextreme=noextreme)
    if mini:
        raise ValueError("MBPP+ does not support mini mode.")
    return get_mbpp_plus(err_incomplete=False, noextreme=noextreme)


def get_benchmark_hash(benchmark: str, *, mini: bool = False, noextreme: bool = False) -> str:
    """Return the benchmark content hash reported by EvalPlus."""
    benchmark = normalize_benchmark_name(benchmark)
    if benchmark == "humaneval_plus":
        return get_human_eval_plus_hash(mini=mini, noextreme=noextreme)
    if mini:
        raise ValueError("MBPP+ does not support mini mode.")
    return get_mbpp_plus_hash(noextreme=noextreme)


def build_metadata_records(benchmark: str, tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-serializable metadata rows for a benchmark."""
    rows: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        rows.append(
            {
                "sample_id": task_id,
                "benchmark": benchmark,
                "entry_point": task["entry_point"],
                "prompt_text": task["prompt"],
                "canonical_solution": task.get("canonical_solution", ""),
                "contract": task.get("contract", ""),
                "atol": task.get("atol", 0),
                "n_base_tests": len(task.get("base_input", [])),
                "n_plus_tests": len(task.get("plus_input", [])),
                "has_assertion": "assertion" in task,
            }
        )
    return rows


def prepare_benchmark_artifacts(
    benchmark: str,
    *,
    data_dir: Path,
    local_dir: Path,
    mini: bool = False,
    noextreme: bool = False,
) -> dict[str, Any]:
    """Download benchmark data via EvalPlus and persist local metadata."""
    benchmark = normalize_benchmark_name(benchmark)
    tasks = load_evalplus_tasks(benchmark, mini=mini, noextreme=noextreme)
    benchmark_hash = get_benchmark_hash(benchmark, mini=mini, noextreme=noextreme)
    local_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = local_dir / "metadata.jsonl"
    manifest_path = local_dir / "manifest.json"
    records = build_metadata_records(benchmark, tasks)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark": benchmark,
        "dataset_hash": benchmark_hash,
        "n_samples": len(tasks),
        "mini": bool(mini),
        "noextreme": bool(noextreme),
        "local_dir": str(local_dir),
        "metadata_path": str(metadata_path),
        "data_dir": str(data_dir),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
