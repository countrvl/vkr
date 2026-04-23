"""Benchmark loaders for the code-generation domain."""

from __future__ import annotations

import json
from pathlib import Path

from code_bench.data.prepare import load_evalplus_tasks, normalize_benchmark_name
from code_bench.data.schema import CodeSample


def load_benchmark(
    benchmark: str,
    data_dir: Path,
    *,
    mini: bool = False,
    noextreme: bool = False,
) -> list[CodeSample]:
    """Load benchmark samples into a unified schema."""
    benchmark = normalize_benchmark_name(benchmark)
    local_samples = _load_prepared_samples(benchmark, data_dir, mini=mini, noextreme=noextreme)
    if local_samples is not None:
        return local_samples

    tasks = load_evalplus_tasks(benchmark, mini=mini, noextreme=noextreme)

    samples: list[CodeSample] = []
    for task_id, task in sorted(tasks.items()):
        samples.append(
            CodeSample(
                id=task_id,
                benchmark=benchmark,
                prompt_text=task["prompt"],
                entry_point=task["entry_point"],
                canonical_solution=task.get("canonical_solution", ""),
                contract=task.get("contract", ""),
                metadata={
                    "atol": task.get("atol", 0),
                    "n_base_tests": len(task.get("base_input", [])),
                    "n_plus_tests": len(task.get("plus_input", [])),
                    "has_assertion": "assertion" in task,
                    "mini": bool(mini),
                    "noextreme": bool(noextreme),
                    "source": "evalplus",
                },
            )
        )
    return samples


def _load_prepared_samples(
    benchmark: str,
    data_dir: Path,
    *,
    mini: bool,
    noextreme: bool,
) -> list[CodeSample] | None:
    manifest_path = data_dir / benchmark / "manifest.json"
    metadata_path = data_dir / benchmark / "metadata.jsonl"
    if not manifest_path.exists() or not metadata_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if bool(manifest.get("mini", False)) != bool(mini):
        return None
    if bool(manifest.get("noextreme", False)) != bool(noextreme):
        return None

    samples: list[CodeSample] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                CodeSample(
                    id=row["sample_id"],
                    benchmark=benchmark,
                    prompt_text=row["prompt_text"],
                    entry_point=row["entry_point"],
                    canonical_solution=row.get("canonical_solution", ""),
                    contract=row.get("contract", ""),
                    metadata={
                        "atol": row.get("atol", 0),
                        "n_base_tests": row.get("n_base_tests", 0),
                        "n_plus_tests": row.get("n_plus_tests", 0),
                        "has_assertion": bool(row.get("has_assertion", False)),
                        "mini": bool(mini),
                        "noextreme": bool(noextreme),
                        "dataset_hash": manifest.get("dataset_hash"),
                        "source": "prepared_metadata",
                    },
                )
            )
    return samples
