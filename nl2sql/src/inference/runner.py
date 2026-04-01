"""Batch inference orchestration."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from nl2sql.src.data.loader import DataSample
from nl2sql.src.inference.base import InferenceBackend
from shared.logging_utils import ProgressType, create_progress
from nl2sql.src.prompt.template import PromptBuilder


LOGGER = logging.getLogger(__name__)

# Flush and fsync to disk every N completed samples.  Writing fsync after
# every single record serializes I/O significantly on large benchmark runs.
_FSYNC_EVERY_N = 50


class ExperimentRunner:
    """Run inference over a benchmark split and persist raw generations."""

    def __init__(
        self,
        backend: InferenceBackend,
        prompt_builder: PromptBuilder,
        output_dir: Path,
        data_root: Path | None = None,
    ) -> None:
        """Initialize a batch experiment runner."""
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._output_dir = output_dir
        self._data_root = data_root.resolve(strict=False) if data_root is not None else None
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        samples: list[DataSample],
        model_name: str,
        benchmark: str,
        run_label: str | None = None,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
        progress: ProgressType | None = None,
    ) -> Path:
        """Run inference over all samples and append to a JSONL file."""
        output_path = self._resolve_output_path(
            model_name=model_name,
            benchmark=benchmark,
            run_label=run_label,
        )
        completed_ids = self._load_completed_sample_ids(output_path)
        pending_samples = [sample for sample in samples if sample.id not in completed_ids]
        resumed_count = len(completed_ids)
        written_count = 0
        error_count = 0
        total_latency_ms = 0.0
        owns_progress = progress is None
        if progress is None:
            progress = create_progress()

        with output_path.open("a", encoding="utf-8") as handle:
            with progress if owns_progress else nullcontext(progress) as active_progress:
                task_id = active_progress.add_task(
                    f"{model_name}:{benchmark}",
                    total=len(samples),
                    completed=resumed_count,
                    status=_format_status(
                        resumed_count=resumed_count,
                        written_count=written_count,
                        error_count=error_count,
                        avg_latency_ms=0.0,
                    ),
                )
                for sample in pending_samples:
                    try:
                        prompt = self._prompt_builder.build(sample)
                        generations = await self._backend.generate(
                            prompt,
                            n=n,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed,
                            top_p=top_p,
                        )
                        record = {
                            "sample_id": sample.id,
                            "benchmark": sample.benchmark,
                            "run_label": run_label,
                            "db_id": sample.db_id,
                            "db_path": self._serialize_db_path(sample.db_path),
                            "question": sample.question,
                            "gold_sql": sample.gold_sql,
                            "difficulty": sample.difficulty,
                            "evidence": sample.evidence,
                            "model_name": model_name,
                            "generations": [
                                {
                                    "sql": generation.sql,
                                    "raw_response": generation.raw_response,
                                    "tokens_input": generation.tokens_input,
                                    "tokens_output": generation.tokens_output,
                                    "latency_ms": generation.latency_ms,
                                    "model_name": generation.model_name,
                                    "metadata": generation.metadata,
                                    "timestamp": generation.timestamp,
                                }
                                for generation in generations
                            ],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed_ids.add(sample.id)
                        written_count += 1
                        total_latency_ms += sum(generation.latency_ms for generation in generations) / max(
                            len(generations), 1
                        )
                        handle.flush()
                        if len(completed_ids) % _FSYNC_EVERY_N == 0:
                            os.fsync(handle.fileno())
                    except Exception as exc:
                        error_count += 1
                        LOGGER.warning("Skipping sample %s after inference failure: %s", sample.id, exc)
                    finally:
                        avg_latency_ms = total_latency_ms / written_count if written_count else 0.0
                        active_progress.update(
                            task_id,
                            advance=1,
                            status=_format_status(
                                resumed_count=resumed_count,
                                written_count=written_count,
                                error_count=error_count,
                                avg_latency_ms=avg_latency_ms,
                            ),
                        )
                active_progress.remove_task(task_id)
            handle.flush()
            os.fsync(handle.fileno())
        return output_path

    def _serialize_db_path(self, db_path: Path) -> str:
        """Prefer a data-root-relative DB path for portable result files."""
        if self._data_root is None:
            return str(db_path)

        try:
            return db_path.resolve(strict=False).relative_to(self._data_root).as_posix()
        except ValueError:
            return str(db_path)

    def _resolve_output_path(
        self,
        *,
        model_name: str,
        benchmark: str,
        run_label: str | None = None,
    ) -> Path:
        """Reuse the latest JSONL for a model+benchmark pair or create a new one."""
        parts = [model_name, benchmark]
        if run_label:
            parts.append(run_label)
        stem = "_".join(parts)
        pattern = f"{stem}_*.jsonl"
        existing = list(self._output_dir.glob(pattern))
        if existing:
            latest = max(existing, key=lambda path: path.stat().st_mtime)
            LOGGER.info("Resuming into existing result file %s", latest)
            return latest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"{stem}_{timestamp}.jsonl"

    @staticmethod
    def _load_completed_sample_ids(path: Path) -> set[str]:
        """Read already completed sample IDs from an existing JSONL file."""
        if not path.exists():
            return set()

        completed: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed JSONL line in %s", path)
                    continue
                sample_id = payload.get("sample_id")
                if isinstance(sample_id, str) and sample_id:
                    completed.add(sample_id)
        return completed


def _format_status(
    *,
    resumed_count: int,
    written_count: int,
    error_count: int,
    avg_latency_ms: float,
) -> str:
    return (
        f"skip={resumed_count} ok={written_count} "
        f"err={error_count} avg_ms={avg_latency_ms:.1f}"
    )
