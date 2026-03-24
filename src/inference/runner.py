"""Batch inference orchestration."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.data.loader import DataSample
from src.inference.base import InferenceBackend
from src.prompt.template import PromptBuilder


LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    """Run inference over a benchmark split and persist raw generations."""

    def __init__(
        self,
        backend: InferenceBackend,
        prompt_builder: PromptBuilder,
        output_dir: Path,
    ) -> None:
        """Initialize a batch experiment runner."""
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        samples: list[DataSample],
        model_name: str,
        benchmark: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Path:
        """Run inference over all samples and append to a JSONL file."""
        output_path = self._resolve_output_path(model_name=model_name, benchmark=benchmark)
        completed_ids = self._load_completed_sample_ids(output_path)

        with output_path.open("a", encoding="utf-8") as handle:
            for sample in tqdm(samples, desc=f"{model_name}:{benchmark}", unit="sample"):
                if sample.id in completed_ids:
                    continue
                try:
                    prompt = self._prompt_builder.build(sample)
                    generations = await self._backend.generate(
                        prompt,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    record = {
                        "sample_id": sample.id,
                        "benchmark": sample.benchmark,
                        "db_id": sample.db_id,
                        "db_path": str(sample.db_path),
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
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed_ids.add(sample.id)
                except Exception as exc:
                    LOGGER.warning("Skipping sample %s after inference failure: %s", sample.id, exc)
        return output_path

    def _resolve_output_path(self, *, model_name: str, benchmark: str) -> Path:
        """Reuse the latest JSONL for a model+benchmark pair or create a new one."""
        pattern = f"{model_name}_{benchmark}_*.jsonl"
        existing = list(self._output_dir.glob(pattern))
        if existing:
            latest = max(existing, key=lambda path: path.stat().st_mtime)
            LOGGER.info("Resuming into existing result file %s", latest)
            return latest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"{model_name}_{benchmark}_{timestamp}.jsonl"

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
