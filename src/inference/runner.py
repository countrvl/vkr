"""Batch inference orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.loader import DataSample
from src.inference.base import InferenceBackend
from src.prompt.template import PromptBuilder


class ExperimentRunner:
    """Run inference over a benchmark split and persist raw generations."""

    def __init__(
        self,
        *,
        experiment_config: dict[str, Any],
        backend: InferenceBackend,
        prompt_builder: PromptBuilder,
        output_dir: Path,
    ) -> None:
        self._experiment_config = experiment_config
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        samples: list[DataSample],
        *,
        benchmark: str,
        model_name: str,
        n: int,
        temperature: float,
    ) -> Path:
        """Run inference and append generations to a JSONL file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self._output_dir / f"{model_name}_{benchmark}_{timestamp}.jsonl"

        with output_path.open("a", encoding="utf-8") as handle:
            for sample in tqdm(samples, desc=f"{model_name}:{benchmark}", unit="sample"):
                prompt = self._prompt_builder.render(sample)
                generations = await self._backend.generate(prompt, n=n, temperature=temperature)
                for index, generation in enumerate(generations):
                    record = {
                        "timestamp_utc": timestamp,
                        "benchmark": benchmark,
                        "model_name": model_name,
                        "sample": {
                            **asdict(sample),
                            "db_path": str(sample.db_path),
                        },
                        "prompt": {
                            "template": "nl2sql.j2",
                            "temperature": temperature,
                            "n": n,
                        },
                        "generation_index": index,
                        "generation": {
                            "sql": generation.sql,
                            "raw_response": generation.raw_response,
                            "tokens_input": generation.tokens_input,
                            "tokens_output": generation.tokens_output,
                            "latency_ms": generation.latency_ms,
                            "model_name": generation.model_name,
                            "metadata": generation.metadata,
                        },
                        "experiment": self._experiment_config,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path
