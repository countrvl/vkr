"""Пакетная оркестрация инференса для benchmark-ов генерации кода."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from code_bench.data.schema import CodeSample
from code_bench.inference.base import InferenceBackend
from code_bench.prompt.template import PromptBuilder
from shared.logging_utils import ProgressType, create_progress


LOGGER = logging.getLogger(__name__)
_FSYNC_EVERY_N = 200


class ExperimentRunner:
    """Запустить инференс по срезу benchmark-а и сохранить raw-генерации."""

    def __init__(
        self,
        backend: InferenceBackend,
        prompt_builder: PromptBuilder,
        output_dir: Path,
    ) -> None:
        self._backend = backend
        self._prompt_builder = prompt_builder
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        samples: list[CodeSample],
        model_key: str,
        model_name: str,
        model_display_name: str | None,
        model_version: str | None,
        benchmark: str,
        run_label: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 768,
        seed: int | None = None,
        top_p: float | None = None,
        prompt_profile: str = "codegen_default",
        progress: ProgressType | None = None,
    ) -> Path:
        output_path = self._resolve_output_path(
            model_name=model_name,
            benchmark=benchmark,
            run_label=run_label,
        )
        completed_ids = self._load_completed_sample_ids(output_path)
        pending_samples = [sample for sample in samples if sample.id not in completed_ids]
        if getattr(self._backend, "supports_batch", False) and n == 1 and pending_samples:
            return await self._run_batch(
                pending_samples=pending_samples,
                completed_ids=completed_ids,
                output_path=output_path,
                model_key=model_key,
                model_name=model_name,
                model_display_name=model_display_name,
                model_version=model_version,
                benchmark=benchmark,
                run_label=run_label,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
                prompt_profile=prompt_profile,
                total_samples=len(samples),
                progress=progress,
            )
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
                        prompt = self._prompt_builder.build(sample, prompt_profile=prompt_profile)
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
                            "model_key": model_key,
                            "entry_point": sample.entry_point,
                            "prompt": sample.prompt_text,
                            "contract": sample.contract,
                            "prompt_profile": prompt_profile,
                            "benchmark_mini": bool(sample.metadata.get("mini", False)),
                            "benchmark_noextreme": bool(sample.metadata.get("noextreme", False)),
                            "benchmark_dataset_hash": sample.metadata.get("dataset_hash"),
                            "model_name": model_name,
                            "model_display_name": model_display_name or model_name,
                            "model_version": model_version,
                            "generations": [
                                {
                                    "candidate_index": idx,
                                    "code": generation.code,
                                    "raw_response": generation.raw_response,
                                    "tokens_input": generation.tokens_input,
                                    "tokens_output": generation.tokens_output,
                                    "latency_ms": generation.latency_ms,
                                    "cost_usd": generation.metadata.get("cost_usd"),
                                    "seed": seed,
                                    "model_name": generation.model_name,
                                    "metadata": generation.metadata,
                                    "timestamp": generation.timestamp,
                                }
                                for idx, generation in enumerate(generations, start=1)
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
                        LOGGER.warning("Skipping sample %s after inference failure: %s", sample.id, exc, exc_info=True)
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

    async def _run_batch(
        self,
        *,
        pending_samples: list[CodeSample],
        completed_ids: set[str],
        output_path: Path,
        model_key: str,
        model_name: str,
        model_display_name: str | None,
        model_version: str | None,
        benchmark: str,
        run_label: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
        prompt_profile: str,
        total_samples: int,
        progress: ProgressType | None,
    ) -> Path:
        resumed_count = len(completed_ids)
        written_count = 0
        error_count = 0
        total_latency_ms = 0.0
        owns_progress = progress is None
        if progress is None:
            progress = create_progress()

        samples_for_batch: list[CodeSample] = []
        prompts: list[str] = []

        with progress if owns_progress else nullcontext(progress) as active_progress:
            task_id = active_progress.add_task(
                f"{model_name}:{benchmark}",
                total=total_samples,
                completed=resumed_count,
                status="submitting batch",
            )
            manifest_path = self._manifest_path(output_path)
            existing_batch_id = self._resumable_batch_id(
                manifest_path=manifest_path,
                output_path=output_path,
                benchmark=benchmark,
                model_name=model_name,
                run_label=run_label,
                completed_ids=completed_ids,
            )
            if existing_batch_id is None:
                self._write_batch_manifest(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    benchmark=benchmark,
                    model_key=model_key,
                    model_name=model_name,
                    run_label=run_label,
                    n_requests=0,
                    status="preparing",
                )
            for sample in pending_samples:
                try:
                    prompts.append(self._prompt_builder.build(sample, prompt_profile=prompt_profile))
                    samples_for_batch.append(sample)
                except Exception as exc:
                    error_count += 1
                    LOGGER.warning(
                        "Skipping sample %s after prompt build failure: %s",
                        sample.id,
                        exc,
                        exc_info=True,
                    )
                    active_progress.update(
                        task_id,
                        advance=1,
                        status=_format_status(
                            resumed_count=resumed_count,
                            written_count=written_count,
                            error_count=error_count,
                            avg_latency_ms=0.0,
                        ),
                    )

            if not prompts:
                active_progress.remove_task(task_id)
                return output_path

            if hasattr(self._backend, "set_batch_status_callback"):
                def _status_callback(payload: dict[str, object]) -> None:
                    status = str(payload.get("status") or "unknown")
                    batch_id = str(payload.get("batch_id") or "")
                    phase = str(payload.get("phase") or "")
                    elapsed = payload.get("elapsed_seconds")
                    message = f"batch={batch_id or '-'} {phase}:{status}"
                    if elapsed is not None:
                        message += f" {elapsed}s"
                    active_progress.update(task_id, status=message)
                    self._write_batch_manifest(
                        manifest_path=manifest_path,
                        output_path=output_path,
                        benchmark=benchmark,
                        model_key=model_key,
                        model_name=model_name,
                        run_label=run_label,
                        n_requests=len(prompts),
                        status=status,
                        batch_id=batch_id or None,
                        phase=phase or None,
                        elapsed_seconds=float(elapsed) if elapsed is not None else None,
                    )

                self._backend.set_batch_status_callback(_status_callback)

            if existing_batch_id is not None and hasattr(self._backend, "resume_batch"):
                active_progress.update(task_id, status=f"batch={existing_batch_id} resuming")
                self._write_batch_manifest(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    benchmark=benchmark,
                    model_key=model_key,
                    model_name=model_name,
                    run_label=run_label,
                    n_requests=len(prompts),
                    status="resuming",
                    batch_id=existing_batch_id,
                    phase="resuming",
                )
                batch_results = await self._backend.resume_batch(
                    batch_id=existing_batch_id,
                    prompts=prompts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )
            else:
                self._write_batch_manifest(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    benchmark=benchmark,
                    model_key=model_key,
                    model_name=model_name,
                    run_label=run_label,
                    n_requests=len(prompts),
                    status="submitting",
                )
                batch_results = await self._backend.generate_batch(
                    prompts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )

            self._write_batch_manifest(
                manifest_path=manifest_path,
                output_path=output_path,
                benchmark=benchmark,
                model_key=model_key,
                model_name=model_name,
                run_label=run_label,
                n_requests=len(prompts),
                status="materializing",
            )

            with output_path.open("a", encoding="utf-8") as handle:
                final_batch_id: str | None = None
                for sample, result in zip(samples_for_batch, batch_results):
                    try:
                        if isinstance(result, Exception):
                            raise result
                        if final_batch_id is None:
                            final_batch_id = next(
                                (
                                    generation.metadata.get("batch_id")
                                    for generation in result
                                    if generation.metadata.get("batch_id")
                                ),
                                None,
                            )
                        record = {
                            "sample_id": sample.id,
                            "benchmark": sample.benchmark,
                            "run_label": run_label,
                            "model_key": model_key,
                            "entry_point": sample.entry_point,
                            "prompt": sample.prompt_text,
                            "contract": sample.contract,
                            "prompt_profile": prompt_profile,
                            "benchmark_mini": bool(sample.metadata.get("mini", False)),
                            "benchmark_noextreme": bool(sample.metadata.get("noextreme", False)),
                            "benchmark_dataset_hash": sample.metadata.get("dataset_hash"),
                            "model_name": model_name,
                            "model_display_name": model_display_name or model_name,
                            "model_version": model_version,
                            "generations": [
                                {
                                    "candidate_index": idx,
                                    "code": generation.code,
                                    "raw_response": generation.raw_response,
                                    "tokens_input": generation.tokens_input,
                                    "tokens_output": generation.tokens_output,
                                    "latency_ms": generation.latency_ms,
                                    "cost_usd": generation.metadata.get("cost_usd"),
                                    "seed": seed,
                                    "model_name": generation.model_name,
                                    "metadata": generation.metadata,
                                    "timestamp": generation.timestamp,
                                }
                                for idx, generation in enumerate(result, start=1)
                            ],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed_ids.add(sample.id)
                        written_count += 1
                        total_latency_ms += sum(g.latency_ms for g in result) / max(len(result), 1)
                        handle.flush()
                        if len(completed_ids) % _FSYNC_EVERY_N == 0:
                            os.fsync(handle.fileno())
                    except Exception as exc:
                        error_count += 1
                        LOGGER.warning(
                            "Skipping sample %s after batch inference failure: %s",
                            sample.id,
                            exc,
                            exc_info=True,
                        )
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
                handle.flush()
                os.fsync(handle.fileno())
            self._write_batch_manifest(
                manifest_path=manifest_path,
                output_path=output_path,
                benchmark=benchmark,
                model_key=model_key,
                model_name=model_name,
                run_label=run_label,
                n_requests=len(prompts),
                status="completed",
                batch_id=final_batch_id,
                phase="completed",
            )
            active_progress.remove_task(task_id)
        return output_path

    def _resolve_output_path(
        self,
        *,
        model_name: str,
        benchmark: str,
        run_label: str,
    ) -> Path:
        parts = [model_name, benchmark, run_label]
        stem = "_".join(parts)
        pattern = f"{stem}_*.jsonl"
        existing = list(self._output_dir.glob(pattern))
        if existing:
            latest = max(existing, key=lambda path: path.stat().st_mtime)
            LOGGER.info("Resuming into existing result file %s", latest)
            return latest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"{stem}_{timestamp}.jsonl"

    def _manifest_path(self, output_path: Path) -> Path:
        batches_dir = self._output_dir.parent / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        return batches_dir / f"{output_path.stem}.json"

    def _resumable_batch_id(
        self,
        *,
        manifest_path: Path,
        output_path: Path,
        benchmark: str,
        model_name: str,
        run_label: str,
        completed_ids: set[str],
    ) -> str | None:
        if completed_ids or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if manifest.get("raw_output_path") != str(output_path):
            return None
        if manifest.get("benchmark") != benchmark or manifest.get("model_name") != model_name:
            return None
        if manifest.get("run_label") != run_label:
            return None
        batch_id = manifest.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            return None
        status = str(manifest.get("status") or "").lower()
        if status in {"failed", "errored", "canceled", "cancelled", "expired"}:
            return None
        return batch_id

    def _write_batch_manifest(
        self,
        *,
        manifest_path: Path,
        output_path: Path,
        benchmark: str,
        model_key: str,
        model_name: str,
        run_label: str,
        n_requests: int,
        status: str,
        batch_id: str | None = None,
        phase: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        existing: dict[str, object] = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}

        manifest = {
            **existing,
            "model_key": model_key,
            "model_name": model_name,
            "benchmark": benchmark,
            "run_label": run_label,
            "n_requests": n_requests,
            "raw_output_path": str(output_path),
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }
        manifest.setdefault("created_at", datetime.now().isoformat())
        if batch_id is not None:
            manifest["batch_id"] = batch_id
        if phase is not None:
            manifest["phase"] = phase
        if elapsed_seconds is not None:
            manifest["elapsed_seconds"] = elapsed_seconds
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_completed_sample_ids(path: Path) -> set[str]:
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
