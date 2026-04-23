"""Structured run logs for strategy-bench executions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AttemptLog:
    """One attempt within a strategy run."""

    attempt_no: int
    prompt: str
    generated_sql: str
    validation_status: dict[str, Any]
    execution_outcome: dict[str, Any] | None
    latency_ms: float
    error_context_used: dict[str, str] | None
    model_called: bool = False


@dataclass(slots=True)
class CaseRunLog:
    """Serializable run log for a single case and strategy."""

    case_id: str
    strategy: str
    input: dict[str, Any]
    route_decision: dict[str, Any] | None
    attempts: int
    generated_sql_final: str | None
    attempt_logs: list[AttemptLog]
    final_result: dict[str, Any]
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, Any] = field(default_factory=dict)
    model_call_count: int = 0


def save_case_logs(path: Path, logs: list[CaseRunLog]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(log) for log in logs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
