"""Strategy orchestration for NL2SQL production-style experiments."""

from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataset import TestCase
from .executor import SqlExecutor
from .logger import AttemptLog, CaseRunLog, save_case_logs
from .metrics import aggregate_case_logs
from .model import ModelInterface
from .retry import RetryPolicy
from .routing import RouteDecision, RuleBasedRouter
from .validation import ValidationModule, ValidationResult


def build_prompt(
    *,
    case: TestCase,
    schema_text: str,
    error_context: str | None = None,
) -> str:
    lines = [
        "You are an NL2SQL system.",
        "Return exactly one read-only SQL query.",
        schema_text,
        f"Question: {case.natural_language_query}",
    ]
    if case.db_target:
        lines.insert(3, f"Target: {case.db_target}")
    if error_context:
        lines.extend(["", error_context])
    return "\n\n".join(lines)


class StrategyRunner(ABC):
    """Base class for one strategy execution mode."""

    def __init__(
        self,
        *,
        model: ModelInterface,
        executor: SqlExecutor,
        validator: ValidationModule,
        retry_policy: RetryPolicy,
        router: RuleBasedRouter | None = None,
    ) -> None:
        self._model = model
        self._executor = executor
        self._validator = validator
        self._retry_policy = retry_policy
        self._router = router

    @abstractmethod
    def run_case(self, case: TestCase, schema_text: str) -> CaseRunLog:
        """Run one case through the strategy."""

    def _attempt_log(
        self,
        *,
        attempt_no: int,
        prompt: str,
        generated_sql: str,
        validation: ValidationResult,
        latency_ms: float,
        error_context_used: dict[str, str] | None,
        model_called: bool,
    ) -> AttemptLog:
        return AttemptLog(
            attempt_no=attempt_no,
            prompt=prompt,
            generated_sql=generated_sql,
            validation_status={
                "success": validation.success,
                "issues": [asdict(issue) for issue in validation.issues],
                "accuracy": validation.accuracy,
                "invalid_reference": validation.invalid_reference,
                "comparison_mode": validation.comparison_mode,
                "matched_expected_result": validation.matched_expected_result,
                "matched_expected_sql_result": validation.matched_expected_sql_result,
            },
            execution_outcome=(
                asdict(validation.execution_outcome)
                if validation.execution_outcome is not None
                else None
            ),
            latency_ms=latency_ms,
            error_context_used=error_context_used,
            model_called=model_called,
        )

    def _finalize_log(
        self,
        *,
        case: TestCase,
        route_decision: RouteDecision | None,
        attempt_logs: list[AttemptLog],
        generated_sql_final: str | None,
        started_at: float,
    ) -> CaseRunLog:
        last_attempt = attempt_logs[-1]
        accuracy_candidates = [
            bool(
                attempt.validation_status["accuracy"]
                if attempt.validation_status["accuracy"] is not None
                else False
            )
            for attempt in attempt_logs
        ]
        model_call_count = sum(1 for attempt in attempt_logs if attempt.model_called)
        model_latency_ms = sum(attempt.latency_ms for attempt in attempt_logs if attempt.model_called)
        final_metrics = {
            "execution_success": bool(last_attempt.validation_status["success"]),
            "execution_accuracy": last_attempt.validation_status["accuracy"],
            "candidate_accuracies": accuracy_candidates or [False],
            "end_to_end_latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "model_latency_ms": model_latency_ms,
        }
        return CaseRunLog(
            case_id=case.id,
            strategy=self.strategy_name,
            input={
                "natural_language_query": case.natural_language_query,
                "db_target": case.db_target,
                "metadata": case.metadata,
            },
            route_decision=asdict(route_decision) if route_decision is not None else None,
            attempts=len(attempt_logs),
            generated_sql_final=generated_sql_final,
            attempt_logs=attempt_logs,
            final_result=last_attempt.validation_status,
            errors=last_attempt.validation_status["issues"],
            metrics=final_metrics,
            timestamps={"started_at_monotonic": started_at},
            model_call_count=model_call_count,
        )


class GenerateOnlyStrategy(StrategyRunner):
    """Single generation attempt without repair."""

    strategy_name = "generate_only"

    def run_case(self, case: TestCase, schema_text: str) -> CaseRunLog:
        started_at = time.perf_counter()
        prompt = build_prompt(case=case, schema_text=schema_text)
        response = self._model.generate_sql(prompt)
        validation = self._validator.validate(case, response.sql, self._executor)
        attempt_log = self._attempt_log(
            attempt_no=1,
            prompt=prompt,
            generated_sql=response.sql,
            validation=validation,
            latency_ms=response.latency_ms,
            error_context_used=None,
            model_called=True,
        )
        return self._finalize_log(
            case=case,
            route_decision=None,
            attempt_logs=[attempt_log],
            generated_sql_final=response.sql,
            started_at=started_at,
        )


class GenerateValidateRetryStrategy(StrategyRunner):
    """Generate with validation-aware repair retries."""

    strategy_name = "generate_validate_retry"

    def run_case(self, case: TestCase, schema_text: str) -> CaseRunLog:
        started_at = time.perf_counter()
        attempt_logs: list[AttemptLog] = []
        error_context: dict[str, str] | None = None
        current_sql: str | None = None
        for attempt_no in range(1, self._retry_policy.max_attempts + 1):
            prompt = build_prompt(
                case=case,
                schema_text=schema_text,
                error_context=(
                    self._retry_policy.render_error_context(error_context) if error_context is not None else None
                ),
            )
            response = self._model.generate_sql(prompt)
            current_sql = response.sql
            validation = self._validator.validate(case, response.sql, self._executor)
            attempt_logs.append(
                self._attempt_log(
                    attempt_no=attempt_no,
                    prompt=prompt,
                    generated_sql=response.sql,
                    validation=validation,
                    latency_ms=response.latency_ms,
                    error_context_used=error_context,
                    model_called=True,
                )
            )
            if not self._retry_policy.should_retry(attempt_no, validation):
                break
            error_context = self._retry_policy.build_error_context(response.sql, validation)
        return self._finalize_log(
            case=case,
            route_decision=None,
            attempt_logs=attempt_logs,
            generated_sql_final=current_sql,
            started_at=started_at,
        )


class RoutingStrategy(StrategyRunner):
    """Reuse/adapt/generate strategy using a deterministic router."""

    strategy_name = "routing"

    def run_case(self, case: TestCase, schema_text: str) -> CaseRunLog:
        if self._router is None:
            raise RuntimeError("RoutingStrategy requires a router")
        started_at = time.perf_counter()
        decision = self._router.route(case.natural_language_query)
        prompt = ""
        latency_ms = 0.0
        model_called = False
        if decision.strategy == "generate":
            prompt = build_prompt(case=case, schema_text=schema_text)
            response = self._model.generate_sql(prompt)
            generated_sql = response.sql
            latency_ms = response.latency_ms
            model_called = True
        else:
            generated_sql = decision.sql or ""
        validation = self._validator.validate(case, generated_sql, self._executor)
        attempt_log = self._attempt_log(
            attempt_no=1,
            prompt=prompt,
            generated_sql=generated_sql,
            validation=validation,
            latency_ms=latency_ms,
            error_context_used=None,
            model_called=model_called,
        )
        return self._finalize_log(
            case=case,
            route_decision=decision,
            attempt_logs=[attempt_log],
            generated_sql_final=generated_sql,
            started_at=started_at,
        )


class ExperimentRunner:
    """Run all configured strategies over the same dataset."""

    def __init__(
        self,
        *,
        model: ModelInterface,
        executor: SqlExecutor,
        validator: ValidationModule | None = None,
        retry_policy: RetryPolicy | None = None,
        router: RuleBasedRouter | None = None,
        output_dir: Path,
    ) -> None:
        self._model = model
        self._executor = executor
        self._validator = validator or ValidationModule()
        self._retry_policy = retry_policy or RetryPolicy()
        self._router = router
        self._output_dir = output_dir

    def run(
        self,
        cases: list[TestCase],
        *,
        strategy: str = "all",
    ) -> dict[str, Any]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        schema_context = self._executor.get_schema_context()
        strategy_runners = self._select_strategies(strategy)
        all_logs: list[CaseRunLog] = []
        per_strategy_logs: dict[str, list[CaseRunLog]] = {}
        for strategy_runner in strategy_runners:
            logs = [strategy_runner.run_case(case, schema_context.text) for case in cases]
            per_strategy_logs[strategy_runner.strategy_name] = logs
            all_logs.extend(logs)
            save_case_logs(self._output_dir / f"per_case_{strategy_runner.strategy_name}.json", logs)
        summary = aggregate_case_logs(all_logs)
        self._write_summary(summary)
        return {
            "per_strategy_logs": per_strategy_logs,
            "summary_metrics": summary,
        }

    def _select_strategies(self, strategy: str) -> list[StrategyRunner]:
        available = {
            "generate_only": GenerateOnlyStrategy(
                model=self._model,
                executor=self._executor,
                validator=self._validator,
                retry_policy=self._retry_policy,
            ),
            "generate_validate_retry": GenerateValidateRetryStrategy(
                model=self._model,
                executor=self._executor,
                validator=self._validator,
                retry_policy=self._retry_policy,
            ),
            "routing": RoutingStrategy(
                model=self._model,
                executor=self._executor,
                validator=self._validator,
                retry_policy=self._retry_policy,
                router=self._router,
            ),
        }
        if strategy == "all":
            if self._router is None:
                del available["routing"]
            return list(available.values())
        if strategy == "routing" and self._router is None:
            raise RuntimeError("Routing strategy requested without catalog/router")
        if strategy not in available:
            raise ValueError(f"Unsupported strategy: {strategy}")
        return [available[strategy]]

    def _write_summary(self, summary: dict[str, dict[str, Any]]) -> None:
        json_path = self._output_dir / "summary_metrics.json"
        csv_path = self._output_dir / "summary_metrics.csv"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fieldnames = [
            "strategy",
            "n_cases",
            "execution_success_rate",
            "execution_accuracy",
            "pass_at_3",
            "latency_ms",
            "model_latency_ms",
            "cost_model_calls",
            "recovery_rate",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary.values():
                writer.writerow({field: row.get(field) for field in fieldnames})
