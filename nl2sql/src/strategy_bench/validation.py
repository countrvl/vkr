"""Validation flow for strategy-bench SQL outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import TestCase
from .executor import ExecutionOutcome, SqlExecutor, ValidationIssue, normalize_expected_result


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating one SQL candidate against a test case."""

    success: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    execution_outcome: ExecutionOutcome | None = None
    accuracy: bool | None = None
    invalid_reference: bool = False
    comparison_mode: str | None = None
    matched_expected_result: bool | None = None
    matched_expected_sql_result: bool | None = None

    @property
    def error_type(self) -> str | None:
        if self.issues:
            return self.issues[0].code
        if self.execution_outcome is not None:
            return self.execution_outcome.error_type
        return None

    @property
    def error_message(self) -> str | None:
        if self.issues:
            return self.issues[0].message
        if self.execution_outcome is not None:
            return self.execution_outcome.error_message
        return None


class ValidationModule:
    """Apply the configured validation flow to a generated SQL query."""

    def validate(self, case: TestCase, sql: str, executor: SqlExecutor) -> ValidationResult:
        syntax_issue = executor.explain_syntax(sql)
        if syntax_issue is not None:
            return ValidationResult(success=False, issues=[syntax_issue])

        execution_outcome = executor.execute(sql)
        if not execution_outcome.success:
            issue = ValidationIssue(
                code=execution_outcome.error_type or "execution_error",
                message=execution_outcome.error_message or "SQL execution failed",
            )
            return ValidationResult(
                success=False,
                issues=[issue],
                execution_outcome=execution_outcome,
            )

        result = ValidationResult(success=True, execution_outcome=execution_outcome)
        if case.expected_result is not None:
            expected_rows = normalize_expected_result(case.expected_result)
            result.comparison_mode = "expected_result"
            result.matched_expected_result = execution_outcome.rows == expected_rows
            result.accuracy = result.matched_expected_result
            return result

        if case.expected_sql is not None:
            reference_outcome = executor.execute(case.expected_sql)
            result.comparison_mode = "expected_sql_result"
            if not reference_outcome.success:
                result.invalid_reference = True
                result.accuracy = None
                issue = ValidationIssue(
                    code=reference_outcome.error_type or "invalid_reference",
                    message=reference_outcome.error_message or "Expected SQL failed during execution",
                )
                result.issues.append(issue)
                return result
            result.matched_expected_sql_result = execution_outcome.rows == reference_outcome.rows
            result.accuracy = result.matched_expected_sql_result
            return result

        return result
