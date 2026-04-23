"""Retry policy and repair context helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ValidationResult


@dataclass(slots=True)
class RetryPolicy:
    """Retry policy for generate+validate strategy."""

    max_attempts: int = 3

    def should_retry(self, attempt_no: int, validation: ValidationResult) -> bool:
        return attempt_no < self.max_attempts and not validation.success

    def build_error_context(self, previous_sql: str, validation: ValidationResult) -> dict[str, str]:
        return {
            "previous_sql": previous_sql,
            "error_type": validation.error_type or "validation_error",
            "error_message": validation.error_message or "Unknown validation failure",
            "repair_instruction": (
                "Fix the SQL using the error details and return exactly one read-only SQL query."
            ),
        }

    def render_error_context(self, context: dict[str, str]) -> str:
        return (
            "Previous SQL:\n"
            f"{context['previous_sql']}\n\n"
            "Validation error:\n"
            f"{context['error_type']}: {context['error_message']}\n\n"
            f"{context['repair_instruction']}"
        )
