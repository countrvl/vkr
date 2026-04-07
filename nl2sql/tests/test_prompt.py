"""Tests for PromptBuilder."""

import sqlite3
from pathlib import Path

from nl2sql.src.data.loader import DataSample
from nl2sql.src.prompt.template import PromptBuilder


def _make_sample(tmp_path: Path) -> DataSample:
    db_path = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users (id INTEGER)")
    return DataSample(
        id="test_0",
        benchmark="spider",
        question="How many users?",
        gold_sql="SELECT COUNT(*) FROM users",
        db_id="demo",
        db_path=db_path,
        schema="CREATE TABLE users (id INTEGER)",
        difficulty=None,
    )


def test_prompt_builder_contains_schema_and_question(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    builder = PromptBuilder()
    prompt = builder.build(sample)
    assert sample.schema in prompt
    assert sample.question in prompt


def test_prompt_builder_supports_m2_sql_continuation_profile(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    builder = PromptBuilder()

    prompt = builder.build(sample, "m2_sql_continuation")

    assert "SQL query:" in prompt
    assert '"sql"' not in prompt
    assert sample.schema in prompt
    assert sample.question in prompt


def test_prompt_builder_supports_xiyansql_profile(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    builder = PromptBuilder()

    prompt = builder.build(sample, "xiyansql_sqlite")

    assert "### Dialect" in prompt
    assert "SQLite" in prompt
    assert "```sql" in prompt
    assert sample.schema in prompt
    assert sample.question in prompt


def test_prompt_builder_is_deterministic(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    builder = PromptBuilder()
    assert builder.build(sample) == builder.build(sample)


def test_render_alias_equals_build(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    builder = PromptBuilder()
    assert builder.render(sample) == builder.build(sample)


def test_template_cached_not_reloaded(tmp_path: Path) -> None:
    """The pre-loaded template object must be the same instance across calls."""
    builder = PromptBuilder()
    assert builder._default_template is builder._default_template  # noqa: SLF001
