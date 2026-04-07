"""Tests for InferenceBackend.extract_sql()."""

from nl2sql.src.inference.base import InferenceBackend, normalize_sql_text

extract_sql = InferenceBackend.extract_sql


def test_extract_sql_plain() -> None:
    assert extract_sql("SELECT * FROM users") == "SELECT * FROM users"


def test_extract_sql_normalizes_literal_escaped_newline_prefix() -> None:
    assert extract_sql("\\nSELECT count(*) FROM singer;") == "SELECT count(*) FROM singer"


def test_normalize_sql_text_handles_literal_escaped_whitespace() -> None:
    assert normalize_sql_text("\\nSELECT\\t1;") == "SELECT\t1"


def test_extract_sql_json_only_response() -> None:
    assert extract_sql('{"sql":"SELECT * FROM users"}') == "SELECT * FROM users"


def test_extract_sql_json_embedded_in_text() -> None:
    raw = 'Here is the answer: {"sql":"SELECT COUNT(*) FROM users;"}'
    assert extract_sql(raw) == "SELECT COUNT(*) FROM users"


def test_extract_sql_sql_fence() -> None:
    raw = "```sql\nSELECT * FROM users\n```"
    assert extract_sql(raw) == "SELECT * FROM users"


def test_extract_sql_generic_fence() -> None:
    raw = "```\nSELECT 1\n```"
    assert extract_sql(raw) == "SELECT 1"


def test_extract_sql_strips_trailing_semicolon() -> None:
    assert extract_sql("SELECT 1;") == "SELECT 1"


def test_extract_sql_empty_returns_empty() -> None:
    assert extract_sql("") == ""
    assert extract_sql("   ") == ""


def test_extract_sql_refusal_returns_empty() -> None:
    assert extract_sql("I cannot generate SQL for this request.") == ""
    assert extract_sql("Sorry, I'm unable to generate SQL here.") == ""


def test_extract_sql_multiline_inside_fence() -> None:
    raw = "```sql\n  SELECT id,\n         name\n  FROM users\n  WHERE id = 1\n```"
    result = extract_sql(raw)
    assert "SELECT" in result
    assert "FROM users" in result
    assert result.strip() == result  # no leading/trailing whitespace
