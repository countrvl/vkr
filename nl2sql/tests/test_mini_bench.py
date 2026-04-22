from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nl2sql.src.inference.base import GenerationResult, InferenceBackend
from nl2sql.src.mini_bench.prepare import EXPECTED_CASE_COUNT, build_snapshot_db, validate_dataset
from nl2sql.src.mini_bench.runner import (
    aggregate_breakdown,
    aggregate_case_results,
    collect_failure_examples,
    mini_case_to_sample,
    run_mini_bench,
)


def test_prepare_mini_bench_builds_snapshot_and_validates_cases(tmp_path: Path) -> None:
    db_path = tmp_path / "mini.sqlite"
    built_path = build_snapshot_db(output_path=db_path, force=True)

    assert built_path.exists()

    cases, issues = validate_dataset(db_path=built_path)

    assert len(cases) == EXPECTED_CASE_COUNT
    assert issues == []


def test_mini_bench_aggregates_results() -> None:
    results = [
        {
            "case_id": "a",
            "category": "filtering",
            "difficulty": "simple",
            "accuracy": True,
            "execution_success": True,
            "model_latency_ms": 10.0,
            "generated_sql": "SELECT 1",
            "expected_sql": "SELECT 1",
            "question": "Q1",
            "error_type": None,
            "error_message": None,
        },
        {
            "case_id": "b",
            "category": "filtering",
            "difficulty": "simple",
            "accuracy": False,
            "execution_success": False,
            "model_latency_ms": 20.0,
            "generated_sql": "SELECT broken",
            "expected_sql": "SELECT 1",
            "question": "Q2",
            "error_type": "syntax_error",
            "error_message": "bad sql",
        },
        {
            "case_id": "c",
            "category": "join",
            "difficulty": "challenging",
            "accuracy": True,
            "execution_success": True,
            "model_latency_ms": 30.0,
            "generated_sql": "SELECT 3",
            "expected_sql": "SELECT 3",
            "question": "Q3",
            "error_type": None,
            "error_message": None,
        },
    ]

    summary = aggregate_case_results(results)
    by_category = aggregate_breakdown(results, "category")
    failures = collect_failure_examples(results, limit=5)

    assert summary["n_cases"] == 3
    assert summary["execution_accuracy"] == 2 / 3
    assert summary["execution_success_rate"] == 2 / 3
    assert summary["model_latency_ms"] == 20.0
    assert by_category[0]["category"] == "filtering"
    assert by_category[0]["n_cases"] == 2
    assert failures[0]["case_id"] == "b"


@dataclass(slots=True)
class _StubCase:
    id: str
    natural_language_query: str
    expected_sql: str
    metadata: dict[str, str]
    db_target: str | None = None


class _FakeBackend(InferenceBackend):
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        self.prompts.append(prompt)
        return [
            GenerationResult(
                sql=self.sql,
                raw_response=self.sql,
                tokens_input=10,
                tokens_output=5,
                latency_ms=12.0,
                model_name="fake",
            )
        ]


def test_mini_case_to_sample_uses_main_data_sample_contract(tmp_path: Path) -> None:
    db_path = build_snapshot_db(output_path=tmp_path / "mini.sqlite", force=True)
    case = _StubCase(
        id="mini_01",
        natural_language_query="List active customer names.",
        expected_sql="SELECT customer_name FROM customers WHERE status = 'active'",
        metadata={"difficulty": "simple", "category": "filtering"},
        db_target="mini_db",
    )

    sample = mini_case_to_sample(case, db_path=db_path)

    assert sample.id == "mini_01"
    assert sample.benchmark == "mini_bench"
    assert sample.question == case.natural_language_query
    assert sample.gold_sql == case.expected_sql
    assert sample.db_id == "mini_db"
    assert sample.db_path == db_path
    assert "CREATE TABLE customers" in sample.schema
    assert sample.difficulty == "simple"


def test_run_mini_bench_uses_main_prompt_templates(monkeypatch, tmp_path: Path) -> None:
    db_path = build_snapshot_db(output_path=tmp_path / "mini.sqlite", force=True)
    cases, issues = validate_dataset(db_path=db_path)
    assert issues == []
    case = cases[0]
    backend = _FakeBackend(case.expected_sql or "SELECT 1")

    monkeypatch.setattr(
        "nl2sql.src.mini_bench.runner._build_backend",
        lambda model_key, models_cfg: backend,
    )

    outcome = run_mini_bench(
        model_key="m1_chatgpt",
        db_path=db_path,
        output_dir=tmp_path / "results",
        limit=1,
    )

    assert outcome["summary"]["n_cases"] == 1
    assert outcome["summary"]["execution_accuracy"] == 1.0
    assert outcome["results"][0]["prompt_profile"] == "nl2sql_json"
    assert backend.prompts, "Expected at least one prompt to be sent to the backend"
    assert 'Return a JSON object with exactly one field: "sql".' in backend.prompts[0]
