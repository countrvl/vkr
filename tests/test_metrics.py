from pathlib import Path

from src.evaluation.ea import execution_accuracy
from src.evaluation.efficiency import compute_efficiency
from src.evaluation.pass_at_k import pass_at_k
from src.inference.base import GenerationResult


def test_execution_accuracy_on_matching_queries(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE users (id INTEGER)")
        connection.executemany("INSERT INTO users (id) VALUES (?)", [(1,), (2,)])
        connection.commit()

    score = execution_accuracy(
        ["SELECT COUNT(*) FROM users"],
        ["SELECT COUNT(*) FROM users"],
        [db_path],
    )
    assert score == 1.0


def test_pass_at_k_formula() -> None:
    score = pass_at_k([[False, True, False], [False, False, False]], k=2)
    assert round(score, 4) == 0.3333


def test_compute_efficiency_with_pricing() -> None:
    results = [
        GenerationResult(
            sql="SELECT 1",
            raw_response="SELECT 1",
            tokens_input=100,
            tokens_output=20,
            latency_ms=50.0,
            model_name="demo",
            metadata={
                "backend": "api",
                "memory_mb": 1024.0,
                "pricing": {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
            },
        )
    ]
    metrics = compute_efficiency(
        results,
        {"efficiency_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.2, "delta": 0.3}},
    )
    assert metrics["Tinf"] == 50.0
    assert metrics["Mem"] == 1024.0
    assert metrics["Tok"] == 120
    assert metrics["Cost"] is not None
    assert metrics["Eff"] is not None
