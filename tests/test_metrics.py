import pytest
from pathlib import Path

from src.evaluation.ea import execution_accuracy
from src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from src.evaluation.expert_score import ExpertEvaluation, expert_score
from src.evaluation.pass_at_k import compute_all_pass_at_k, pass_at_k
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


# ---------------------------------------------------------------------------
# Efficiency weights validation
# ---------------------------------------------------------------------------

_VALID_WEIGHTS = {"efficiency_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.2, "delta": 0.3}}
_RESULT = GenerationResult(
    sql="SELECT 1",
    raw_response="SELECT 1",
    tokens_input=10,
    tokens_output=5,
    latency_ms=100.0,
    model_name="m",
    metadata={"backend": "ollama"},
)


def test_efficiency_weights_sum_to_one_passes() -> None:
    compute_efficiency([_RESULT], _VALID_WEIGHTS)  # must not raise


def test_efficiency_weights_too_high_raises() -> None:
    bad = {"efficiency_weights": {"alpha": 0.4, "beta": 0.2, "gamma": 0.2, "delta": 0.3}}
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_efficiency([_RESULT], bad)


def test_efficiency_weights_too_low_raises() -> None:
    bad = {"efficiency_weights": {"alpha": 0.2, "beta": 0.2, "gamma": 0.2, "delta": 0.3}}
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_efficiency([_RESULT], bad)


# ---------------------------------------------------------------------------
# normalize_efficiency_rows
# ---------------------------------------------------------------------------

def _eff_row(tinf: float, tok: float) -> dict:
    return {
        "Tinf": tinf,
        "Mem": None,
        "Tok": tok,
        "Cost": 0.0,
        "_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.2, "delta": 0.3},
    }


def test_normalize_efficiency_min_is_zero_max_is_one() -> None:
    rows = [_eff_row(100.0, 50.0), _eff_row(200.0, 150.0)]
    result = normalize_efficiency_rows(rows)
    assert result[0]["Tinf_norm"] == pytest.approx(0.0)
    assert result[1]["Tinf_norm"] == pytest.approx(1.0)


def test_normalize_efficiency_span_zero_yields_zero() -> None:
    rows = [_eff_row(100.0, 100.0), _eff_row(100.0, 100.0)]
    result = normalize_efficiency_rows(rows)
    assert result[0]["Tinf_norm"] == pytest.approx(0.0)
    assert result[1]["Tinf_norm"] == pytest.approx(0.0)


def test_normalize_efficiency_none_component_stays_none() -> None:
    rows = [_eff_row(100.0, 50.0), _eff_row(200.0, 150.0)]
    result = normalize_efficiency_rows(rows)
    # Mem is None in both rows
    assert result[0]["Mem_norm"] is None
    assert result[1]["Mem_norm"] is None
    assert result[0]["Eff_normalized"] is None  # can't compute without Mem


def test_normalize_efficiency_does_not_mutate_input() -> None:
    rows = [_eff_row(100.0, 50.0), _eff_row(200.0, 150.0)]
    original_keys = set(rows[0].keys())
    normalize_efficiency_rows(rows)
    assert set(rows[0].keys()) == original_keys


# ---------------------------------------------------------------------------
# compute_all_pass_at_k
# ---------------------------------------------------------------------------

def test_compute_all_pass_at_k_consistency() -> None:
    results = [[True, False, True], [False, False, False]]
    k_values = [1, 2, 3]
    combined = compute_all_pass_at_k(results, k_values)
    for k in k_values:
        assert combined[k] == pytest.approx(pass_at_k(results, k))


def test_compute_all_pass_at_k_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_all_pass_at_k([[True]], [])


def test_compute_all_pass_at_k_nonpositive_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_all_pass_at_k([[True]], [0, 1])


def test_compute_all_pass_at_k_duplicates_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        compute_all_pass_at_k([[True]], [1, 1, 5])


# ---------------------------------------------------------------------------
# expert_score and ExpertEvaluation
# ---------------------------------------------------------------------------

def test_expert_score_formula() -> None:
    assert expert_score(3.0, 4.0, 5.0) == pytest.approx(4.0)
    assert expert_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert expert_score(5.0, 5.0, 5.0) == pytest.approx(5.0)


def test_expert_score_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        expert_score(6.0, 3.0, 3.0)
    with pytest.raises(ValueError):
        expert_score(3.0, 0.0, 3.0)


def test_expert_evaluation_score_equals_expert_score() -> None:
    ev = ExpertEvaluation(sample_id="s1", completeness=3, efficiency=4, readability=5)
    assert ev.score == pytest.approx(expert_score(3, 4, 5))


def test_expert_evaluation_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        ExpertEvaluation(sample_id="s1", completeness=6, efficiency=3, readability=3)
    with pytest.raises(ValueError):
        ExpertEvaluation(sample_id="s1", completeness=3, efficiency=0, readability=3)
