import asyncio
import csv
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import ea as ea_module
from src.evaluation.ea import candidate_execution_matches, execution_accuracy
from src.evaluation.executor import ExecutionResult
from src.evaluation.efficiency import compute_efficiency, normalize_efficiency_rows
from src.evaluation.expert_score import ExpertEvaluation, expert_score
from src.evaluation.pass_at_k import compute_all_pass_at_k, pass_at_k
from src.inference.api_backend import APIBackend
from src.inference.base import GenerationResult


_EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "script_03_evaluate",
    Path(__file__).resolve().parents[1] / "scripts" / "03_evaluate.py",
)
assert _EVALUATE_SPEC is not None and _EVALUATE_SPEC.loader is not None
_EVALUATE_MODULE = importlib.util.module_from_spec(_EVALUATE_SPEC)
_EVALUATE_SPEC.loader.exec_module(_EVALUATE_MODULE)


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


def test_candidate_execution_matches_executes_gold_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_execute_sql(sql: str, db_path: Path, timeout: int = 30) -> ExecutionResult:
        calls.append(sql)
        if sql == "SELECT gold":
            return ExecutionResult(success=True, rows=[("1",)])
        if sql == "SELECT hit":
            return ExecutionResult(success=True, rows=[("1",)])
        return ExecutionResult(success=True, rows=[("2",)])

    monkeypatch.setattr(ea_module, "execute_sql", fake_execute_sql)

    result = candidate_execution_matches(
        ["SELECT hit", "SELECT miss", "SELECT hit"],
        "SELECT gold",
        Path("demo.sqlite"),
    )

    assert result == [True, False, True]
    assert calls.count("SELECT gold") == 1
    assert calls.count("SELECT hit") == 2
    assert calls.count("SELECT miss") == 1


def test_evaluate_candidate_predictions_returns_sample_level_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_execute_sql(sql: str, db_path: Path, timeout: int = 30) -> ExecutionResult:
        calls.append(sql)
        if sql == "SELECT gold":
            return ExecutionResult(success=True, rows=[("1",)])
        if sql == "SELECT hit":
            return ExecutionResult(success=True, rows=[("1",)])
        if sql == "SELECT broken":
            return ExecutionResult(success=False, rows=None, error="syntax error")
        return ExecutionResult(success=True, rows=[("2",)])

    monkeypatch.setattr(ea_module, "execute_sql", fake_execute_sql)

    result = ea_module.evaluate_candidate_predictions(
        ["SELECT broken", "SELECT hit", "SELECT miss"],
        "SELECT gold",
        Path("demo.sqlite"),
    )

    assert result == {
        "gold_success": True,
        "gold_error": None,
        "candidate_hits": [False, True, False],
        "first_pred_success": False,
        "first_pred_error": "syntax error",
    }
    assert calls.count("SELECT gold") == 1


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


def test_compute_efficiency_ollama_cost_is_zero() -> None:
    metrics = compute_efficiency([_RESULT], _VALID_WEIGHTS)
    assert metrics["Cost"] == 0.0


def test_resolve_db_path_supports_relative_and_absolute(tmp_path: Path) -> None:
    relative = _EVALUATE_MODULE._resolve_db_path("spider/database/db/db.sqlite", tmp_path)
    absolute = _EVALUATE_MODULE._resolve_db_path("/tmp/demo.sqlite", tmp_path)

    assert relative == tmp_path / "spider/database/db/db.sqlite"
    assert absolute == Path("/tmp/demo.sqlite")


def test_resolve_db_path_supports_legacy_data_prefixed_paths(tmp_path: Path) -> None:
    legacy = tmp_path / "data" / "spider" / "database" / "db" / "db.sqlite"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")

    resolved = _EVALUATE_MODULE._resolve_db_path("data/spider/database/db/db.sqlite", tmp_path / "data")

    assert resolved == legacy


def test_validate_records_rejects_missing_db_path(tmp_path: Path) -> None:
    grouped = {
        ("Demo", "spider", "pass_k"): [
            {
                "sample_id": "s1",
                "model_name": "Demo",
                "benchmark": "spider",
                "run_label": "pass_k",
                "db_path": "spider/database/missing/missing.sqlite",
                "generations": [{"sql": "SELECT 1"} for _ in range(10)],
                "_source_path": "demo.jsonl",
            }
        ]
    }

    with pytest.raises(ValueError, match="db_path does not exist"):
        _EVALUATE_MODULE._validate_records(
            grouped,
            experiment_cfg={"k_values": [1, 5, 10]},
            data_dir=tmp_path / "data",
        )


def test_validate_records_rejects_mixed_legacy_and_labeled_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "spider" / "database" / "db" / "db.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")

    grouped = {
        ("Demo", "spider", "legacy"): [
            {
                "sample_id": "s1",
                "model_name": "Demo",
                "benchmark": "spider",
                "db_path": "spider/database/db/db.sqlite",
                "generations": [{"sql": "SELECT 1"}],
                "_source_path": "legacy.jsonl",
            }
        ],
        ("Demo", "spider", "pass_k"): [
            {
                "sample_id": "s1",
                "model_name": "Demo",
                "benchmark": "spider",
                "run_label": "pass_k",
                "db_path": "spider/database/db/db.sqlite",
                "generations": [{"sql": "SELECT 1"} for _ in range(10)],
                "_source_path": "pass_k.jsonl",
            }
        ],
    }

    with pytest.raises(ValueError, match="Mixed legacy and labeled raw files"):
        _EVALUATE_MODULE._validate_records(
            grouped,
            experiment_cfg={"k_values": [1, 5, 10]},
            data_dir=tmp_path / "data",
        )


def test_validate_records_rejects_wrong_generation_count(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "spider" / "database" / "db" / "db.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")

    grouped = {
        ("Demo", "spider", "pass_k"): [
            {
                "sample_id": "s1",
                "model_name": "Demo",
                "benchmark": "spider",
                "run_label": "pass_k",
                "db_path": "spider/database/db/db.sqlite",
                "generations": [{"sql": "SELECT 1"}],
                "_source_path": "pass_k.jsonl",
            }
        ]
    }

    with pytest.raises(ValueError, match="expected 10"):
        _EVALUATE_MODULE._validate_records(
            grouped,
            experiment_cfg={"k_values": [1, 5, 10]},
            data_dir=tmp_path / "data",
        )


def test_evaluate_writes_sample_metrics_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "metrics"
    data_dir = tmp_path / "data"
    db_path = data_dir / "spider" / "database" / "db" / "db.sqlite"
    db_path.parent.mkdir(parents=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE demo (value INTEGER)")
        connection.execute("INSERT INTO demo (value) VALUES (1)")
        connection.commit()

    record = {
        "sample_id": "s1",
        "model_name": "Demo",
        "benchmark": "spider",
        "run_label": "ea",
        "question": "How many rows?",
        "gold_sql": "SELECT COUNT(*) FROM demo",
        "db_id": "db",
        "db_path": "spider/database/db/db.sqlite",
        "difficulty": "easy",
        "evidence": "",
        "generations": [
            {
                "sql": "SELECT COUNT(*) FROM demo",
                "raw_response": "SELECT COUNT(*) FROM demo",
                "tokens_input": 10,
                "tokens_output": 5,
                "latency_ms": 12.5,
                "model_name": "Demo",
                "metadata": {"backend": "ollama"},
            }
        ],
    }
    (raw_dir / "demo.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        _EVALUATE_MODULE,
        "parse_args",
        lambda: SimpleNamespace(
            config_dir=Path("configs"),
            raw_dir=raw_dir,
            data_dir=data_dir,
            output_dir=output_dir,
        ),
    )

    _EVALUATE_MODULE.main()

    sample_metrics_path = output_dir / "sample_metrics.csv"
    assert sample_metrics_path.exists()
    with sample_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["candidate_hits"] == "[true]"
    assert rows[0]["first_hit"] == "True"
    assert rows[0]["first_pred_success"] == "True"


def test_notebook_helper_loads_persisted_sample_metrics_without_sql_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_utils_spec = importlib.util.spec_from_file_location(
        "analysis_utils_under_test",
        Path(__file__).resolve().parents[1] / "notebooks" / "analysis_utils.py",
    )
    assert analysis_utils_spec is not None and analysis_utils_spec.loader is not None
    analysis_utils = importlib.util.module_from_spec(analysis_utils_spec)
    analysis_utils_spec.loader.exec_module(analysis_utils)

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    sample_metrics_path = metrics_dir / "sample_metrics.csv"
    sample_metrics_path.write_text(
        "\n".join(
            [
                "sample_id,model_name,benchmark,run_label,candidate_hits,first_hit,any_hit,first_pred_success,empty_sql",
                's1,Demo,spider,ea,"[true, false]",True,True,False,False',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_execute_sql(*args, **kwargs):
        raise AssertionError("execute_sql should not be called when sample_metrics.csv exists")

    monkeypatch.setattr(analysis_utils, "SAMPLE_METRICS_PATH", sample_metrics_path)
    monkeypatch.setattr(analysis_utils, "execute_sql", fail_execute_sql)
    analysis_utils.load_sample_metrics.cache_clear()
    analysis_utils.compute_sample_outcomes.cache_clear()

    outcomes_df = analysis_utils.compute_sample_outcomes()

    assert len(outcomes_df) == 1
    assert outcomes_df.iloc[0]["candidate_hits"] == [True, False]
    assert bool(outcomes_df.iloc[0]["first_hit"]) is True
    assert bool(outcomes_df.iloc[0]["first_pred_success"]) is False


def test_api_backend_includes_normalized_pricing_metadata() -> None:
    class DummyCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
                choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT 1"))],
            )

    backend = APIBackend(
        model_id="demo-model",
        base_url="https://example.com",
        api_key="test-key",
        model_name="Demo",
        pricing={"input_per_1m": 0.28, "output_per_1m": 0.42, "cache_hit_per_1m": 0.028},
    )
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    results = asyncio.run(
        backend._generate_native(
            prompt="question",
            n=1,
            temperature=0.0,
            max_tokens=32,
            seed=None,
            top_p=None,
        )
    )

    assert len(results) == 1
    assert results[0].metadata["backend"] == "api"
    assert results[0].metadata["pricing"] == {
        "input_per_mtok": pytest.approx(0.28),
        "output_per_mtok": pytest.approx(0.42),
        "cache_hit_per_mtok": pytest.approx(0.028),
    }


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
