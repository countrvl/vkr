from pathlib import Path

from nl2sql.src.synthetic_bench.prepare import (
    DATASET_PATH,
    EDGE_CASES_PATH,
    build_snapshot_db,
    load_core_queries,
    load_edge_queries,
    validate_queries,
)


def test_synthetic_dataset_contract() -> None:
    core_queries = load_core_queries(DATASET_PATH)
    edge_queries = load_edge_queries(EDGE_CASES_PATH)

    assert len(core_queries) == 30
    assert len([q for q in core_queries if q.difficulty == "easy"]) == 10
    assert len([q for q in core_queries if q.difficulty == "medium"]) == 10
    assert len([q for q in core_queries if q.difficulty == "hard"]) == 10
    assert 5 <= len(edge_queries) <= 10


def test_synthetic_snapshot_builds_and_validates(tmp_path: Path) -> None:
    db_path = build_snapshot_db(output_path=tmp_path / "synthetic.sqlite", force=True)
    issues, summary = validate_queries(db_path=db_path)

    assert issues == []
    assert summary["query_counts"]["core"] == 30
    assert summary["query_counts"]["edge_case"] >= 5
    assert summary["row_counts"]["customers"] >= 300
    assert summary["row_counts"]["orders"] >= 400
    assert summary["unused_tables"] == []
    assert summary["intent_distribution"]["aggregation"] >= 1
    assert summary["selectivity_distribution"]["selective"] >= 1
    assert "data_quality_summary" in summary
    assert summary["data_quality_summary"]["unusual_value_checks"]["orders.payment_method:crypto"] > 0
    assert summary["data_quality_summary"]["unusual_value_checks"]["returns.return_reason:empty_string"] > 0
