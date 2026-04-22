import sqlite3
from pathlib import Path

from nl2sql.src.strategy_bench.dataset import TestCase
from nl2sql.src.strategy_bench.executor import SQLiteExecutor
from nl2sql.src.strategy_bench.metrics import aggregate_case_logs
from nl2sql.src.strategy_bench.model import ModelInterface, ModelResponse
from nl2sql.src.strategy_bench.retry import RetryPolicy
from nl2sql.src.strategy_bench.routing import CatalogEntry, MatchRule, RoutingCatalog, RuleBasedRouter
from nl2sql.src.strategy_bench.runner import ExperimentRunner


def _build_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        connection.execute("CREATE TABLE orders (customer TEXT, amount INTEGER)")
        connection.executemany("INSERT INTO users (id, name) VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
        connection.executemany(
            "INSERT INTO orders (customer, amount) VALUES (?, ?)",
            [("Alice", 10), ("Bob", 20)],
        )
        connection.commit()


class FakeModel(ModelInterface):
    def __init__(self) -> None:
        self.calls = 0

    def generate_sql(self, prompt: str) -> ModelResponse:
        self.calls += 1
        prompt_lower = prompt.lower()
        if "how many users" in prompt_lower:
            sql = "SELECT COUNT(*) FROM users" if "previous sql" in prompt_lower else "SELECT missing FROM users"
        elif "list all users" in prompt_lower:
            sql = "SELECT name FROM users"
        elif "orders for alice" in prompt_lower:
            sql = "SELECT amount FROM orders WHERE customer = 'Alice'"
        else:
            sql = "SELECT 1"
        return ModelResponse(sql=sql, raw_response=sql, latency_ms=5.0)


def test_metrics_compute_recovery_cost_and_accuracy() -> None:
    from nl2sql.src.strategy_bench.logger import AttemptLog, CaseRunLog

    logs = [
        CaseRunLog(
            case_id="case-1",
            strategy="generate_validate_retry",
            input={},
            route_decision=None,
            attempts=2,
            generated_sql_final="SELECT 1",
            attempt_logs=[
                AttemptLog(1, "", "bad", {"success": False, "accuracy": False, "issues": []}, None, 3.0, None, True),
                AttemptLog(2, "", "good", {"success": True, "accuracy": True, "issues": []}, None, 4.0, None, True),
            ],
            final_result={"success": True},
            metrics={
                "execution_success": True,
                "execution_accuracy": True,
                "candidate_accuracies": [False, True],
                "end_to_end_latency_ms": 10.0,
                "model_latency_ms": 7.0,
            },
            model_call_count=2,
        ),
        CaseRunLog(
            case_id="case-2",
            strategy="generate_validate_retry",
            input={},
            route_decision=None,
            attempts=1,
            generated_sql_final="SELECT 1",
            attempt_logs=[
                AttemptLog(1, "", "good", {"success": True, "accuracy": None, "issues": []}, None, 2.0, None, True)
            ],
            final_result={"success": True},
            metrics={
                "execution_success": True,
                "execution_accuracy": None,
                "candidate_accuracies": [False],
                "end_to_end_latency_ms": 4.0,
                "model_latency_ms": 2.0,
            },
            model_call_count=1,
        ),
    ]

    summary = aggregate_case_logs(logs)["generate_validate_retry"]

    assert summary["execution_success_rate"] == 1.0
    assert summary["execution_accuracy"] == 1.0
    assert summary["cost_model_calls"] == 1.5
    assert summary["recovery_rate"] == 1.0


def test_runner_integration_runs_all_strategies_and_writes_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    cases = [
        TestCase(
            id="count-users",
            natural_language_query="How many users?",
            expected_result=[[2]],
        ),
        TestCase(
            id="reuse-users",
            natural_language_query="List all users",
            expected_result=[["Alice"], ["Bob"]],
        ),
        TestCase(
            id="adapt-orders",
            natural_language_query="Show orders for Alice",
            expected_result=[[10]],
        ),
    ]
    catalog = RoutingCatalog(
        [
            CatalogEntry(
                id="reuse-users",
                route_type="reuse",
                match_rules=[MatchRule(type="keyword", keywords=["list", "users"], priority=10)],
                sql="SELECT name FROM users",
            ),
            CatalogEntry(
                id="adapt-orders",
                route_type="adapt",
                match_rules=[
                    MatchRule(type="keyword", keywords=["orders"]),
                    MatchRule(type="regex", pattern=r"for (?P<customer>[A-Za-z]+)", priority=20),
                ],
                template="SELECT amount FROM orders WHERE customer = '{customer}'",
                placeholders=["customer"],
            ),
        ]
    )
    runner = ExperimentRunner(
        model=FakeModel(),
        executor=SQLiteExecutor(db_path),
        retry_policy=RetryPolicy(max_attempts=3),
        router=RuleBasedRouter(catalog),
        output_dir=tmp_path / "out",
    )

    result = runner.run(cases, strategy="all")

    summary = result["summary_metrics"]
    assert set(summary) == {"generate_only", "generate_validate_retry", "routing"}
    assert summary["generate_only"]["execution_success_rate"] < 1.0
    assert summary["generate_validate_retry"]["execution_success_rate"] >= summary["generate_only"]["execution_success_rate"]
    assert summary["generate_validate_retry"]["recovery_rate"] > 0.0
    assert summary["routing"]["execution_success_rate"] >= (2 / 3)
    routing_logs = result["per_strategy_logs"]["routing"]
    reuse_log = next(log for log in routing_logs if log.case_id == "reuse-users")
    adapt_log = next(log for log in routing_logs if log.case_id == "adapt-orders")
    assert reuse_log.route_decision is not None
    assert reuse_log.route_decision["strategy"] == "reuse"
    assert adapt_log.route_decision is not None
    assert adapt_log.route_decision["strategy"] == "adapt"
    assert (tmp_path / "out" / "per_case_generate_only.json").exists()
    assert (tmp_path / "out" / "per_case_generate_validate_retry.json").exists()
    assert (tmp_path / "out" / "per_case_routing.json").exists()
    assert (tmp_path / "out" / "summary_metrics.json").exists()
    assert (tmp_path / "out" / "summary_metrics.csv").exists()
