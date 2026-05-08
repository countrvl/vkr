"""Prepare and validate the synthetic e-commerce SQL benchmark."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from nl2sql.src.strategy_bench.executor import SQLiteExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "nl2sql" / "synthetic_ecommerce"
DATASET_PATH = DATA_ROOT / "dataset_v1.json"
EDGE_CASES_PATH = DATA_ROOT / "edge_cases_v1.json"
SEED_SQL_PATH = DATA_ROOT / "seed.sql"
DB_PATH = DATA_ROOT / "snapshot.sqlite"
COVERAGE_PATH = DATA_ROOT / "coverage_summary.json"


@dataclass(slots=True)
class SyntheticQuery:
    sql: str
    difficulty: str
    group: str
    edge_type: str | None = None


@dataclass(slots=True)
class SyntheticValidationIssue:
    query_index: int
    message: str


_FROM_JOIN_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)
_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", flags=re.IGNORECASE)
_WHERE_RE = re.compile(r"\bWHERE\b", flags=re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORDER\s+BY\b", flags=re.IGNORECASE)
_GROUP_RE = re.compile(r"\bGROUP\s+BY\b", flags=re.IGNORECASE)
_HAVING_RE = re.compile(r"\bHAVING\b", flags=re.IGNORECASE)
_COUNT_RE = re.compile(r"\bCOUNT\s*\(", flags=re.IGNORECASE)
_SUM_RE = re.compile(r"\bSUM\s*\(", flags=re.IGNORECASE)
_AVG_RE = re.compile(r"\bAVG\s*\(", flags=re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\b", flags=re.IGNORECASE)
_EXISTS_RE = re.compile(r"\b(?:NOT\s+)?EXISTS\b", flags=re.IGNORECASE)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_core_queries(path: Path = DATASET_PATH) -> list[SyntheticQuery]:
    payload = load_json(path)
    return [
        SyntheticQuery(sql=row["sql"], difficulty=row["difficulty"], group="core")
        for row in payload["queries"]
    ]


def load_edge_queries(path: Path = EDGE_CASES_PATH) -> list[SyntheticQuery]:
    payload = load_json(path)
    return [
        SyntheticQuery(
            sql=row["sql"],
            difficulty=row.get("difficulty", "edge_case"),
            group=row.get("group", "edge_case"),
            edge_type=row.get("edge_type"),
        )
        for row in payload["queries"]
    ]


def build_snapshot_db(
    *,
    output_path: Path = DB_PATH,
    dataset_path: Path = DATASET_PATH,
    seed_path: Path = SEED_SQL_PATH,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    dataset = load_json(dataset_path)
    connection = sqlite3.connect(output_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for table in dataset["schema"]["tables"]:
            connection.execute(f"CREATE TABLE {table}")
        connection.executescript(seed_path.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()
    return output_path


def _table_columns(dataset: dict[str, Any]) -> dict[str, list[str]]:
    def split_columns(raw: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in raw:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            if char == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(char)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    table_columns: dict[str, list[str]] = {}
    for table_def in dataset["schema"]["tables"]:
        table_name, raw_cols = table_def.split("(", 1)
        columns: list[str] = []
        for part in split_columns(raw_cols.rstrip(")")):
            chunk = part.strip()
            lowered = chunk.lower()
            if lowered.startswith("foreign key"):
                continue
            column_name = chunk.split()[0]
            columns.append(column_name)
        table_columns[table_name.strip()] = columns
    return table_columns


def _query_aliases(sql: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table_name, alias in _FROM_JOIN_ALIAS_RE.findall(sql):
        aliases[alias] = table_name
    return aliases


def _query_tables(sql: str) -> list[str]:
    return list(dict.fromkeys(_TABLE_RE.findall(sql)))


def _classify_intents(sql: str) -> list[str]:
    intents: list[str] = []
    if _WHERE_RE.search(sql):
        intents.append("filtering")
    if _ORDER_RE.search(sql) and _LIMIT_RE.search(sql):
        intents.append("ranking")
    if _COUNT_RE.search(sql) or _SUM_RE.search(sql) or _AVG_RE.search(sql):
        intents.append("aggregation")
    if _GROUP_RE.search(sql):
        intents.append("segmentation")
    if _HAVING_RE.search(sql) or _EXISTS_RE.search(sql):
        intents.append("comparison")
    return intents or ["filtering"]


def _classify_selectivity(sql: str) -> str:
    if _WHERE_RE.search(sql) or _LIMIT_RE.search(sql) or _EXISTS_RE.search(sql):
        return "selective"
    return "non_selective"


def _result_distribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    zero_rows = sum(1 for row in results if row["rows_returned"] == 0)
    one_row = sum(1 for row in results if row["rows_returned"] == 1)
    multi_rows = sum(1 for row in results if row["rows_returned"] > 1)
    total = len(results)
    zero_or_one_ratio = ((zero_rows + one_row) / total) if total else 0.0
    return {
        "zero_rows_queries": zero_rows,
        "one_row_queries": one_row,
        "multi_row_queries": multi_rows,
        "zero_or_one_ratio": round(zero_or_one_ratio, 4),
        "is_skewed_to_zero_or_one": zero_or_one_ratio >= 0.5,
    }


def _column_usage(sql: str, table_columns: dict[str, list[str]]) -> dict[str, set[str]]:
    usage: dict[str, set[str]] = defaultdict(set)
    aliases = _query_aliases(sql)
    for alias, table_name in aliases.items():
        for column_name in table_columns.get(table_name, []):
            if re.search(rf"\b{re.escape(alias)}\.{re.escape(column_name)}\b", sql):
                usage[table_name].add(column_name)

    tables = _query_tables(sql)
    if len(tables) == 1:
        table_name = tables[0]
        for column_name in table_columns.get(table_name, []):
            if re.search(rf"\b{re.escape(column_name)}\b", sql):
                usage[table_name].add(column_name)
    return usage


def _duplicate_signature(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    normalized = re.sub(r"'[^']*'", "'?'", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "?", normalized)
    return normalized


def _row_count_summary(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        counts = {}
        for table_name in ("customers", "categories", "products", "orders", "order_items", "returns"):
            counts[table_name] = int(cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        return counts
    finally:
        connection.close()


def _data_quality_summary(db_path: Path, table_columns: dict[str, list[str]]) -> dict[str, Any]:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        null_ratios: dict[str, float] = {}
        row_counts = _row_count_summary(db_path)
        for table_name, columns in table_columns.items():
            table_rows = row_counts[table_name]
            for column_name in columns:
                null_count = int(
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
                    ).fetchone()[0]
                )
                ratio = (null_count / table_rows) if table_rows else 0.0
                null_ratios[f"{table_name}.{column_name}"] = round(ratio, 4)

        unusual_value_checks = {
            "products.brand:Brand-X/Legacy": int(
                cursor.execute(
                    "SELECT COUNT(*) FROM products WHERE brand = 'Brand-X/Legacy'"
                ).fetchone()[0]
            ),
            "orders.payment_method:crypto": int(
                cursor.execute(
                    "SELECT COUNT(*) FROM orders WHERE payment_method = 'crypto'"
                ).fetchone()[0]
            ),
            "returns.return_reason:empty_string": int(
                cursor.execute(
                    "SELECT COUNT(*) FROM returns WHERE return_reason = ''"
                ).fetchone()[0]
            ),
        }

        null_ratio_bands = {
            "gte_0.10": sorted(column for column, ratio in null_ratios.items() if ratio >= 0.10),
            "between_0.05_and_0.10": sorted(
                column for column, ratio in null_ratios.items() if 0.05 <= ratio < 0.10
            ),
            "gt_0_and_lt_0.05": sorted(
                column for column, ratio in null_ratios.items() if 0.0 < ratio < 0.05
            ),
        }
        return {
            "null_ratio_by_column": dict(sorted(null_ratios.items())),
            "null_ratio_bands": null_ratio_bands,
            "unusual_value_checks": unusual_value_checks,
        }
    finally:
        connection.close()


def validate_queries(
    *,
    db_path: Path = DB_PATH,
    dataset_path: Path = DATASET_PATH,
    edge_cases_path: Path = EDGE_CASES_PATH,
) -> tuple[list[SyntheticValidationIssue], dict[str, Any]]:
    dataset = load_json(dataset_path)
    table_columns = _table_columns(dataset)
    queries = load_core_queries(dataset_path) + load_edge_queries(edge_cases_path)
    executor = SQLiteExecutor(db_path)

    issues: list[SyntheticValidationIssue] = []
    table_usage = Counter()
    column_usage = Counter()
    intent_usage = Counter()
    selectivity_usage = Counter()
    duplicate_signatures = Counter()
    edge_type_distribution = Counter()
    results: list[dict[str, Any]] = []

    for index, query in enumerate(queries, start=1):
        syntax_issue = executor.explain_syntax(query.sql)
        if syntax_issue is not None:
            issues.append(SyntheticValidationIssue(index, syntax_issue.message))
            continue
        outcome = executor.execute(query.sql)
        if not outcome.success:
            issues.append(SyntheticValidationIssue(index, outcome.error_message or "Execution failed"))
            continue
        query_tables = _query_tables(query.sql)
        if len(query_tables) > 1 and " AS " not in query.sql.upper():
            issues.append(SyntheticValidationIssue(index, "JOIN queries must use aliases"))
        for table_name in query_tables:
            table_usage[table_name] += 1
        usage = _column_usage(query.sql, table_columns)
        for table_name, columns in usage.items():
            for column_name in columns:
                column_usage[f"{table_name}.{column_name}"] += 1
        for intent in _classify_intents(query.sql):
            intent_usage[intent] += 1
        selectivity_usage[_classify_selectivity(query.sql)] += 1
        if query.edge_type:
            edge_type_distribution[query.edge_type] += 1
        duplicate_signatures[_duplicate_signature(query.sql)] += 1
        results.append(
            {
                "index": index,
                "group": query.group,
                "difficulty": query.difficulty,
                "edge_type": query.edge_type,
                "rows_returned": len(outcome.rows or []),
                "execution_time_ms": round(outcome.latency_ms, 4),
                "tables": query_tables,
                "intents": _classify_intents(query.sql),
                "selectivity": _classify_selectivity(query.sql),
            }
        )

    unused_tables = sorted(set(table_columns) - set(table_usage))
    all_columns = {f"{table}.{column}" for table, columns in table_columns.items() for column in columns}
    used_columns = set(column_usage)
    unused_columns = sorted(all_columns - used_columns)
    low_usage_columns = sorted(column for column, count in column_usage.items() if count == 1)
    duplicate_like_queries = [
        {"signature": signature, "count": count}
        for signature, count in duplicate_signatures.items()
        if count > 1
    ]

    summary = {
        "row_counts": _row_count_summary(db_path),
        "query_counts": {
            "core": len(load_core_queries(dataset_path)),
            "edge_case": len(load_edge_queries(edge_cases_path)),
            "total": len(queries),
        },
        "data_quality_summary": _data_quality_summary(db_path, table_columns),
        "table_usage": dict(sorted(table_usage.items())),
        "column_usage": dict(sorted(column_usage.items())),
        "unused_tables": unused_tables,
        "unused_columns": unused_columns,
        "low_usage_columns": low_usage_columns,
        "intent_distribution": dict(sorted(intent_usage.items())),
        "selectivity_distribution": dict(sorted(selectivity_usage.items())),
        "edge_type_distribution": dict(sorted(edge_type_distribution.items())),
        "result_distribution": _result_distribution(results),
        "duplicate_like_queries": duplicate_like_queries,
        "query_results": results,
    }
    return issues, summary


def write_coverage_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
