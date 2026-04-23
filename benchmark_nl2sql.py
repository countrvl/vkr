"""Run a simplified NL2SQL benchmark on the synthetic e-commerce snapshot."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv

from nl2sql.src.data.loader import DataSample
from nl2sql.src.evaluation.pass_at_k import pass_at_k
from nl2sql.src.data.schema import serialize_schema
from nl2sql.src.inference.base import GenerationResult
from nl2sql.src.inference.runtime import EXPERIMENT_CONFIG, build_backend
from nl2sql.src.prompt.template import PromptBuilder
from shared.evaluation.statistics import bootstrap_quantile_fields, wilson_interval
from shared.config import load_domain_models
from shared.logging_utils import configure_logging, create_progress


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "nl2sql" / "synthetic_ecommerce" / "dataset_v1.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nl2sql" / "synthetic_ecommerce" / "snapshot.sqlite"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "nl2sql" / "synthetic_benchmark"
DEFAULT_SEED = int(EXPERIMENT_CONFIG.get("seed", 42))
DEFAULT_PASS_K = max(int(value) for value in EXPERIMENT_CONFIG.get("k_values", [5]))
DEFAULT_TEMPERATURE_PASS_K = float(EXPERIMENT_CONFIG.get("temperature_pass_k", 0.8))
DEFAULT_TOP_P = EXPERIMENT_CONFIG.get("top_p")
DEFAULT_MAX_TOKENS = int(EXPERIMENT_CONFIG.get("max_tokens", 512))

_WHERE_EQUALITY_RE = re.compile(
    r"\bWHERE\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('([^']*)'|[0-9]+(?:\.[0-9]+)?)",
    flags=re.IGNORECASE,
)
_ORDER_LIMIT_RE = re.compile(
    r"\bORDER\s+BY\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*(ASC|DESC)?\s+LIMIT\s+([0-9]+)",
    flags=re.IGNORECASE,
)
_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)", flags=re.IGNORECASE)
_SQL_FROM_RE = re.compile(
    r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    flags=re.IGNORECASE,
)
_SQL_JOIN_RE = re.compile(
    r"\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    flags=re.IGNORECASE,
)
_AGGREGATE_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\((.*?)\)", flags=re.IGNORECASE | re.DOTALL)
_HAS_AGGREGATE_RE = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", flags=re.IGNORECASE)
_HAS_JOIN_RE = re.compile(r"\bJOIN\b", flags=re.IGNORECASE)
_HAS_GROUP_RE = re.compile(r"\bGROUP\s+BY\b", flags=re.IGNORECASE)
_HAS_WHERE_RE = re.compile(r"\bWHERE\b", flags=re.IGNORECASE)
_HAS_HAVING_RE = re.compile(r"\bHAVING\b", flags=re.IGNORECASE)


@dataclass(slots=True)
class QueryRecord:
    sql: str
    difficulty: str


@dataclass(slots=True)
class InferenceContext:
    model_key: str | None
    mode: str
    seed: int
    k: int
    stub_mode: str
    prompt_builder: PromptBuilder
    db_path: Path = DEFAULT_DB_PATH
    backend: Any | None = None
    model_name: str = "stub"
    prompt_profile: str = "nl2sql_json"
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float | None = DEFAULT_TOP_P
    oracle_sql_by_question: dict[str, str] | None = None


@dataclass(slots=True)
class SqlExecution:
    success: bool
    rows: list[tuple[str, ...]] | None
    error: str | None = None


@dataclass(slots=True)
class CandidateEvaluation:
    sql: str
    valid: bool
    execution_match: bool
    error: str | None = None


_INFERENCE_CONTEXT = InferenceContext(
    model_key=None,
    mode="ea",
    seed=DEFAULT_SEED,
    k=1,
    stub_mode="heuristic",
    prompt_builder=PromptBuilder(),
    db_path=DEFAULT_DB_PATH,
)


def load_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    schema = payload.get("schema")
    queries = payload.get("queries")
    if not isinstance(schema, dict) or not isinstance(schema.get("tables"), list):
        raise ValueError(f"Invalid dataset schema in {dataset_path}")
    if not isinstance(queries, list):
        raise ValueError(f"Invalid queries section in {dataset_path}")

    normalized_queries: list[QueryRecord] = []
    for index, row in enumerate(queries):
        if not isinstance(row, dict):
            raise ValueError(f"Query row {index} must be an object")
        sql = row.get("sql")
        difficulty = row.get("difficulty")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"Query row {index} has invalid sql")
        if not isinstance(difficulty, str) or not difficulty.strip():
            raise ValueError(f"Query row {index} has invalid difficulty")
        normalized_queries.append(QueryRecord(sql=sql.strip(), difficulty=difficulty.strip()))

    return {
        "schema_tables": [str(table).strip() for table in schema["tables"]],
        "queries": normalized_queries,
    }


def build_database(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite snapshot not found: {db_path}")
    try:
        connection = sqlite3.connect(db_path)
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"Failed to open SQLite snapshot {db_path}: {exc}") from exc
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
    return connection


def generate_data() -> dict[str, Any]:
    return {
        "mode": "existing_snapshot",
        "generated": False,
        "message": "Using existing synthetic_ecommerce snapshot.sqlite without regenerating data.",
    }


def generate_nl(queries: list[QueryRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for query in queries:
        sql = query.sql.strip().rstrip(";")
        question = _sql_to_question(sql)
        rows.append(
            {
                "question": question,
                "sql": query.sql,
                "difficulty": query.difficulty,
            }
        )
    return rows


def generate_sql(question: str, schema: str) -> str:
    if _INFERENCE_CONTEXT.backend is None:
        return _generate_stub_sql(question=question, schema=schema, stub_mode=_INFERENCE_CONTEXT.stub_mode)
    return asyncio.run(_generate_sql_async(question=question, schema=schema))


async def run_inference(
    nl_queries: list[dict[str, str]],
    schema: str,
    *,
    db_path: Path,
    raw_output_path: Path | None = None,
    model_key: str | None,
    mode: str,
    k: int,
    seed: int,
    stub_mode: str,
) -> list[dict[str, Any]]:
    prompt_builder = PromptBuilder()
    oracle_sql = {row["question"]: row["sql"] for row in nl_queries} if stub_mode == "oracle" else None
    temperature, n = _resolve_mode_params(mode=mode, k=k)
    backend = None
    model_name = "stub"
    max_tokens = DEFAULT_MAX_TOKENS
    prompt_profile = "nl2sql_json"

    if model_key is not None:
        models_cfg = load_domain_models("supports_sql")
        if model_key not in models_cfg:
            available = ", ".join(sorted(models_cfg))
            raise ValueError(f"Unknown model {model_key!r}. Available: {available}")
        model_cfg = models_cfg[model_key]
        backend = build_backend(model_key, model_cfg)
        model_name = str(model_cfg.get("display_name") or model_cfg["name"])
        max_tokens = int(model_cfg.get("max_tokens", DEFAULT_MAX_TOKENS))
        prompt_profile = str(model_cfg.get("prompt_profile", "nl2sql_json"))

    global _INFERENCE_CONTEXT
    _INFERENCE_CONTEXT = InferenceContext(
        model_key=model_key,
        mode=mode,
        seed=seed,
        k=n,
        stub_mode=stub_mode,
        prompt_builder=prompt_builder,
        db_path=db_path,
        backend=backend,
        model_name=model_name,
        prompt_profile=prompt_profile,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=DEFAULT_TOP_P,
        oracle_sql_by_question=oracle_sql,
    )

    rows: list[dict[str, Any]] = []
    raw_handle = None
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = raw_output_path.open("w", encoding="utf-8")
    with create_progress() as progress:
        task_id = progress.add_task(
            "Inference",
            total=len(nl_queries),
            status=f"{model_name}/{mode}",
        )
        for item in nl_queries:
            generations: list[GenerationResult] = []
            sql_pred = ""
            inference_error: str | None = None
            try:
                generations = await _generate_candidates_async(
                    question=item["question"],
                    schema=schema,
                    n=n,
                )
                sql_pred = generations[0].sql if generations else ""
            except Exception as exc:
                inference_error = str(exc)
                LOGGER.warning("Inference failed for question %r: %s", item["question"], exc)

            record = {
                "question": item["question"],
                "sql_gt": item["sql"],
                "sql_pred": sql_pred,
                "difficulty": item["difficulty"],
                "model_name": model_name,
                "mode": mode,
                "valid": False,
                "execution_match": False,
                "inference_error": inference_error,
                "sql_candidates": [generation.sql for generation in generations],
            }
            rows.append(record)
            if raw_handle is not None:
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_handle.flush()
            progress.update(task_id, advance=1, status=f"{model_name}/{mode}")
    if raw_handle is not None:
        raw_handle.close()
    return rows


def evaluate(results: list[dict[str, Any]], connection: sqlite3.Connection, *, mode: str, k: int) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for row in results:
        gt_outcome = _execute_sql(connection, row["sql_gt"])
        candidate_evaluations = _evaluate_candidates(
            row.get("sql_candidates", []),
            gt_outcome,
            connection,
        )
        first_candidate = candidate_evaluations[0] if candidate_evaluations else CandidateEvaluation(
            sql=row["sql_pred"],
            valid=False,
            execution_match=False,
            error="empty query",
        )
        evaluated.append(
            {
                "question": row["question"],
                "sql_gt": row["sql_gt"],
                "sql_pred": first_candidate.sql,
                "valid": first_candidate.valid,
                "execution_match": first_candidate.execution_match,
                "difficulty": row["difficulty"],
                "inference_error": row.get("inference_error"),
                "prediction_error": first_candidate.error,
                "ground_truth_error": gt_outcome.error,
                "sql_candidates": row.get("sql_candidates", []),
                "candidate_results": [
                    {
                        "sql": candidate.sql,
                        "valid": candidate.valid,
                        "execution_match": candidate.execution_match,
                        "error": candidate.error,
                    }
                    for candidate in candidate_evaluations
                ],
            }
        )

    summary = {
        "overall": _compute_metrics(evaluated, mode=mode, k=k),
        "by_difficulty": {},
    }
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in evaluated if row["difficulty"] == difficulty]
        summary["by_difficulty"][difficulty] = _compute_metrics(subset, mode=mode, k=k)
    return {"summary": summary, "details": evaluated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified synthetic NL2SQL benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", help="Optional SQL-capable model key from shared/configs/models.yaml.")
    parser.add_argument("--mode", choices=["ea", "pass_k"], default="ea")
    parser.add_argument("--k", type=int, default=DEFAULT_PASS_K)
    parser.add_argument("--limit", type=int, help="Limit the number of evaluated queries.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stub-mode", choices=["heuristic", "select1", "oracle"], default="heuristic")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    connection = build_database(args.db_path)
    try:
        schema = serialize_schema(args.db_path)
        if not schema.strip():
            schema = "\n\n".join(f"CREATE TABLE {table};" for table in dataset["schema_tables"])
        generate_data()
        nl_queries = generate_nl(dataset["queries"])
        if args.limit is not None:
            nl_queries = nl_queries[: args.limit]
        inference_rows = asyncio.run(
            run_inference(
                nl_queries,
                schema,
                db_path=args.db_path,
                raw_output_path=args.output_dir / "raw_predictions.jsonl",
                model_key=args.model,
                mode=args.mode,
                k=args.k,
                seed=args.seed,
                stub_mode=args.stub_mode,
            )
        )
        evaluation = evaluate(inference_rows, connection, mode=args.mode, k=args.k)
    finally:
        connection.close()

    (args.output_dir / "results.json").write_text(
        json.dumps(evaluation["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "detailed_results.json").write_text(
        json.dumps(evaluation["details"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info(
        "Finished synthetic benchmark: EA=%.4f valid_sql_rate=%.4f total=%d",
        evaluation["summary"]["overall"]["execution_accuracy"],
        evaluation["summary"]["overall"]["valid_sql_rate"],
        len(evaluation["details"]),
    )


async def _generate_sql_async(question: str, schema: str) -> str:
    generations = await _generate_candidates_async(question=question, schema=schema, n=1)
    return generations[0].sql if generations else ""


async def _generate_candidates_async(question: str, schema: str, *, n: int) -> list[GenerationResult]:
    context = _INFERENCE_CONTEXT
    if context.backend is None:
        sql = _generate_stub_sql(question=question, schema=schema, stub_mode=context.stub_mode)
        return [
            GenerationResult(
                sql=sql,
                raw_response=sql,
                tokens_input=0,
                tokens_output=0,
                latency_ms=0.0,
                model_name=context.model_name,
                metadata={"backend": "stub", "mode": context.stub_mode},
            )
            for _ in range(n)
        ]

    sample = DataSample(
        id="synthetic_eval",
        benchmark="synthetic_ecommerce",
        question=question,
        gold_sql="",
        db_id="synthetic_ecommerce",
        db_path=context.db_path,
        schema=schema,
        difficulty=None,
        evidence=None,
    )
    prompt = context.prompt_builder.build(sample, context.prompt_profile)
    # Keep pass@k compatible with API backends that reject n > 1 by issuing
    # multiple single-sample calls. Ollama already samples sequentially itself,
    # but using one-call-per-candidate keeps behavior consistent across backends.
    if n > 1:
        all_generations: list[GenerationResult] = []
        for _ in range(n):
            all_generations.extend(
                await context.backend.generate(
                    prompt,
                    n=1,
                    temperature=context.temperature,
                    max_tokens=context.max_tokens,
                    seed=None,
                    top_p=context.top_p,
                )
            )
        return all_generations[:n]
    return await context.backend.generate(
        prompt,
        n=1,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed if context.mode == "ea" else None,
        top_p=context.top_p,
    )


def _generate_stub_sql(*, question: str, schema: str, stub_mode: str) -> str:
    if stub_mode == "oracle":
        oracle_sql = (_INFERENCE_CONTEXT.oracle_sql_by_question or {}).get(question)
        return oracle_sql or "SELECT 1"
    if stub_mode == "select1":
        return "SELECT 1"

    table_name = _first_table_name(schema) or "customers"
    lowered = question.lower()
    if "count" in lowered:
        return f"SELECT COUNT(*) FROM {table_name}"
    if "average" in lowered or "avg" in lowered:
        return f"SELECT AVG(rowid) FROM {table_name}"
    if "top" in lowered or "ordered" in lowered:
        return f"SELECT * FROM {table_name} LIMIT 5"
    return f"SELECT * FROM {table_name} LIMIT 10"


def _sql_to_question(sql: str) -> str:
    columns = _extract_selected_columns(sql)
    primary_table, _ = _extract_primary_table(sql)
    joined_tables = _extract_joined_tables(sql)
    all_tables = _unique_names([primary_table, *joined_tables])
    group_columns = _extract_group_columns(sql)
    aggregation = _extract_aggregation(sql)
    condition = _extract_condition(sql)
    has_join = bool(joined_tables)
    has_aggregation = bool(_HAS_AGGREGATE_RE.search(sql))
    has_group = bool(_HAS_GROUP_RE.search(sql))
    has_filter = bool(condition) or bool(_HAS_HAVING_RE.search(sql))
    table_phrase = _table_phrase(primary_table, all_tables)
    join_phrase = _join_phrase(all_tables)

    if re.search(r"\b(?:NOT\s+)?EXISTS\b", sql, flags=re.IGNORECASE):
        return _exists_question(sql, columns, primary_table)

    if has_aggregation and has_group and has_filter:
        metric = aggregation or "the metric"
        group = _format_list(group_columns) or "group"
        condition_text = condition or _extract_having_condition(sql) or "the filtered condition"
        source = join_phrase or table_phrase
        return f"for each {group}, compute {metric} {source} where {condition_text}"

    if has_aggregation and has_group:
        metric = aggregation or "the metric"
        group = _format_list(group_columns) or "group"
        source = join_phrase or table_phrase
        return f"for each {group}, compute {metric} {source}"

    if has_join:
        question = f"find {_format_list(columns)} {join_phrase}"
        if condition:
            question += f" where {condition}"
        order_text = _extract_order_phrase(sql)
        if order_text:
            question += f" {order_text}"
        return question

    question = f"find {_format_list(columns)} {table_phrase}"
    if condition:
        question += f" where {condition}"
    order_text = _extract_order_phrase(sql)
    if order_text:
        question += f" {order_text}"
    return question


def _extract_selected_columns(sql: str) -> list[str]:
    match = re.search(r"\bSELECT\b\s+(.*?)\s+\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return ["records"]
    raw_columns = match.group(1)
    if raw_columns.strip().upper().startswith("DISTINCT "):
        raw_columns = raw_columns.strip()[len("DISTINCT ") :]
    columns = [_format_expression(part) for part in _split_top_level(raw_columns)]
    return [column for column in columns if column] or ["records"]


def _extract_primary_table(sql: str) -> tuple[str, str | None]:
    match = _SQL_FROM_RE.search(sql)
    if match is None:
        return "records", None
    table, alias = match.groups()
    if alias and alias.upper() in {"WHERE", "JOIN", "GROUP", "ORDER", "HAVING", "LIMIT", "ON"}:
        alias = None
    return table, alias


def _extract_joined_tables(sql: str) -> list[str]:
    tables: list[str] = []
    for table, alias in _SQL_JOIN_RE.findall(sql):
        if table.upper() in {"SELECT", "WHERE"}:
            continue
        if alias and alias.upper() == "ON":
            alias = ""
        _ = alias
        tables.append(table)
    return tables


def _extract_group_columns(sql: str) -> list[str]:
    match = re.search(
        r"\bGROUP\s+BY\b\s+(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return []
    return [_format_expression(part) for part in _split_top_level(match.group(1))]


def _extract_aggregation(sql: str) -> str:
    match = _AGGREGATE_RE.search(sql)
    if match is None:
        return ""
    function_name, expression = match.groups()
    metric = _format_expression(expression)
    return f"{_aggregate_name(function_name)} of {metric}"


def _extract_condition(sql: str) -> str:
    match = re.search(
        r"\bWHERE\b\s+(.*?)(?:\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    return _format_condition(match.group(1))


def _extract_having_condition(sql: str) -> str:
    match = re.search(
        r"\bHAVING\b\s+(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    return _format_condition(match.group(1))


def _extract_order_phrase(sql: str) -> str:
    match = _ORDER_LIMIT_RE.search(sql)
    if match is None:
        return ""
    column, direction, limit = match.groups()
    direction_text = "descending" if (direction or "").upper() == "DESC" else "ascending"
    return f"ordered by {_format_expression(column)} {direction_text} limit {limit}"


def _exists_question(sql: str, columns: list[str], primary_table: str) -> str:
    entities = _format_list(columns)
    has_exists = bool(re.search(r"(?<!NOT\s)\bEXISTS\b", sql, flags=re.IGNORECASE))
    has_not_exists = bool(re.search(r"\bNOT\s+EXISTS\b", sql, flags=re.IGNORECASE))
    positive_relation = _exists_relation(sql, negative=False)
    negative_relation = _exists_relation(sql, negative=True)

    question = f"find {entities} from the {primary_table} table"
    if has_exists and has_not_exists:
        return f"{question} that have {positive_relation} and do not have {negative_relation}"
    if has_not_exists:
        return f"{question} that do not have {negative_relation}"
    return f"{question} that have {positive_relation}"


def _exists_relation(sql: str, *, negative: bool) -> str:
    pattern = r"\bNOT\s+EXISTS\s*\((.*?)\)" if negative else r"(?<!NOT\s)\bEXISTS\s*\((.*?)\)"
    match = re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return "related records"
    tables = _unique_names(_extract_joined_tables(match.group(1)) + [_extract_primary_table(match.group(1))[0]])
    if not tables or tables == ["records"]:
        return "related records"
    relation = " and ".join(table for table in tables if table != "records")
    if negative and "returns" in relation:
        return "returned order items"
    return relation


def _format_condition(condition: str) -> str:
    text = " ".join(condition.strip().rstrip(";").split())
    text = re.sub(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)", "", text)
    text = re.sub(r"'([^']*)'", r"\1", text)
    text = re.sub(r"\bBETWEEN\s+([^\s]+)\s+AND\s+([^\s)]+)", r"is between \1 and \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*>=\s*", " is at least ", text)
    text = re.sub(r"\s*<=\s*", " is at most ", text)
    text = re.sub(r"\s*<>\s*", " is not equal to ", text)
    text = re.sub(r"\s*>\s*", " is greater than ", text)
    text = re.sub(r"\s*<\s*", " is less than ", text)
    text = re.sub(r"\s*=\s*", " is ", text)
    text = re.sub(r"\bAND\b", "and", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOR\b", "or", text, flags=re.IGNORECASE)
    return text


def _format_expression(expression: str) -> str:
    text = " ".join(expression.strip().rstrip(";").split())
    text = re.sub(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)", "", text)
    text = re.sub(r"\bAS\b", "as", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDISTINCT\b", "distinct", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCOUNT\s*\((.*?)\)", r"count of \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSUM\s*\((.*?)\)", r"sum of \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAVG\s*\((.*?)\)", r"average of \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMIN\s*\((.*?)\)", r"minimum of \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMAX\s*\((.*?)\)", r"maximum of \1", text, flags=re.IGNORECASE)
    if text == "*":
        return "all records"
    text = text.replace("*", " times ")
    return " ".join(text.split())


def _split_top_level(raw: str) -> list[str]:
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


def _format_list(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return "records"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _unique_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _table_phrase(primary_table: str, all_tables: list[str]) -> str:
    if len(all_tables) > 1:
        return _join_phrase(all_tables)
    return f"from the {primary_table} table"


def _join_phrase(tables: list[str]) -> str:
    if len(tables) < 2:
        return ""
    return f"by joining {_format_list(tables)}"


def _aggregate_name(function_name: str) -> str:
    mapping = {
        "COUNT": "count",
        "SUM": "sum",
        "AVG": "average",
        "MIN": "minimum",
        "MAX": "maximum",
    }
    return mapping.get(function_name.upper(), function_name.lower())


def _first_table_name(schema: str) -> str | None:
    match = _TABLE_RE.search(schema)
    if match is not None:
        return match.group(1)
    return None


def _resolve_mode_params(*, mode: str, k: int) -> tuple[float, int]:
    if mode == "ea":
        return 0.0, 1
    return DEFAULT_TEMPERATURE_PASS_K, max(1, k)


def _normalize_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return sorted(normalized)


def _execute_sql(connection: sqlite3.Connection, sql: str) -> SqlExecution:
    if not sql or not sql.strip():
        return SqlExecution(success=False, rows=None, error="empty query")
    try:
        rows = connection.execute(sql).fetchall()
    except sqlite3.DatabaseError as exc:
        return SqlExecution(success=False, rows=None, error=str(exc))
    normalized = _normalize_rows([tuple(row) for row in rows])
    return SqlExecution(success=True, rows=normalized)


def _evaluate_candidates(
    candidate_sql: list[str],
    gold_outcome: SqlExecution,
    connection: sqlite3.Connection,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    for sql in candidate_sql:
        outcome = _execute_sql(connection, sql)
        evaluations.append(
            CandidateEvaluation(
                sql=sql,
                valid=outcome.success,
                execution_match=bool(
                    gold_outcome.success and outcome.success and outcome.rows == gold_outcome.rows
                ),
                error=outcome.error,
            )
        )
    return evaluations


def _compute_metrics(rows: list[dict[str, Any]], *, mode: str, k: int) -> dict[str, float | None]:
    if not rows:
        empty: dict[str, float | None] = {
            "execution_accuracy": 0.0,
            "valid_sql_rate": 0.0,
        }
        if mode == "pass_k":
            for pass_k_value in _resolve_pass_k_values(k):
                empty[f"pass@{pass_k_value}"] = 0.0
                empty[f"pass@{pass_k_value}_ci_low"] = None
                empty[f"pass@{pass_k_value}_ci_high"] = None
                empty[f"pass@{pass_k_value}_q05"] = None
                empty[f"pass@{pass_k_value}_q50"] = None
                empty[f"pass@{pass_k_value}_q95"] = None
        return empty
    total = len(rows)
    metrics: dict[str, float | None] = {
        "execution_accuracy": sum(1 for row in rows if row["execution_match"]) / total,
        "valid_sql_rate": sum(1 for row in rows if row["valid"]) / total,
    }
    if mode == "pass_k":
        candidate_hits = [
            [bool(candidate["execution_match"]) for candidate in row.get("candidate_results", [])]
            for row in rows
        ]
        for pass_k_value in _resolve_pass_k_values(k):
            metrics[f"pass@{pass_k_value}"] = pass_at_k(candidate_hits, pass_k_value)
            ci_low, ci_high = wilson_interval(
                successes=sum(1 for hits in candidate_hits if any(hits[: min(pass_k_value, len(hits))])),
                total=len(candidate_hits),
            )
            metrics[f"pass@{pass_k_value}_ci_low"] = ci_low
            metrics[f"pass@{pass_k_value}_ci_high"] = ci_high
            metrics.update(
                bootstrap_quantile_fields(
                    candidate_hits,
                    lambda sample, pass_k_value=pass_k_value: pass_at_k(list(sample), pass_k_value),
                    prefix=f"pass@{pass_k_value}",
                    quantiles=(0.05, 0.5, 0.95),
                    n_resamples=200,
                    seed=DEFAULT_SEED + pass_k_value,
                )
            )
    return metrics


def _resolve_pass_k_values(k: int) -> list[int]:
    configured = [int(value) for value in EXPERIMENT_CONFIG.get("k_values", [1, 3, 5])]
    resolved = {1, max(1, k)}
    resolved.update(value for value in configured if value <= max(1, k))
    return sorted(resolved)


if __name__ == "__main__":
    main()
