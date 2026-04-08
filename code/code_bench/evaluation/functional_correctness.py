"""Execution-based проверка корректности для HumanEval+ и MBPP+."""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any

from evalplus.eval import PASS, TIMEOUT, MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth

from code_bench.data.prepare import get_benchmark_hash, load_evalplus_tasks, normalize_benchmark_name


def _dataset_name_for_evalplus(benchmark: str) -> str:
    benchmark = normalize_benchmark_name(benchmark)
    return "humaneval" if benchmark == "humaneval_plus" else "mbpp"


@lru_cache(maxsize=8)
def load_problems(
    benchmark: str,
    *,
    mini: bool = False,
    noextreme: bool = False,
) -> dict[str, dict[str, Any]]:
    """Загрузить полные задачи EvalPlus для benchmark-а."""
    return load_evalplus_tasks(benchmark, mini=mini, noextreme=noextreme)


@lru_cache(maxsize=8)
def load_expected_outputs(
    benchmark: str,
    *,
    mini: bool = False,
    noextreme: bool = False,
) -> dict[str, dict[str, Any]]:
    """Загрузить или вычислить эталонные outputs для benchmark-а."""
    problems = load_problems(benchmark, mini=mini, noextreme=noextreme)
    dataset_hash = get_benchmark_hash(benchmark, mini=mini, noextreme=noextreme)
    tasks_only_output_not_none = [] if benchmark == "humaneval_plus" else MBPP_OUTPUT_NOT_NONE_TASKS
    return get_groundtruth(problems, dataset_hash, tasks_only_output_not_none)


def normalize_solution_code(code: str, problem: dict[str, Any], benchmark: str) -> str:
    """Нормализовать извлеченный код в формат, ожидаемый EvalPlus."""
    cleaned = (code or "").strip()
    if not cleaned:
        return ""

    entry_point = problem["entry_point"]
    if f"def {entry_point}" in cleaned:
        return cleaned

    if benchmark == "humaneval_plus":
        body = cleaned
        if not body.startswith((" ", "\t")):
            body = "\n".join(
                f"    {line}" if line.strip() else line
                for line in body.splitlines()
            )
        return f"{problem['prompt'].rstrip()}\n{body.rstrip()}\n"

    return cleaned


def evaluate_code_candidate(
    *,
    benchmark: str,
    task_id: str,
    candidate_index: int,
    code: str,
    execution_cfg: dict[str, Any],
    mini: bool = False,
    noextreme: bool = False,
) -> dict[str, Any]:
    """Оценить один кандидат кода на тестах EvalPlus."""
    problems = load_problems(benchmark, mini=mini, noextreme=noextreme)
    expected_outputs = load_expected_outputs(benchmark, mini=mini, noextreme=noextreme)
    problem = problems[task_id]
    normalized_code = normalize_solution_code(code, problem, benchmark)

    if not normalized_code.strip():
        return {
            "normalized_code": "",
            "compiled_ok": False,
            "tests_passed": False,
            "functional_correctness": False,
            "error_type": "empty_code",
            "base_status": "fail",
            "plus_status": "fail",
            "base_passed": 0,
            "base_total": len(problem.get("base_input", [])),
            "plus_passed": 0,
            "plus_total": len(problem.get("plus_input", [])),
        }

    try:
        ast.parse(normalized_code)
        compile(normalized_code, "<generated>", "exec")
    except SyntaxError:
        return {
            "normalized_code": normalized_code,
            "compiled_ok": False,
            "tests_passed": False,
            "functional_correctness": False,
            "error_type": "syntax_error",
            "base_status": "fail",
            "plus_status": "fail",
            "base_passed": 0,
            "base_total": len(problem.get("base_input", [])),
            "plus_passed": 0,
            "plus_total": len(problem.get("plus_input", [])),
        }

    result = check_correctness(
        _dataset_name_for_evalplus(benchmark),
        candidate_index,
        problem,
        normalized_code,
        expected_outputs[task_id],
        fast_check=bool(execution_cfg.get("fast_check", False)),
        min_time_limit=float(execution_cfg.get("min_time_limit", 1.0)),
        gt_time_limit_factor=float(execution_cfg.get("gt_time_limit_factor", 4.0)),
    )
    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]
    tests_passed = base_status == PASS and plus_status == PASS
    if tests_passed:
        error_type = ""
    elif TIMEOUT in (base_status, plus_status):
        error_type = "timeout"
    else:
        error_type = "failed_tests"

    return {
        "normalized_code": normalized_code,
        "compiled_ok": True,
        "tests_passed": tests_passed,
        "functional_correctness": tests_passed,
        "error_type": error_type,
        "base_status": base_status,
        "plus_status": plus_status,
        "base_passed": sum(bool(item) for item in base_details),
        "base_total": len(problem.get("base_input", [])),
        "plus_passed": sum(bool(item) for item in plus_details),
        "plus_total": len(problem.get("plus_input", [])),
    }
