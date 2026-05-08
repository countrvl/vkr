"""Validate the synthetic e-commerce benchmark and write coverage summary."""

from __future__ import annotations

from nl2sql.src.synthetic_bench.prepare import (
    COVERAGE_PATH,
    DB_PATH,
    build_snapshot_db,
    validate_queries,
    write_coverage_summary,
)


def main() -> int:
    build_snapshot_db(output_path=DB_PATH, force=False)
    issues, summary = validate_queries(db_path=DB_PATH)
    write_coverage_summary(COVERAGE_PATH, summary)
    if issues:
        for issue in issues:
            print(f"[query {issue.query_index}] {issue.message}")
        return 1
    print("Synthetic benchmark validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
