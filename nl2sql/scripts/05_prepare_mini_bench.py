"""Prepare the local NL2SQL mini-benchmark snapshot and validate the dataset."""

from __future__ import annotations

import argparse

from nl2sql.src.mini_bench.prepare import prepare_mini_bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local NL2SQL mini-benchmark.")
    parser.add_argument("--force", action="store_true", help="Rebuild the SQLite snapshot even if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path, cases, issues = prepare_mini_bench(force=args.force)
    if issues:
        for issue in issues:
            print(f"[ERROR] {issue.case_id}: {issue.message}")
        raise SystemExit(1)
    print(f"SQLite snapshot ready: {db_path}")
    print(f"Validated {len(cases)} mini-benchmark cases")


if __name__ == "__main__":
    main()
