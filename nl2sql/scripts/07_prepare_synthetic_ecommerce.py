"""Build the synthetic e-commerce SQLite snapshot."""

from nl2sql.src.synthetic_bench.prepare import build_snapshot_db


if __name__ == "__main__":
    build_snapshot_db(force=True)
