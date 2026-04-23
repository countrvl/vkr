"""SQLite schema serialization helpers."""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def serialize_schema(db_path: Path) -> str:
    """Serialize SQLite tables as CREATE TABLE statements.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        CREATE TABLE statements separated by blank lines.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    query = """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY name
    """
    try:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(query).fetchall()
    except sqlite3.DatabaseError as exc:
        LOGGER.warning("Failed to read SQLite schema from %s: %s", db_path, exc)
        return ""

    statements = []
    for (sql,) in rows:
        if sql is None:
            continue
        statements.append(str(sql).strip())
    return "\n\n".join(statements)
