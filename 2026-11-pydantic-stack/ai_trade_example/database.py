from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import logfire

DB_PATH = Path(__file__).parent / 'ai_trade.duckdb'


def _connect() -> duckdb.DuckDBPyConnection:
    """Open the AI trade database read-only.

    The agent only ever *analyses* this data, so the file is opened `read_only`: any accidental write is
    rejected by DuckDB rather than mutating the file.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f'{DB_PATH} not found — run `uv run ai_trade_example/build_db.py` first.')
    return duckdb.connect(str(DB_PATH), read_only=True)


def _convert(value: Any) -> Any:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


@dataclass
class Database:
    """Read-only DuckDB access to the AI trade database.

    A DuckDB connection is not thread-safe and pydantic-ai dispatches sync tool functions via
    `run_in_executor` (thread-pool workers), so every call takes a per-thread cursor from the shared
    connection instead of using it directly.
    """

    _conn: duckdb.DuckDBPyConnection = field(default_factory=_connect)

    def _run(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        try:
            result = cursor.execute(sql, params or [])
            columns = [col[0] for col in result.description]
            return [dict(zip(columns, (_convert(v) for v in row), strict=False)) for row in result.fetchall()]
        finally:
            cursor.close()

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query (DuckDB dialect) and return rows as a list of dicts.

        SELECTs return rows as `[{column: value}, ...]`; dates come back as ISO strings. The database is opened
        read-only, so writes and SQL errors come back as a single `{"error": ...}` row rather than raising.
        """
        try:
            return self._run(sql)
        except Exception as e:
            message = f'{type(e).__name__}: {e}'
            logfire.warning('SQL query failed: {message}', message=message, sql=sql)
            return [{'error': message}]

    def list_tables(self) -> list[str]:
        """List all tables and views in the database."""
        rows = self._run(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        )
        return [row['table_name'] for row in rows]

    def describe_table(self, name: str) -> list[dict[str, Any]]:
        """Describe a table's schema (column name, type, nullable, primary key)."""
        try:
            rows = self._run(f'DESCRIBE {name}')
        except Exception as e:
            return [{'error': f'{type(e).__name__}: {e}'}]
        return [
            {
                'name': row['column_name'],
                'type': row['column_type'],
                'nullable': row['null'] == 'YES',
                'primary_key': row['key'] == 'PRI',
            }
            for row in rows
        ]

    def table_count(self, table: str) -> int:
        """Return the row count of `table` (or -1 on error)."""
        try:
            rows = self._run(f'SELECT count(*) AS n FROM {table}')
        except Exception:
            return -1
        return int(rows[0]['n'])
