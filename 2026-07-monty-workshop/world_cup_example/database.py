import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logfire

# The pre-built database lives alongside this package (built by build_wc2026_db.py).
DB_PATH = Path(__file__).parent / 'worldcup2026.db'


def _connect() -> sqlite3.Connection:
    """Open the World Cup database read-only.

    The agent only ever *analyses* this data, so we open in `mode=ro` — any
    accidental write is rejected by SQLite rather than mutating the file.

    `check_same_thread=False` is required because pydantic-ai dispatches sync
    tool functions via run_in_executor (thread-pool workers) while the
    connection is created on the main async thread.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f'{DB_PATH} not found — run `uv run world_cup_example/build_wc2026_db.py` first.'
        )
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class Database:
    """Read-only SQLite access to the 2026 World Cup database."""

    _conn: sqlite3.Connection = field(default_factory=_connect)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query and return rows as a list of dicts.

        SELECTs return rows as `[{column: value}, ...]`. The database is opened
        read-only, so writes will come back as an `{"error": ...}` dict.
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            return [{'rows_affected': cursor.rowcount}]
        except Exception as e:
            message = f'{type(e).__name__}: {e}'
            logfire.warning('SQL query failed: {message}', message=message, sql=sql)
            return [{'error': message}]

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def describe_table(self, name: str) -> list[dict[str, str | bool]]:
        """Describe a table's schema (column name, type, nullable, primary key)."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(f'PRAGMA table_info({name})')
            return [
                {
                    'name': row[1],
                    'type': row[2],
                    'nullable': not row[3],
                    'primary_key': bool(row[5]),
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            return [{'error': f'{type(e).__name__}: {e}'}]

    def table_count(self, table: str) -> int:
        """Return the row count of `table` (or -1 on error)."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            return cursor.fetchone()[0]
        except Exception:
            return -1
