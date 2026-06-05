import sqlite3
from dataclasses import dataclass, field
from typing import Any

import logfire

from .synthetic_data import seed_database


@dataclass
class Database:
    """In-memory SQLite database for a single eval case.

    `check_same_thread=False` is required because pydantic-ai dispatches
    sync tool functions via run_in_executor (thread-pool workers), while
    the connection itself is created on the main async thread. Each eval
    gets its own isolated Database instance, so there is no concurrent
    write risk.
    """

    _conn: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(':memory:', check_same_thread=False))

    def __post_init__(self) -> None:
        self._conn.row_factory = sqlite3.Row
        seed_database(self)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL statement and return rows as a list of dicts.

        Supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, etc.
        SELECTs return rows as `[{column: value}, ...]`.
        Writes return `[{"rows_affected": N}]`.
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            self._conn.commit()
            return [{'rows_affected': cursor.rowcount}]
        except Exception as e:
            message = f'{type(e).__name__}: {e}'
            # "no such table" / "no such column" is the fingerprint of a schema
            # the system prompt describes wrongly (see the migration note on
            # `seed_database`). We surface those as error-level EXCEPTION spans
            # so they're visible (red) in the trace, countable on dashboards
            # (is_exception), and strong *failure* evidence for the optimizer
            # (its candidate query treats level>=error / is_exception as a
            # failure; `failure_reason` becomes the one-line summary). The
            # agent still recovers from the returned error dict, so the
            # conversation transcript shows both the break and the fix.
            #
            # Incidental SQL errors (an agent typo, etc.) stay at WARNING so the
            # post-fix "after" state shows ~zero errors.
            lowered = str(e).lower()
            schema_mismatch = 'no such table' in lowered or 'no such column' in lowered
            if schema_mismatch:
                # Record a real EXCEPTION span: let SchemaMismatchError escape a
                # logfire.span (so Logfire marks it is_exception=true / red in
                # the trace and it's countable as an error), then swallow it
                # here. The agent never sees the exception — it gets the error
                # dict below and recovers exactly as before.
                try:
                    with logfire.span(
                        'SQL failed against the live schema: {message}',
                        message=message,
                        sql=sql,
                        _level='error',
                        failure_reason=f"{message} — the prompt's documented schema is stale",
                        likely_cause='schema_mismatch',
                    ):
                        raise SchemaMismatchError(message)
                except SchemaMismatchError:
                    pass
            else:
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

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert multiple rows into `table`. `rows` is a list of column-value dicts."""
        if not rows:
            return {'inserted': 0}
        columns = list(rows[0].keys())
        placeholders = ', '.join(['?'] * len(columns))
        col_str = ', '.join(columns)
        sql = f'INSERT INTO {table} ({col_str}) VALUES ({placeholders})'
        try:
            self._conn.executemany(sql, [tuple(r.get(c) for c in columns) for r in rows])
            self._conn.commit()
            return {'inserted': len(rows)}
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}

    def table_count(self, table: str) -> int:
        """Return the row count of `table` (or -1 on error)."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            return cursor.fetchone()[0]
        except Exception:
            return -1


class SchemaMismatchError(ValueError):
    """Raised and immediately caught inside `query` to record an *exception*
    span when SQL hits the migrated-away schema.

    It never propagates into the sandbox — letting it escape a `logfire.span`
    is just how we get Logfire to mark that span `is_exception=true` (so the
    failure shows as a red error in the trace and is countable on dashboards),
    while the agent still recovers from the returned error dict exactly as
    before.
    """
