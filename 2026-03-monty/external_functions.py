from __future__ import annotations

from typing import Any

import duckdb

_conn = duckdb.connect(':memory:')

_pypi_data = _conn.read_csv('pypi_downloads.parquet')


async def sql_query(sql: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute SQL query on a CSV file using DuckDB.

    Args:
        sql: SQL query to execute. The CSV data is available as a table named 'data'.
        parameters: Optional dictionary of parameters to bind to the SQL query.

    Returns:
        List of dictionaries, one per row, with column names as keys.
    """

    # NOTE! duckdb (horribly) reads locals as tables, hence `pypi_data` here that isn't used
    pypi_data = _pypi_data
    # Execute the user's query
    result_rel = _conn.execute(sql, parameters)
    del pypi_data
    # Get column names and rows, then convert to list of dicts
    columns = [desc[0] for desc in result_rel.description]
    rows = result_rel.fetchall()
    return [dict(zip(columns, row)) for row in rows]
