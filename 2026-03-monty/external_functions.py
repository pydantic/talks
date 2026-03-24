from __future__ import annotations

import datetime
import subprocess
from typing import Any

import duckdb
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

matplotlib.use('Agg')

_conn = duckdb.connect(':memory:')

_pypi_data = _conn.read_parquet('pypi_downloads.parquet')

_console = Console()


async def sql_query(sql: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute SQL query on the PyPI downloads data using DuckDB.

    You may query on the `pypi_data` table.

    Args:
        sql: SQL query to execute. The data is available as a table named 'pypi_data'.
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
    return [dict(zip(columns, (_convert(v) for v in row))) for row in rows]


def _convert(v: Any) -> Any:
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    return v


async def plot(
    x: list[float],
    y: list[float],
    *,
    kind: str = 'line',
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    label: str | None = None,
) -> None:
    """Add a plot to the current figure.

    Args:
        x: X-axis data.
        y: Y-axis data.
        kind: Plot type — 'line', 'bar', or 'scatter'.
        title: Optional title for the plot.
        x_label: Optional label for the x-axis.
        y_label: Optional label for the y-axis.
        label: Optional legend label for this series.
    """
    if kind == 'bar':
        plt.bar(x, y, label=label)  # pyright: ignore[reportUnknownMemberType]
    elif kind == 'scatter':
        plt.scatter(x, y, label=label)  # pyright: ignore[reportUnknownMemberType]
    else:
        plt.plot(x, y, label=label)  # pyright: ignore[reportUnknownMemberType]

    if title:
        plt.title(title)  # pyright: ignore[reportUnknownMemberType]
    if x_label:
        plt.xlabel(x_label)  # pyright: ignore[reportUnknownMemberType]
    if y_label:
        plt.ylabel(y_label)  # pyright: ignore[reportUnknownMemberType]
    if label:
        plt.legend()  # pyright: ignore[reportUnknownMemberType]


async def show_plot(plot_name: str) -> str:
    """Display the current figure and reset the plot state.

    Args:
        plot_name: Optional name for the plot

    Returns:
        The file path where the figure was saved.
    """
    save_path = f'charts/{plot_name.replace("/", "_").replace("\\", "_")}'
    if not save_path.endswith('.png'):
        save_path = f'{save_path}.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)  # pyright: ignore[reportUnknownMemberType]
    plt.close('all')
    print(f'Saved plot to {save_path}')
    subprocess.run(['open', save_path])

    return f'File saved to {save_path}'


async def display_table(headers: list[str], rows: list[list[str]], *, title: str | None = None) -> str:
    """Print a formatted table to the console.

    Args:
        headers: Column header names.
        rows: List of rows, where each row is a list of string values.
        title: Optional title for the table.

    Returns:
        A JSON string representation of the displayed table data.
    """
    import json

    table = Table(title=title)
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*[str(v) for v in row])
    _console.print(table)

    data = [dict(zip(headers, row)) for row in rows]
    return f'TABLE DISPLAYED: {json.dumps(data)}'
