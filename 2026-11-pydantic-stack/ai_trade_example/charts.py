# matplotlib ships no type stubs, so member types are unknown under strict pyright.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import matplotlib
from rich import get_console

matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Charts are written to ./charts at the repo root (created on first use).
CHARTS_DIR = Path(__file__).parent.parent / 'charts'

_ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


async def draw_chart(
    x: list[str],
    series: dict[str, list[float]],
    *,
    name: str,
    kind: str = 'line',
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    annotations: dict[str, str] | None = None,
    log_scale: bool = False,
) -> str:
    """Draw a chart with one or more series and save it to ./charts as a PNG.

    Args:
        x: X-axis values, shared by every series. ISO dates (`'2026-07-30'`) are plotted on a date axis;
            anything else is treated as category labels.
        series: Mapping of legend label to y-values; every list must have the same length as `x`.
        name: File name for the chart (`.png` is appended if missing).
        kind: Plot type — 'line', 'bar', or 'scatter'. Bars with several series are grouped side by side.
        title: Optional title for the chart.
        x_label: Optional label for the x-axis.
        y_label: Optional label for the y-axis.
        annotations: Optional mapping of x value to a short label, drawn as labelled vertical lines
            (e.g. dated fund events on a price chart).
        log_scale: Use a logarithmic y-axis.

    Returns:
        The file path where the chart was saved.
    """
    if not series:
        return 'Error: `series` is empty'
    for label, values in series.items():
        if len(values) != len(x):
            return f'Error: series {label!r} has {len(values)} values but x has {len(x)}'

    is_dates = bool(x) and all(_ISO_DATE.match(v) for v in x)
    # matplotlib wants numeric x positions: date numbers for a date axis, 0..n-1 for categories.
    if is_dates:
        positions = [float(mdates.date2num(date.fromisoformat(v))) for v in x]
    else:
        positions = [float(i) for i in range(len(x))]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if kind == 'bar':
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            offset = (i - (len(series) - 1) / 2) * width
            ax.bar([p + offset for p in positions], values, width=width, label=label)
    else:
        for label, values in series.items():
            if kind == 'scatter':
                ax.scatter(positions, values, label=label)
            else:
                ax.plot(positions, values, label=label)

    if is_dates and kind != 'bar':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        fig.autofmt_xdate()
    else:
        ax.set_xticks(positions)
        rotate = len(x) > 6
        ax.set_xticklabels(x, rotation=45 if rotate else 0, ha='right' if rotate else 'center')

    if annotations and kind != 'bar':
        for x_value, text in annotations.items():
            if is_dates and _ISO_DATE.match(x_value):
                position = float(mdates.date2num(date.fromisoformat(x_value)))
            elif x_value in x:
                position = float(x.index(x_value))
            else:
                continue
            ax.axvline(position, color='grey', linestyle='--', linewidth=0.8)
            ax.annotate(
                text,
                xy=(position, 1.0),
                xycoords=('data', 'axes fraction'),
                rotation=90,
                va='top',
                ha='right',
                fontsize=7,
                color='grey',
            )

    if log_scale:
        ax.set_yscale('log')
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if len(series) > 1 or list(series) != [name]:
        ax.legend()
    ax.grid(True, alpha=0.3)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = name.replace('/', '_').replace('\\', '_')
    if not file_name.endswith('.png'):
        file_name = f'{file_name}.png'
    save_path = CHARTS_DIR / file_name

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # Go through rich's global console (shared with the REPL's live display) so the line prints cleanly
    # above the streaming answer instead of corrupting it.
    try:
        shown = save_path.relative_to(Path.cwd())
    except ValueError:
        shown = save_path
    get_console().print(f'[dim]Chart written to {shown}[/dim]')

    return f'Chart saved to {save_path}'
