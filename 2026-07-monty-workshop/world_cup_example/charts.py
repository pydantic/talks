from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

# Charts are written to ./charts at the repo root (created on first use).
CHARTS_DIR = Path(__file__).parent.parent / 'charts'


async def draw_chart(
    x: list[float],
    y: list[float],
    *,
    name: str,
    kind: str = 'line',
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    label: str | None = None,
) -> str:
    """Draw a chart, save it to ./charts, and open it.

    Args:
        x: X-axis data.
        y: Y-axis data.
        name: File name for the chart (`.png` is appended if missing).
        kind: Plot type — 'line', 'bar', or 'scatter'.
        title: Optional title for the chart.
        x_label: Optional label for the x-axis.
        y_label: Optional label for the y-axis.
        label: Optional legend label for the series.

    Returns:
        The file path where the chart was saved.
    """
    fig, ax = plt.subplots()
    if kind == 'bar':
        ax.bar(x, y, label=label)
    elif kind == 'scatter':
        ax.scatter(x, y, label=label)
    else:
        ax.plot(x, y, label=label)

    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if label:
        ax.legend()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = name.replace('/', '_').replace('\\', '_')
    if not file_name.endswith('.png'):
        file_name = f'{file_name}.png'
    save_path = CHARTS_DIR / file_name

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f'Saved chart to {save_path}')
    subprocess.run(['open', str(save_path)])

    return f'Chart saved to {save_path}'
