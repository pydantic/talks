# Background documents

Plain-text background on the 2026 AI trade and the collapse of Situational Awareness LP, distilled from
press coverage (every file ends with its sources). Read them from the sandbox with `pathlib`, e.g.

```python
from pathlib import Path
print(Path('/docs/unwind_timeline.md').read_text())
```

| File | What it covers |
|---|---|
| `fund_background.md` | Who Leopold Aschenbrenner is, how the fund started, backers, growth, strategy, returns |
| `positions_and_leverage.md` | What the fund owned and shorted, how much leverage it ran, what a 13F does and doesn't show |
| `july_2026_selloff.md` | The market backdrop: the memory boom and bust, the chip selloff, winners and losers |
| `unwind_timeline.md` | Day-by-day sequence of the margin calls, the Citadel block trade and the rebound |
| `sources.md` | The articles the above were drawn from |

These are summaries of reporting, not primary data. Where outlets disagree on a figure, both numbers are
given. The database (`fund_holdings`, `prices`, `fund_events`) is the primary source for anything numeric;
use the documents for context, framing and the parts of the story the data can't show (private holdings,
prime brokers, the sequence of events inside the fund).
