# The Pydantic Stack

November 2026

A one-hour technical introduction to the Pydantic stack: 20 minutes of slides, a 25 minute demo, 15 minutes of questions.

See [deck-v1.pdf](deck-v1.pdf) for slides.

```bash
make install  # deck and Python dependencies
make          # build deck-v1.html and deck-v1.pdf
make dev      # deck with live reload
```

## Demo

One agent carried through four stages. It is a Pydantic AI agent using code mode (`CodeMode` from the harness, running on Monty) over a DuckDB database of prices and 13F filings, investigating the July 2026 AI selloff and the collapse of Situational Awareness LP. Press coverage in `ai_trade_example/docs/` is mounted read-only into the sandbox at `/docs`, and charts are written to `charts/`.

It needs `PYDANTIC_AI_GATEWAY_API_KEY`, `LOGFIRE_TOKEN` and, for Agent Sandboxes, `LOGFIRE_API_KEY` with the `project:execute_agent_sandboxes` scope.

Build the database once with `make ai-trade-db`. Downloads are cached in `ai_trade_example/raw_data/`, so it works offline.

| # | Minutes | Command | Show |
|---|---|---|---|
| 1 | 8 | `make ai-trade` | Pydantic AI, Logfire and the harness. Walk through `agent.py`, ask the first two questions below, then open the trace in Logfire. |
| 2 | 5 | `make ai-trade-sandbox` | Agent Sandboxes. The one-line `monty_sandbox=logfire.agent_sandboxes` change in `agent.py`, the same question, then the sandbox spans in the trace. |
| 3 | 6 | `make evals` | Offline evals. `evals_offline.py`: four cases with known answers, a code check and an LLM judge. The report prints in the terminal and the experiment is in Logfire. |
| 4 | 6 | `make ai-trade-online-evals` | Online evals. `evals_online.py`: the same agent with `OnlineEvaluation` added. Ask a question, then find the evaluator results on the live trace. |

Questions, with the answers to expect:

- The press says the fund made 439% in the first half of 2026. Does the long book in its March filing, at the leverage the press reported, account for that? (Yes: the March longs gained 110.8% by 30 June, which at the reported 4x is 443%. About 15 seconds.)
- Chart July returns by AI theme. (One bar chart: neoclouds and memory down 25 to 35%, software up 21%. About 10 seconds.)
- Which hurt the fund more in July, Sandisk or SK Hynix? (Sandisk, -47% vs -30%. About 6 seconds.)
- Was the Adobe short a hedge in July? (No, Adobe rose 22% while the book fell 33%. About 10 seconds.)

Longer questions that produce a full write-up and a chart, if there is time:

- Reconstruct what the fund's Q2 book did day by day through July. Which positions did the damage? Chart the cumulative P&L with the fund events annotated.
- Was that a fund-specific blow-up or the whole AI trade unwinding? Compare the theme baskets.
