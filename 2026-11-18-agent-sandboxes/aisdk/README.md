# Code mode with the Vercel AI SDK

A cut-down version of the
[logfire-js example](https://github.com/pydantic/logfire-js/tree/main/examples/agent-sandboxes-ai-sdk).

[`index.ts`](index.ts) defines two ordinary AI SDK tools, `query(sql)` over an in-memory SQLite database (created in [`db.ts`](db.ts)) and
`chart(data)`, which prints a bar chart. `codeMode()` gives the model a single `run_code` tool instead, and the Python
it writes calls `query` and `chart` from inside Logfire Agent Sandboxes.

Needs Node 24 or later. Set `LOGFIRE_API_KEY` and `OPENAI_API_KEY` in `.env`, then:

```bash
pnpm install
pnpm run start
```
