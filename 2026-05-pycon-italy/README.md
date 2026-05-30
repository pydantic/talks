# PyCon Italia

Bologna, 2026-04-29

* [Talk schedule](https://2026.pycon.it/en/keynotes/duable-agents-long-running-ai-workflows-in-a-flakey-world)
* [Deck PDF](deck-v1.pdf)
* [Pydantic AI + temporal docs](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/)

---

## `twenty_questions_temporal_mv.py` — Managed Variable Demo

This variant of the 20 Questions demo drives the questioner's system prompt from
a [Logfire Managed Variable](https://logfire.pydantic.dev/docs/guides/managed-variables/)
so the prompt can be optimized live via the Logfire Prompt Optimizer without
restarting the process.

### Prerequisites

* [Temporal CLI](https://docs.temporal.io/cli) installed (`brew install temporal`)
* A Logfire project (local dev server or cloud) with a **write token** (`pylf_v2_…`)
* `uv` for running the project

### Environment variables

Create a `.env` file (or export these in your shell):

```
# Logfire write token — used for both trace export and the variables REST API
LOGFIRE_TOKEN=pylf_v1_local_…        # v1 token (trace export)
LOGFIRE_API_KEY=pylf_v2_local_…      # v2 token (variables API + trace export)

# Point at your Logfire instance (omit for cloud)
LOGFIRE_BASE_URL=http://localhost:3000

# LLM credentials (example uses MiniMax Anthropic-compatible endpoint)
ANTHROPIC_API_KEY=…
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
```

`LOGFIRE_TOKEN` and `LOGFIRE_API_KEY` can be the same v2 token — the code maps
`LOGFIRE_TOKEN` → `LOGFIRE_API_KEY` automatically if only one is set.

### First-time setup: seed the managed variable

Before the first run you need to register the variable definition with Logfire
and add an initial prompt version:

**1. Push the variable schema**

```bash
uv run python -c "
import os, logfire
logfire.configure(variables=logfire.VariablesOptions())
from twenty_questions_temporal_mv import _questioner_var
logfire.variables_push(yes=True)
print('Variable pushed.')
"
```

This creates the variable in Logfire (name, schema, description). You only need
to do this once, or whenever the variable definition changes.

**2. Set the initial value in Logfire**

Open your Logfire project → **Managed Variables** tab, find
`questioner_prompt_twenty_questions_temporal`, and add a version with the prompt
text you want to use as the starting point.

### Running the demo

Start Temporal, then run the game:

```bash
temporal server start-dev --headless &
uv run python twenty_questions_temporal_mv.py
```

### Using the Prompt Optimizer

1. Run several games to accumulate spans in Logfire.
2. In the Logfire UI open your project → **Managed Variables** →
   `questioner_prompt_twenty_questions_temporal`.
3. Click **Optimize** → **Generate Proposal**.
4. Review the proposed prompt changes and publish a new version.
5. The next game run will automatically pick up the new version — no restart
   needed.

#### Known issues with the Prompt Optimizer

* **Large span count**: The optimizer may throw an error if there are a very
  large number of spans associated with the variable. If this happens, reduce
  the time window or number of runs before generating a proposal.
* **"Nothing to optimize"**: The optimizer may reject a proposal if it doesn't
  detect a meaningful improvement opportunity. This is expected when the current
  prompt is already performing well — try running more games with a deliberately
  weak prompt first, or adjust the optimizer settings.

Both of these are waiting on the next prod deployment as of 05/30 to resolve.