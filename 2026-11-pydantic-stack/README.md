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

An underwriter's assistant for a portfolio of 300 properties in central Montpelier, Vermont, a town on the Winooski River that flooded in July 2023. It answers questions about one property or the whole portfolio and draws the answer on a map.

| Part | Where | What it shows |
|---|---|---|
| Geo risk API | `geo_service/` | Our own FastAPI service over local data: flood zone, elevation, distance to water and an aerial image per property. Nothing third-party is called during the demo. |
| Main agent | `flood_risk_example/agent.py`, `tools.py` | Pydantic AI with code mode from the harness. One portfolio question is one piece of model-written code making over a thousand API requests. |
| Decision model | `flood_risk_example/decision.py` | TypeSafe's Jev decides accept, refer or decline for each property, with a confidence, and falls back to a language model when unsure. |
| Vision agent | `flood_risk_example/vision.py` | A multimodal model reads the aerial image into structured observations. |
| Web UI | `flood_risk_example/app.py`, `static/index.html` | Chat on the left, map on the right. The map draws only our own data, so it needs no basemap tiles. |
| Evals | `flood_risk_example/evals_offline.py`, `evals_online.py` | A labelled decision dataset run on two models, and evaluators on every live run. |

It needs two keys in `.env`, which the `make` targets below load for you: `TYPESAFE_API_KEY`, and a Logfire API key as `LOGFIRE_API_KEY`. The targets also use it as `LOGFIRE_TOKEN` and `PYDANTIC_AI_GATEWAY_API_KEY`, so the one key covers tracing, Agent Sandboxes and the Gateway, so everything lands in one Logfire project. Without a TypeSafe key, set `DECISION_MODEL` and `JUDGE_MODEL` to a language model such as `gateway/anthropic:claude-sonnet-5-5`.

### Session replay

The web UI can also trace the page and record a session replay, so the Logfire trace starts at the click in the browser. In the Logfire project open **Frontend → Session replays → Set up**, create a frontend application, and add its two values to `.env`:

```bash
export LOGFIRE_FRONTEND_TOKEN=...
export LOGFIRE_FRONTEND_BASE_URL=https://logfire-eu.pydantic.dev
```

Without them the page sends nothing. Session replay is in Early Access, so the organization needs it enabled. The demo records every session with text, inputs and images unmasked, since all the data is invented or public. The map is drawn on a canvas, which replay does not record, so it is blank in playback.

The browser SDK is bundled into `flood_risk_example/static/vendor/logfire/` from `flood_risk_example/browser/telemetry.js`. `make browser-telemetry` rebuilds it.

### Running order

`make run` starts everything: the geo risk API on port 8001 and the web UI. Each `make` target below starts the geo risk API on port 8001 alongside it and stops it afterwards, so one terminal is enough. The web UI is at http://localhost:8000.

| # | Minutes | Command | Show |
|---|---|---|---|
| 1 | 8 | `make app` | Ask about one property. Walk through `agent.py`, `decision.py` and `vision.py`, then the trace in Logfire: the HTTP request, the agent, the vision call, the decision with its confidence, and the calls to the geo risk API. |
| 2 | 6 | (same app) | Agent Sandboxes, where the agent's code always runs. Triage the whole portfolio, then ask the what-if. The page counts the API requests as they happen; the trace has one `run_code` with hundreds of calls under it. |
| 3 | 5 | `make evals` | Offline evals. 60 borderline properties with a known correct outcome, on the decision model and then on a language model. Compare accuracy, time and cost in the terminal and in Logfire. `make evals MODELS="..."` compares any list of models. |
| 4 | 6 | `make app-online-evals` | Online evals. Ask a question, then find the three evaluator results on the live trace. |

For a quick run without the web UI, `make agent` asks the agent one question in the terminal and prints the answer, and `make agent Q="Which street has the most referrals?"` asks your own. It runs `flood_risk_example/agent.py` as a module (`uv run -m flood_risk_example.agent`).

### Questions

- Should we insure 34 State St? Take the aerial image into account. (Decline: it is in the regulatory floodway. About 15 seconds.)
- Triage the whole portfolio and show me the declines. (The guide gives 37 declines; a run returns slightly fewer, because the decision model misses a few. 300 decisions and about 1,200 API requests.)
- What changes if the flood zones grow by 50 m? (Declines rise sharply, because the floodway grows too. About 2,400 API requests.)
- Which street has the most referrals? Look at the aerial images for that street.

`make evals-portfolio` runs a second offline dataset: four portfolio questions whose answers are computed from the data.

### Data

`make build-data` downloads and preprocesses everything into `geo_service/data/`, which is committed, so it only needs running to change the town or the number of properties.

The 300 aerial images are about 10 MB and are not committed. `make images` downloads them into `geo_service/data/images/`, and every target that starts the geo risk API runs it first, so the first run needs the network and takes a little longer.

| Data | Source |
|---|---|
| Address points | Vermont E911 site locations, Vermont Open Geodata Portal |
| Flood zones | FEMA National Flood Hazard Layer, from Esri's hosted copy |
| Rivers and brooks | USGS National Hydrography Dataset |
| Elevation | USGS 3DEP |
| Aerial imagery | USDA NAIP, from the USGS image service |

The underwriting guide and every decision are invented for the demo. They say nothing about the real insurability of these addresses.

### Deploying the geo risk API

`geo_service/Dockerfile` builds the service on its own. Point the demo at it with `GEO_SERVICE_URL`.
