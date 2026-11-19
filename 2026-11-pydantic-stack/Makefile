.DEFAULT_GOAL := pdf
VERSION := 1

.PHONY: install
install: ## Install dependencies
	cd deck && bun install
	uv sync --frozen

.PHONY: dev
dev: ## Start the dev server with live reload
	cd deck && bunx deckx dev

.PHONY: typecheck
typecheck: ## Typecheck the deck components
	cd deck && bun run tsc --noEmit

.PHONY: html
html: ## Build the deck to deck-v$(VERSION).html
	rm -f deck-v*
	cd deck && bunx deckx html ../deck-v$(VERSION).html
	@echo "Slides written to deck-v$(VERSION).html"

.PHONY: pdf
pdf: html ## Build the deck to deck-v$(VERSION).pdf
	cd deck && bunx deckx html-to-pdf ../deck-v$(VERSION).html ../deck-v$(VERSION).pdf
	@echo "PDF written to deck-v$(VERSION).pdf"

.PHONY: build-data
build-data: ## Download and preprocess the open data (one-off; the output is committed)
	uv run -m geo_service.build_data

# Loads the keys from .env if it exists (the Logfire API key is also the write token and the Gateway key,
# and has to override any LOGFIRE_TOKEN already in the shell, which may point at another project), then starts the geo risk API in the background for the life of the command that follows, and stops it
# afterwards. If `make serve` is already running, the second copy fails to bind and the first is used.
WITH_GEO_SERVICE = if [ -f .env ]; then . ./.env; export LOGFIRE_TOKEN=$$LOGFIRE_API_KEY PYDANTIC_AI_GATEWAY_API_KEY=$$LOGFIRE_API_KEY; fi; \
	uv run uvicorn geo_service.app:app --port 8001 --log-level warning & \
	GEO_PID=$$!; trap "kill $$GEO_PID 2>/dev/null" EXIT INT TERM; sleep 2;

.PHONY: serve
serve: ## Run the geo risk API on its own on port 8001
	uv run uvicorn geo_service.app:app --port 8001

.PHONY: run
run: app ## Run the whole demo: the geo risk API and the web UI on http://localhost:8000

.PHONY: app
app: ## Demos 1 and 2: the geo risk API and the web UI on http://localhost:8000
	@$(WITH_GEO_SERVICE) uv run uvicorn flood_risk_example.app:app --port 8000

.PHONY: agent
agent: ## Ask the agent one question in the terminal, e.g. make agent Q="Which street has the most referrals?"
	@$(WITH_GEO_SERVICE) uv run -m flood_risk_example.agent $(Q)

.PHONY: evals
evals: ## Demo 3: offline evals of the decision model against a language model
	@$(WITH_GEO_SERVICE) uv run -m flood_risk_example.evals_offline decisions $(MODELS)

.PHONY: evals-portfolio
evals-portfolio: ## Offline evals of the main agent on portfolio questions
	@$(WITH_GEO_SERVICE) uv run -m flood_risk_example.evals_offline portfolio

.PHONY: app-online-evals
app-online-evals: ## Demo 4: the same, with online evals on every run
	@$(WITH_GEO_SERVICE) FLOOD_ONLINE_EVALS=1 uv run uvicorn flood_risk_example.app:app --port 8000

.PHONY: browser-telemetry
browser-telemetry: ## Rebuild the Logfire browser SDK and session replay bundle (the output is committed)
	cd flood_risk_example/browser && bun install && bun run build

.PHONY: test
test: ## Test the geo risk API
	uv run pytest geo_service -q

.PHONY: requests
requests: ## Send a few questions to the running web app, for recent traces and online evals (N=3 to send fewer)
	uv run -m flood_risk_example.send_requests $(N)
