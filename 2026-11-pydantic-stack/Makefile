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

.PHONY: serve
serve: ## Run the geo risk API on port 8001; every demo needs it
	uv run uvicorn geo_service.app:app --port 8001

.PHONY: app
app: ## Demo 1: the web UI on port 8000, code running in a local Monty process
	uv run uvicorn flood_risk_example.app:app --port 8000

.PHONY: app-sandbox
app-sandbox: ## Demo 2: the web UI, code running in Logfire Agent Sandboxes
	FLOOD_SANDBOX=1 uv run uvicorn flood_risk_example.app:app --port 8000

.PHONY: evals
evals: ## Demo 3: offline evals of the decision model against a language model
	uv run -m flood_risk_example.evals_offline decisions

.PHONY: evals-portfolio
evals-portfolio: ## Offline evals of the main agent on portfolio questions
	uv run -m flood_risk_example.evals_offline portfolio

.PHONY: app-online-evals
app-online-evals: ## Demo 4: the web UI on Agent Sandboxes with online evals on every run
	FLOOD_SANDBOX=1 FLOOD_ONLINE_EVALS=1 uv run uvicorn flood_risk_example.app:app --port 8000

.PHONY: test
test: ## Test the geo risk API
	uv run pytest geo_service -q
