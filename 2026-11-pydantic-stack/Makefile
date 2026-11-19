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

.PHONY: ai-trade-db
ai-trade-db: ## Build the DuckDB database for the demo agent
	uv run ai_trade_example/build_db.py

.PHONY: ai-trade
ai-trade: ## Demo 1: the agent REPL, code running in a local Monty process
	uv run -m ai_trade_example

.PHONY: ai-trade-sandbox
ai-trade-sandbox: ## Demo 2: the same REPL, code running in Logfire Agent Sandboxes
	uv run -m ai_trade_example --sandbox

.PHONY: evals
evals: ## Demo 3: offline evals against the dataset of known answers
	uv run -m ai_trade_example.evals_offline

.PHONY: ai-trade-online-evals
ai-trade-online-evals: ## Demo 4: the REPL on Agent Sandboxes with online evals on every run
	uv run -m ai_trade_example --sandbox --online-evals
