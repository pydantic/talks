.DEFAULT_GOAL := pdf
VERSION := 1

.PHONY: install
install: ## Install dependencies
	cd deck && bun install
	uv sync --frozen

.PHONY: format
format: ## Format Python code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint Python code with ruff and basedpyright (strict)
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck
typecheck: ## Run basedpyright over the Python code
	uv run basedpyright

.PHONY: dev
dev: ## Start the dev server with live reload
	cd deck && bunx deckx dev

.PHONY: html
html: ## Build the deck to deck-v$(VERSION).html
	rm -f deck-v*
	cd deck && bun run typecheck && bunx deckx html ../deck-v$(VERSION).html
	@echo "Slides written to deck-v$(VERSION).html"

.PHONY: pdf
pdf: html ## Build the deck to deck-v$(VERSION).pdf
	cd deck && bunx deckx html-to-pdf ../deck-v$(VERSION).html ../deck-v$(VERSION).pdf
	@echo "PDF written to deck-v$(VERSION).pdf"
