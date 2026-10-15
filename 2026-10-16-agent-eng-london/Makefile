.DEFAULT_GOAL := pdf
VERSION := 1

.PHONY: install
install: ## Install dependencies
	cd deck && bun install

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
