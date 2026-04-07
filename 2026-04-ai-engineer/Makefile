.DEFAULT_GOAL := main

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: install
install: .uv ## Install python dependencies
	# --only-dev to avoid building the python package, use make dev for that
	uv sync --all-packages --only-dev

.PHONY: format
format: ## Format Python code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint Python code with ruff
	uv run ruff format --check
	uv run ruff check
	uv run basedpyright

.PHONY: main
main: format lint ## run formatting and linting

# (must stay last!)
.PHONY: help
help: ## Show this help (usage: make help)
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
	    helpMessage = match($$0, /## (.*)/); \
	        if (helpMessage) { \
	            recipe = $$1; \
	            sub(/:/, "", recipe); \
	            printf "  \033[36mmake %-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
	    } \
	}' $(MAKEFILE_LIST)
