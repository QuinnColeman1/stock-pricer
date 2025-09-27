
# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := poetry run python
POETRY := poetry run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

format: ## Format code with Ruff
	@echo "🛠️  Formatting code..."
	$(POETRY) ruff format stock_pricer/ tests/

lint-fix: ## Automatically fix linting issues where possible
	@echo "🔧 Fixing linting issues..."
	$(POETRY) ruff check --fix --exit-zero stock_pricer/ tests/

lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	$(POETRY) ruff check stock_pricer/ tests/

check-types: ## Run static type checking with mypy
	@echo "🔎 Running type checks..."
	$(POETRY) mypy stock_pricer/


test: ## Run tests with coverage
	@echo "\n🔍 Running tests..."
	poetry run pytest tests/ -v --cov=stock_pricer --cov-report=term-missing --cov-report=html


security: ## Run security checks with bandit
	@echo "🔒 Running security checks..."
	$(POETRY) bandit -r stock_pricer/

update: ## Update all dependencies
	poetry update --no-cache


clean: ## Remove all cache and build files
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -f coverage.xml
	rm -f *.cover
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	rm -rf .ruff_cache/
	rm -rf .pylint.d/
	find . -name '.DS_Store' -exec rm -f {} \;


all: format lint-fix check-types test security update clean
	@echo "✅ All tests passed successfully! 🎉"