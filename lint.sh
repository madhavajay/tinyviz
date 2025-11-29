#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_VENV_CLEAR=1
uv venv
uv pip install -e .
uv pip install pytest ruff mypy vulture

echo "Running ruff format..."
uv run ruff format src

echo "Running ruff check with fixes..."
uv run ruff check src --fix

echo "Running mypy..."
uv run mypy src/tinyviz

echo "Running vulture to detect dead code..."
uv run vulture src --min-confidence 80

echo "✓ All linting checks passed!"
