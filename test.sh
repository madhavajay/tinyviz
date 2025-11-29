#!/bin/bash
set -e
export UV_VENV_CLEAR=1
uv venv .vevn-test
uv pip install -e ./
uv pip install pytest
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Run only pytest-compatible tests (not integration scripts)
uv run pytest
