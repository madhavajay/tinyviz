#!/bin/bash
uv venv --clear
source .venv/bin/activate
uv pip install -U cleon jupyter
uv pip install -e ./
jupyter lab
