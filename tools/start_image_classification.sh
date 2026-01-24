#!/bin/bash
# Startet das Image Classification Tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
PYTHON_BIN="python3"

cd "$PROJECT_ROOT"
$PYTHON_BIN streamlit run tools/image_classification_tool.py "$@"
