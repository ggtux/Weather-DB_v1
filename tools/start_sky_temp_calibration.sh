#!/bin/bash
# Startet das Sky Temperature Calibration Tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

cd "$PROJECT_ROOT"
streamlit run tools/sky_temp_calibration.py
