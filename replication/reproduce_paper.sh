#!/bin/bash
################################################################################
# Reproduce All Figures from fastLPR JSS Paper
#
# This script automates the reproduction of all figures (Fig 2-6) from the
# Journal of Statistical Software (JSS) submission for the fastLPR package.
#
# Usage:
#   bash reproduce_paper.sh
#
# Prerequisites:
#   - Python 3.9+ installed
#   - uv installed (https://docs.astral.sh/uv/)
#
# Output:
#   - Figures saved to fastLPR_py/fig/reproduced/
#   - Summary report printed to console
#   - Log file saved to replication/reproduction_log.txt
#
# Copyright (c) 2019-2026 Ying Wang, Min Li
# License: GPL-3.0-or-later
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output (ANSI escape codes)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_PKG_DIR="$PROJECT_ROOT/fastLPR_py"

echo ""
echo "================================================================================"
echo "Reproducing All Figures from fastLPR JSS Paper"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if fastLPR package directory exists
if [ ! -d "$PYTHON_PKG_DIR" ]; then
    echo -e "${RED}ERROR: fastLPR_py directory not found at $PYTHON_PKG_DIR${NC}"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo -e "${RED}ERROR: uv not found. Install uv first:${NC}"
    echo "  https://docs.astral.sh/uv/"
    exit 1
fi

cd "$PYTHON_PKG_DIR"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv) via uv..."
    uv venv
fi

echo "Installing fastlpr package and dependencies..."
uv pip install -e .

if ! uv run python -c "import fastlpr" &> /dev/null; then
    echo -e "${RED}ERROR: fastlpr import failed after install${NC}"
    exit 1
fi

# Print dependency versions
echo ""
echo "Dependency versions:"
uv run python -c "import numpy; print('  numpy:', numpy.__version__)"
uv run python -c "import scipy; print('  scipy:', scipy.__version__)"
uv run python -c "import matplotlib; print('  matplotlib:', matplotlib.__version__)"

# Create output directory
OUTPUT_DIR="$PYTHON_PKG_DIR/fig/reproduced"
mkdir -p "$OUTPUT_DIR"
echo ""
echo "Output directory: $OUTPUT_DIR"

# Run the master reproduction script
echo ""
echo "================================================================================"
echo "Running Reproduction Script"
echo "================================================================================"
echo ""

cd "$PYTHON_PKG_DIR/examples"
LOG_FILE="$SCRIPT_DIR/reproduction_log.txt"

# Run with timestamp
START_TIME=$(date +%s)
uv run python reproduce_all_figures.py 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "Reproduction Complete"
echo "================================================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] All figures reproduced successfully${NC}"
    echo ""
    echo "Total time: ${ELAPSED} seconds ($(($ELAPSED / 60)) minutes)"
    echo "Output directory: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""

    # Count generated files
    PNG_COUNT=$(find "$OUTPUT_DIR" -name "*_python.png" 2>/dev/null | wc -l)
    PDF_COUNT=$(find "$OUTPUT_DIR" -name "*_python.pdf" 2>/dev/null | wc -l)

    echo "Generated files:"
    echo "  PNG files: $PNG_COUNT"
    echo "  PDF files: $PDF_COUNT"
    echo ""

    # List generated files
    if [ $PNG_COUNT -gt 0 ]; then
        echo "PNG files:"
        find "$OUTPUT_DIR" -name "*_python.png" -exec basename {} \; | sort | sed 's/^/  - /'
        echo ""
    fi

    exit 0
else
    echo -e "${RED}[FAILED] Some figures failed to reproduce${NC}"
    echo ""
    echo "Total time: ${ELAPSED} seconds ($(($ELAPSED / 60)) minutes)"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Please check the log file for details on failures."
    echo ""
    exit 1
fi
