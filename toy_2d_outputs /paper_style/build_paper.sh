#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf FIG_DPS_Toy2D_Paper.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex FIG_DPS_Toy2D_Paper.tex
  pdflatex FIG_DPS_Toy2D_Paper.tex
else
  echo "No LaTeX engine found. Install latexmk or pdflatex and rerun this script."
  exit 1
fi
