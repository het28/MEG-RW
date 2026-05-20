#!/usr/bin/env bash
# Copy paper figures and key CSVs into results/ for GitHub (no raw experiment trees).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/figures results/tables

FIG_SRC="${FIG_SRC:-experiments/cikm2026/paper_figures}"
if [[ -d "$FIG_SRC" ]]; then
  cp -f "$FIG_SRC"/fig_*.{pdf,png} results/figures/ 2>/dev/null || true
  echo "Figures: $(ls results/figures | wc -l | tr -d ' ') files"
else
  echo "Skip figures (missing $FIG_SRC)"
fi

copy_csv() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "results/tables/$dst"
    echo "  tables/$dst"
  fi
}

copy_csv experiments/cikm2026/all_runs_metrics_export.csv all_runs_metrics_export.csv
copy_csv experiments/cikm2026/paper_tables_all/paper_runs_cleaned.csv paper_runs_cleaned.csv
copy_csv experiments/cikm2026_SOTA/sota_baseline_only_metrics_fixed.csv sota_baseline_only_metrics_fixed.csv
copy_csv experiments/cikm2026/paper_tables_all/paper_aggregated_alpha.csv paper_aggregated_alpha.csv
copy_csv experiments/cikm2026/paper_tables_all/paper_compact_table.csv paper_compact_table.csv

echo "Done. results/ ready for git add results/"
