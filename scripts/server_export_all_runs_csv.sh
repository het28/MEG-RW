#!/usr/bin/env bash
# Run ON THE SERVER from CIKM2026 repo root after ablations finished.
# Prints row count and writes a single flat CSV (like SOTA export), without rsyncing checkpoints.
#
#   cd ~/oqiw47aj/CIKM2026
#   chmod +x scripts/server_export_all_runs_csv.sh
#   ./scripts/server_export_all_runs_csv.sh
#
# Successful runs only:
#   ONLY_SUCCESS=1 ./scripts/server_export_all_runs_csv.sh
#
# scp to Mac:
#   scp user@host:~/oqiw47aj/CIKM2026/experiments/cikm2026/all_runs_metrics_export.csv .

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT="${OUT:-experiments/cikm2026/all_runs_metrics_export.csv}"
ROOTS="${ROOTS:-experiments/cikm2026,experiments/cikm2026_SOTA}"
EXTRA=()
if [[ "${ONLY_SUCCESS:-0}" == "1" ]]; then
  EXTRA+=(--only-success)
fi

python3 scripts/export_all_runs_metrics_csv.py --roots "${ROOTS}" --out "${OUT}" "${EXTRA[@]}"
ls -la "${OUT}"
