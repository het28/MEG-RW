#!/usr/bin/env bash
# Pull experiment artifacts from the CIKM grid server (ablation tags: dominance, catalog, uniform, etc.)
# plus manifest. Run from repo root on your Mac after: ssh-add <your-key>
#
# Usage:
#   ./scripts/fetch_server_experiments.sh
#   REMOTE_USER=oqiw47aj REMOTE_HOST=dtdh206.cs.uni-magdeburg.de REMOTE_CIkm_ROOT=~/oqiw47aj/CIKM2026 ./scripts/fetch_server_experiments.sh
#
# Dry run (no writes):
#   DRY_RUN=1 ./scripts/fetch_server_experiments.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE_USER="${REMOTE_USER:-oqiw47aj}"
REMOTE_HOST="${REMOTE_HOST:-dtdh206.cs.uni-magdeburg.de}"
REMOTE_CIkm_ROOT="${REMOTE_CIkm_ROOT:-~/oqiw47aj/CIKM2026}"

# rsync remote path (expand ~ on remote)
REMOTE_BASE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CIkm_ROOT}"

RSYNC_FLAGS=(-avz --human-readable --partial --inplace)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(-n)
  echo "[dry-run] no files will be written"
fi

echo "Remote: ${REMOTE_BASE}"
echo "Local:  ${ROOT}/experiments/"
echo ""

mkdir -p "${ROOT}/experiments/cikm2026" "${ROOT}/experiments/cikm2026_SOTA"

# Main grid: all ablation namespaces + manifest + any loose files under cikm2026
echo ">>> rsync experiments/cikm2026/ (manifest + rwmode_* ablations)"
rsync "${RSYNC_FLAGS[@]}" \
  "${REMOTE_BASE}/experiments/cikm2026/" \
  "${ROOT}/experiments/cikm2026/"

# SOTA rerank runs (separate manifest/manifest_SOTA if used)
if [[ "${FETCH_SOTA:-1}" == "1" ]]; then
  echo ">>> rsync experiments/cikm2026_SOTA/ (if present on server)"
  rsync "${RSYNC_FLAGS[@]}" \
    "${REMOTE_BASE}/experiments/cikm2026_SOTA/" \
    "${ROOT}/experiments/cikm2026_SOTA/" || true
fi

echo ""
echo "Done. Rebuild paper CSVs locally:"
echo "  export PYTHONPATH=${ROOT}/src:\${PYTHONPATH:-}"
echo "  python scripts/build_megrw_paper_tables.py \\"
echo "    --experiments-roots experiments/cikm2026,experiments/cikm2026_SOTA \\"
echo "    --out-dir experiments/cikm2026/paper_tables_all \\"
echo "    --datasets ml1m,lastfm \\"
echo "    --models weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf \\"
echo "    --alphas 0.0,0.1,0.2,0.4,0.8"
