#!/usr/bin/env bash
# =============================================================================
# ML-1M weighted LightGCN — Batch 1–3 (baseline → α runs with eval_inputs.json)
# =============================================================================
# Batch 1: seed 0 — baseline + α=0.2 only
# Batch 2: seed 0 — baseline + α ∈ {0.1, 0.2, 0.4, 0.8}
# Batch 3: seeds 0–4 — same α grid per seed
#
# Usage (from repo root):
#   chmod +x scripts/run_ml1m_batches_1_3.sh
#   BATCH=1 ./scripts/run_ml1m_batches_1_3.sh
#   BATCH=2 ./scripts/run_ml1m_batches_1_3.sh
#   BATCH=3 ./scripts/run_ml1m_batches_1_3.sh
#   BATCH=all ./scripts/run_ml1m_batches_1_3.sh    # runs 1 then 2 then 3
#
# Quick pipeline check (2 epochs):
#   SMOKETEST=1 BATCH=1 ./scripts/run_ml1m_batches_1_3.sh
#
# Rules:
#   • Each (seed) baseline runs first; saves eval_inputs.json
#   • Every α run uses --fairness-baseline-eval-json pointing at that seed’s baseline
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

BATCH="${BATCH:-1}"
SMOKETEST="${SMOKETEST:-0}"

DATASET_SHORT="ml1m"
MODEL_DIR="weighted_lightgcn"
MANIFEST="experiments/cikm2026/manifest.csv"

if [[ "$SMOKETEST" == "1" ]]; then
  BASE_CFG=(config/cikm_base.yaml config/cikm_ml1m_lightgcn.yaml config/cikm_smoke_train.yaml)
else
  BASE_CFG=(config/cikm_base.yaml config/cikm_ml1m_lightgcn.yaml)
fi

exp_root() {
  echo "experiments/cikm2026/${DATASET_SHORT}/${MODEL_DIR}/seed_${1}"
}

baseline_dir() { echo "$(exp_root "$1")/baseline"; }
alpha_dir() { echo "$(exp_root "$1")/alpha_${2}"; }

write_baseline_yaml() {
  local seed="$1"
  local f="$2"
  cat >"$f" <<EOF
seed: ${seed}
cikm_backbone: weighted_lightgcn
meg_rw_alpha: 0.0
cikm_dataset_short: ${DATASET_SHORT}
cikm_model_short: ${MODEL_DIR}
cikm_run_label: baseline
cikm_manifest_path: ${MANIFEST}
EOF
}

write_alpha_yaml() {
  local seed="$1"
  local alpha="$2"
  local f="$3"
  local bl
  bl="$(baseline_dir "$seed")"
  cat >"$f" <<EOF
seed: ${seed}
cikm_backbone: weighted_lightgcn
meg_rw_alpha: ${alpha}
cikm_dataset_short: ${DATASET_SHORT}
cikm_model_short: ${MODEL_DIR}
cikm_run_label: alpha_${alpha}
cikm_manifest_path: ${MANIFEST}
cikm_baseline_run_dir: ${bl}
fairness_baseline_eval_json: ${bl}/eval_inputs.json
EOF
}

run_baseline() {
  local seed="$1"
  local bd tmp
  bd="$(baseline_dir "$seed")"
  mkdir -p "$bd"
  tmp="$(mktemp -t cikm_baseline_XXXXXX.yaml)"
  write_baseline_yaml "$seed" "$tmp"
  echo ">>> [baseline] seed=${seed} -> ${bd}"
  python3 -m cikm_train.run_experiment \
    --config "${BASE_CFG[@]}" "$tmp" \
    --fairness-audit \
    --fairness-report-dir "$bd" \
    --fairness-save-eval-inputs
  rm -f "$tmp"
  [[ -f "${bd}/eval_inputs.json" ]] || {
    echo "ERROR: missing ${bd}/eval_inputs.json" >&2
    exit 1
  }
}

run_alpha() {
  local seed="$1"
  local alpha="$2"
  local ad bl tmp
  ad="$(alpha_dir "$seed" "$alpha")"
  bl="$(baseline_dir "$seed")"
  mkdir -p "$ad"
  tmp="$(mktemp -t cikm_alpha_XXXXXX.yaml)"
  write_alpha_yaml "$seed" "$alpha" "$tmp"
  echo ">>> [alpha=${alpha}] seed=${seed} -> ${ad}"
  python3 -m cikm_train.run_experiment \
    --config "${BASE_CFG[@]}" "$tmp" \
    --fairness-audit \
    --fairness-report-dir "$ad"
  rm -f "$tmp"
}

do_batch1() {
  echo "========== Batch 1: seed 0, baseline + α=0.2 =========="
  run_baseline 0
  run_alpha 0 0.2
}

do_batch2() {
  echo "========== Batch 2: seed 0, baseline + α grid =========="
  run_baseline 0
  for a in 0.1 0.2 0.4 0.8; do
    run_alpha 0 "$a"
  done
}

do_batch3() {
  echo "========== Batch 3: seeds 0–4, baseline + α grid =========="
  local s
  for s in 0 1 2 3 4; do
    run_baseline "$s"
    for a in 0.1 0.2 0.4 0.8; do
      run_alpha "$s" "$a"
    done
  done
}

case "$BATCH" in
  1) do_batch1 ;;
  2) do_batch2 ;;
  3) do_batch3 ;;
  all)
    do_batch1
    do_batch2
    do_batch3
    ;;
  *)
    echo "Set BATCH=1|2|3|all (got: ${BATCH})" >&2
    exit 1
    ;;
esac

echo "Done (BATCH=${BATCH}). Manifest: ${MANIFEST}"
