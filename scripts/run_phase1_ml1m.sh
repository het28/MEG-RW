#!/usr/bin/env bash
# Phase 1 smoke: ML-1M, seed 0 — baseline then α=0.2.
# Baseline = WeightedLightGCN with meg_rw_alpha=0 (uniform weights; avoids stock LightGCN + scipy dok issue).
# For full training remove config/cikm_smoke_train.yaml from the command lines below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

BASE_CFG=(config/cikm_base.yaml config/cikm_ml1m_lightgcn.yaml config/cikm_smoke_train.yaml)
BL="experiments/cikm2026/ml1m/weighted_lightgcn/seed_0/baseline"
AL="experiments/cikm2026/ml1m/weighted_lightgcn/seed_0/alpha_0.2"

echo "=== Baseline (WeightedLightGCN α=0, eval_inputs) -> ${BL}"
python3 -m cikm_train.run_experiment \
  "${BASE_CFG[@]}" config/experiments/ml1m_weighted_lightgcn_seed0_baseline.yaml \
  --fairness-audit \
  --fairness-report-dir "${BL}" \
  --fairness-save-eval-inputs

echo "=== MEG-RW weighted LightGCN α=0.2 -> ${AL}"
python3 -m cikm_train.run_experiment \
  "${BASE_CFG[@]}" config/experiments/ml1m_weighted_lightgcn_seed0_alpha_0.2.yaml \
  --fairness-audit \
  --fairness-report-dir "${AL}"

echo "Done. Inspect fairness_report.json, fairness_sweep.csv, manifest.csv"
