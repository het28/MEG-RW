#!/usr/bin/env bash
# One chained pipeline: remaining dominance fixes + catalog/uniform + G=10 + optional G=20 + optional SOTA λ
# + post-hoc aggregation/stats/figures.
#
# Usage (server, tmux):
#   chmod +x scripts/run_cikm_remaining_pipeline.sh
#   ./scripts/run_cikm_remaining_pipeline.sh 2>&1 | tee experiments/cikm2026/pipeline_$(date +%Y%m%d_%H%M%S).log
#
# Control:
#   RUN_FIX_NEUMF=1          # lastfm weighted_neumf seed9 four alphas (dominance)
#   RUN_CATALOG=1
#   RUN_UNIFORM=1
#   RUN_G10=1                # dominance, G=10, w_lightgcn w_ngcf only
#   RUN_G20_OPTIONAL=0       # small grid: ml1m lastfm, w_lightgcn w_ngcf, seeds 0-9, alphas 0.2 0.4
#   RUN_SOTA_LAMBDA_OPTIONAL=0  # pop_inverse + xquad_pop × λ 0.1 0.2 0.4
#   RUN_POSTPROCESS=1        # merge tables + stats + figures
#
# Defaults use main manifest root experiments/cikm2026. SOTO uses experiments/cikm2026_SOTA when CIKM_SOTA_METHOD != none.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# --- toggles (override: RUN_CATALOG=0 ./scripts/...) ---
RUN_FIX_NEUMF="${RUN_FIX_NEUMF:-1}"
RUN_CATALOG="${RUN_CATALOG:-1}"
RUN_UNIFORM="${RUN_UNIFORM:-1}"
RUN_G10="${RUN_G10:-1}"
RUN_G20_OPTIONAL="${RUN_G20_OPTIONAL:-0}"
RUN_SOTA_LAMBDA_OPTIONAL="${RUN_SOTA_LAMBDA_OPTIONAL:-0}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"

# shared training knobs
EPOCHS="${EPOCHS:-100}"
STOPPING_STEP="${STOPPING_STEP:-20}"
EVAL_STEP="${EVAL_STEP:-1}"

ALL_SEEDS="${ALL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ABLAT_SEEDS="${ABLAT_SEEDS:-0 1 2 3 4 5 6 7 8 9}"   # catalog/uniform full multi-seed
ALPHAS_GRID="${ALPHAS_GRID:-0.1 0.2 0.4 0.8}"
ALL_MODELS="${ALL_MODELS:-lightgcn weighted_lightgcn ngcf weighted_ngcf bpr weighted_bpr itemknn weighted_itemknn neumf weighted_neumf}"
G10_MODELS="${G10_MODELS:-weighted_lightgcn weighted_ngcf}"
G20_SEEDS="${G20_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
G20_ALPHAS="${G20_ALPHAS:-0.2 0.4}"

G10_FRACS="${G10_FRACS:-0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}"
G20_FRACS="${G20_FRACS:-0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05}"

echo "=============================================="
echo "CIKM remaining pipeline (start $(date -u +%Y-%m-%dT%H:%MZ))"
echo "ROOT=$ROOT"
echo "RUN_FIX_NEUMF=$RUN_FIX_NEUMF RUN_CATALOG=$RUN_CATALOG RUN_UNIFORM=$RUN_UNIFORM"
echo "RUN_G10=$RUN_G10 RUN_G20_OPTIONAL=$RUN_G20_OPTIONAL RUN_SOTA_LAMBDA_OPTIONAL=$RUN_SOTA_LAMBDA_OPTIONAL"
echo "RUN_POSTPROCESS=$RUN_POSTPROCESS"
echo "=============================================="

if [[ "$RUN_FIX_NEUMF" == "1" ]]; then
  echo ""
  echo ">>> PHASE A: dominance — finish lastfm weighted_neumf seed9 alphas (4 runs)"
  env \
    EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
    SKIP_SUCCESS="${SKIP_SUCCESS:-1}" SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
    BETAS="${BETAS:-0.0}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}" MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}" \
    MEG_RW_MODE="dominance" MEG_RW_GROUP_FRACS="0.1,0.2,0.3,0.4" MEG_RW_NORMALIZE_PHI="false" \
    CIKM_SOTA_METHOD="none" \
    SEEDS="9" DATASETS="lastfm" MODELS="weighted_neumf" \
    ALPHAS="0.1 0.2 0.4 0.8" \
    bash scripts/run_full_fairness_grid.sh
fi

if [[ "$RUN_CATALOG" == "1" ]]; then
  echo ""
  echo ">>> PHASE B: catalog ablation (full grid)"
  env \
    EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
    SKIP_SUCCESS="${SKIP_SUCCESS:-1}" SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
    BETAS="${BETAS:-0.0}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}" MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}" \
    MEG_RW_MODE="catalog" MEG_RW_GROUP_FRACS="0.1,0.2,0.3,0.4" MEG_RW_NORMALIZE_PHI="false" \
    CIKM_SOTA_METHOD="none" \
    SEEDS="$ABLAT_SEEDS" DATASETS="ml1m lastfm" MODELS="$ALL_MODELS" \
    ALPHAS="$ALPHAS_GRID" \
    bash scripts/run_full_fairness_grid.sh
fi

if [[ "$RUN_UNIFORM" == "1" ]]; then
  echo ""
  echo ">>> PHASE C: uniform ablation (full grid)"
  env \
    EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
    SKIP_SUCCESS="${SKIP_SUCCESS:-1}" SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
    BETAS="${BETAS:-0.0}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}" MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}" \
    MEG_RW_MODE="uniform" MEG_RW_GROUP_FRACS="0.1,0.2,0.3,0.4" MEG_RW_NORMALIZE_PHI="false" \
    CIKM_SOTA_METHOD="none" \
    SEEDS="$ABLAT_SEEDS" DATASETS="ml1m lastfm" MODELS="$ALL_MODELS" \
    ALPHAS="$ALPHAS_GRID" \
    bash scripts/run_full_fairness_grid.sh
fi

if [[ "$RUN_G10" == "1" ]]; then
  echo ""
  echo ">>> PHASE D: G=10 group granularity (dominance, weighted LightGCN + NGCF only)"
  env \
    EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
    SKIP_SUCCESS="${SKIP_SUCCESS:-1}" SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
    BETAS="${BETAS:-0.0}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}" MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}" \
    MEG_RW_MODE="dominance" MEG_RW_GROUP_FRACS="$G10_FRACS" MEG_RW_NORMALIZE_PHI="false" \
    CIKM_SOTA_METHOD="none" \
    SEEDS="$ABLAT_SEEDS" DATASETS="ml1m lastfm" MODELS="$G10_MODELS" \
    ALPHAS="$ALPHAS_GRID" \
    bash scripts/run_full_fairness_grid.sh
fi

if [[ "$RUN_G20_OPTIONAL" == "1" ]]; then
  echo ""
  echo ">>> PHASE E (optional): G=20 focused ablation"
  env \
    EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
    SKIP_SUCCESS="${SKIP_SUCCESS:-1}" SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
    BETAS="${BETAS:-0.0}" EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}" MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}" \
    MEG_RW_MODE="dominance" MEG_RW_GROUP_FRACS="$G20_FRACS" MEG_RW_NORMALIZE_PHI="false" \
    CIKM_SOTA_METHOD="none" \
    SEEDS="$G20_SEEDS" DATASETS="ml1m lastfm" MODELS="weighted_lightgcn weighted_ngcf" \
    ALPHAS="$G20_ALPHAS" \
    bash scripts/run_full_fairness_grid.sh
fi

if [[ "$RUN_SOTA_LAMBDA_OPTIONAL" == "1" ]]; then
  echo ""
  echo ">>> PHASE F (optional): SOTA λ sensitivity (pop_inverse + xquad_pop)"
  for LAM in 0.1 0.2 0.4; do
    for METH in pop_inverse xquad_pop; do
      echo "    SOTA method=$METH lambda=$LAM"
      env \
        EPOCHS="$EPOCHS" STOPPING_STEP="$STOPPING_STEP" EVAL_STEP="$EVAL_STEP" \
        SKIP_SUCCESS="${SKIP_SUCCESS:-1}" \
        SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}" \
        BETAS="${BETAS:-0.0}" \
        EXPERIMENTS_ROOT="experiments/cikm2026_SOTA" \
        MANIFEST_NAME="manifest_SOTA.csv" \
        CIKM_SOTA_METHOD="$METH" \
        CIKM_SOTA_LAMBDA="$LAM" \
        CIKM_SOTA_CANDIDATE_MULT="${CIKM_SOTA_CANDIDATE_MULT:-20}" \
        MEG_RW_MODE="dominance" \
        MEG_RW_GROUP_FRACS="0.1,0.2,0.3,0.4" \
        MEG_RW_NORMALIZE_PHI="false" \
        SEEDS="$ALL_SEEDS" \
        DATASETS="ml1m lastfm" \
        MODELS="weighted_lightgcn weighted_ngcf weighted_bpr weighted_itemknn weighted_neumf" \
        ALPHAS="" \
        bash scripts/run_full_fairness_grid.sh
    done
  done
fi

if [[ "$RUN_POSTPROCESS" == "1" ]]; then
  echo ""
  echo ">>> POST: merge roots → tables + bootstrap/Cliff + figures"
  OUT_T="${POST_OUT_DIR:-experiments/cikm2026/paper_tables_all}"
  FIG_T="${POST_FIG_DIR:-paper/figures_all}"
  mkdir -p "$OUT_T" "$FIG_T"

  # Adjust --experiments-roots if you add more top-level folders
  ROOTS="experiments/cikm2026,experiments/cikm2026_SOTA"
  python3 scripts/build_megrw_paper_tables.py \
    --experiments-roots "$ROOTS" \
    --out-dir "$OUT_T" \
    --datasets ml1m,lastfm \
    --models weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf \
    --alphas 0.0,0.1,0.2,0.4,0.8

  python3 scripts/make_cikm_stats.py \
    --runs-csv "$OUT_T/paper_runs_cleaned.csv" \
    --out-dir "$OUT_T" \
    --datasets ml1m,lastfm \
    --models weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf

  python3 scripts/make_cikm_figures.py \
    --aggregated-csv "$OUT_T/paper_aggregated_alpha.csv" \
    --best-alpha-csv "$OUT_T/paper_best_alpha.csv" \
    --runs-cleaned-csv "$OUT_T/paper_runs_cleaned.csv" \
    --experiments-roots "$ROOTS" \
    --out-dir "$FIG_T" \
    --datasets ml1m,lastfm \
    --models weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf

  echo "Wrote tables under: $OUT_T"
  echo "Wrote figures under: $FIG_T"
fi

echo ""
echo "=============================================="
echo "CIKM remaining pipeline (done $(date -u +%Y-%m-%dT%H:%MZ))"
echo "=============================================="
