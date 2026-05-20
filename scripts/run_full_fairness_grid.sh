#!/usr/bin/env bash
set -euo pipefail

# General full-grid runner for CIKM fairness experiments.
# Baseline is always alpha=0.0 and must complete before alpha sweeps per (dataset, model, seed).
# By default, alpha sweeps are only meaningful for weighted backbones.
#
# Example:
#   chmod +x scripts/run_full_fairness_grid.sh
#   EPOCHS=100 \
#   DATASETS="ml1m" \
#   MODELS="weighted_lightgcn weighted_ngcf bpr itemknn neumf" \
#   SEEDS="0 1 2 3 4" \
#   ALPHAS="0.1 0.2 0.4 0.8" \
#   ./scripts/run_full_fairness_grid.sh
#
# Server example (nohup):
#   nohup EPOCHS=200 DATASETS="ml1m" MODELS="weighted_lightgcn weighted_ngcf bpr itemknn neumf" \
#     SEEDS="0 1 2 3 4" ALPHAS="0.1 0.2 0.4 0.8" \
#     ./scripts/run_full_fairness_grid.sh > experiments/cikm2026/full_grid.log 2>&1 &

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# -----------------------------
# Run matrix knobs (override via env)
# -----------------------------
EPOCHS="${EPOCHS:-100}"                  # try 100 first; 200 for final sweep
STOPPING_STEP="${STOPPING_STEP:-20}"     # early stopping patience
EVAL_STEP="${EVAL_STEP:-1}"

DATASETS="${DATASETS:-ml1m lastfm amazonbooks}"  # project datasets
MODELS="${MODELS:-weighted_lightgcn weighted_ngcf weighted_bpr weighted_itemknn weighted_neumf bpr itemknn neumf}"
SEEDS="${SEEDS:-0 1 2 3 4}"
# IMPORTANT: use "-" (not ":-") so ALPHAS="" stays empty when explicitly set.
# This enables baseline-only runs (e.g., SOTA baselines without alpha sweeps).
ALPHAS="${ALPHAS-0.1 0.2 0.4 0.8}"
BETAS="${BETAS:-0.0}"                            # semantic knob grid; 0.0 = disabled
RUN_UNWEIGHTED_ALPHAS="${RUN_UNWEIGHTED_ALPHAS:-0}"  # set 1 to force alpha runs for non-weighted models
SKIP_SUCCESS="${SKIP_SUCCESS:-1}"                    # set 0 to force re-run even if manifest has success
SAVE_EVAL_INPUTS_ALL="${SAVE_EVAL_INPUTS_ALL:-0}"    # set 1 to save eval_inputs.json for alpha/beta runs too
MEG_RW_MODE="${MEG_RW_MODE:-dominance}"              # dominance | catalog | uniform
MEG_RW_GROUP_FRACS="${MEG_RW_GROUP_FRACS:-0.1,0.2,0.3,0.4}"  # comma-separated fractions
MEG_RW_NORMALIZE_PHI="${MEG_RW_NORMALIZE_PHI:-false}"         # true|false
CIKM_SOTA_METHOD="${CIKM_SOTA_METHOD:-none}"                 # none | pop_inverse | head_penalty | xquad_pop | mmr_pop | calib_pop
CIKM_SOTA_LAMBDA="${CIKM_SOTA_LAMBDA:-0.2}"
CIKM_SOTA_CANDIDATE_MULT="${CIKM_SOTA_CANDIDATE_MULT:-20}"

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/cikm2026}"
MANIFEST_NAME="${MANIFEST_NAME:-manifest.csv}"
# Safety default: SOTA reranking runs go to separate root/manifest unless explicitly overridden.
if [[ "${CIKM_SOTA_METHOD}" != "none" ]]; then
  if [[ "${EXPERIMENTS_ROOT}" == "experiments/cikm2026" ]]; then
    EXPERIMENTS_ROOT="experiments/cikm2026_SOTA"
  fi
  if [[ "${MANIFEST_NAME}" == "manifest.csv" ]]; then
    MANIFEST_NAME="manifest_SOTA.csv"
  fi
fi
MANIFEST="${EXPERIMENTS_ROOT}/${MANIFEST_NAME}"

# -----------------------------
# Ablation namespace (avoid path/manifest collisions)
# -----------------------------
sanitize_token() {
  local s="$1"
  s="${s//,/__}"
  s="${s// /_}"
  s="${s//[!a-zA-Z0-9_.-]/_}"
  echo "$s"
}

ablation_tag() {
  local mode_tag fracs_tag norm_tag sota_method_tag sota_lambda_tag
  mode_tag="$(sanitize_token "${MEG_RW_MODE}")"
  fracs_tag="$(sanitize_token "${MEG_RW_GROUP_FRACS}")"
  norm_tag="$(sanitize_token "${MEG_RW_NORMALIZE_PHI}")"
  sota_method_tag="$(sanitize_token "${CIKM_SOTA_METHOD}")"
  sota_lambda_tag="$(sanitize_token "${CIKM_SOTA_LAMBDA}")"
  echo "rwmode_${mode_tag}__fracs_${fracs_tag}__norm_${norm_tag}__sota_${sota_method_tag}__slam_${sota_lambda_tag}"
}

ABLATION_TAG="$(ablation_tag)"

# -----------------------------
# Helpers
# -----------------------------
dataset_yaml() {
  # Expected dataset YAML naming: config/cikm_<dataset_short>_<backbone>.yaml
  # For weighted models, reuse the base backbone config.
  local ds="$1"
  local model="$2"
  local base_model="$model"
  if [[ "$model" == weighted_* ]]; then
    base_model="${model#weighted_}"
  fi
  echo "config/cikm_${ds}_${base_model}.yaml"
}

exp_root() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  echo "${EXPERIMENTS_ROOT}/${ABLATION_TAG}/${ds}/${model}/seed_${seed}"
}

baseline_dir() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  echo "$(exp_root "$ds" "$model" "$seed")/baseline"
}

alpha_dir() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local alpha="$4"
  echo "$(exp_root "$ds" "$model" "$seed")/alpha_${alpha}"
}

write_train_override_yaml() {
  local f="$1"
  cat >"$f" <<EOF
epochs: ${EPOCHS}
stopping_step: ${STOPPING_STEP}
eval_step: ${EVAL_STEP}
EOF
}

write_baseline_yaml() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local f="$4"
  cat >"$f" <<EOF
seed: ${seed}
cikm_backbone: ${model}
meg_rw_alpha: 0.0
meg_rw_mode: ${MEG_RW_MODE}
meg_rw_group_fracs: [${MEG_RW_GROUP_FRACS}]
meg_rw_normalize_phi: ${MEG_RW_NORMALIZE_PHI}
cikm_sota_method: ${CIKM_SOTA_METHOD}
cikm_sota_lambda: ${CIKM_SOTA_LAMBDA}
cikm_sota_candidate_mult: ${CIKM_SOTA_CANDIDATE_MULT}
cikm_dataset_short: ${ds}
cikm_model_short: ${model}
cikm_run_label: baseline__${ABLATION_TAG}
cikm_manifest_path: ${MANIFEST}
EOF
}

write_alpha_yaml() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local alpha="$4"
  local beta="$5"
  local f="$6"
  local bl
  bl="$(baseline_dir "$ds" "$model" "$seed")"
  local sem_enable="false"
  if [[ "$beta" != "0" && "$beta" != "0.0" ]]; then
    sem_enable="true"
  fi
  cat >"$f" <<EOF
seed: ${seed}
cikm_backbone: ${model}
meg_rw_alpha: ${alpha}
meg_rw_mode: ${MEG_RW_MODE}
meg_rw_group_fracs: [${MEG_RW_GROUP_FRACS}]
meg_rw_normalize_phi: ${MEG_RW_NORMALIZE_PHI}
cikm_sota_method: ${CIKM_SOTA_METHOD}
cikm_sota_lambda: ${CIKM_SOTA_LAMBDA}
cikm_sota_candidate_mult: ${CIKM_SOTA_CANDIDATE_MULT}
meg_sem_enable: ${sem_enable}
meg_sem_beta: ${beta}
cikm_dataset_short: ${ds}
cikm_model_short: ${model}
cikm_run_label: alpha_${alpha}_beta_${beta}__${ABLATION_TAG}
cikm_manifest_path: ${MANIFEST}
cikm_baseline_run_dir: ${bl}
fairness_baseline_eval_json: ${bl}/eval_inputs.json
EOF
}

run_baseline() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local model_cfg train_cfg local_cfg outdir
  model_cfg="$(dataset_yaml "$ds" "$model")"
  outdir="$(baseline_dir "$ds" "$model" "$seed")"

  [[ -f "$model_cfg" ]] || { echo "Missing config: $model_cfg" >&2; exit 1; }
  mkdir -p "$outdir"

  train_cfg="$(mktemp -t cikm_train_override_XXXXXX.yaml)"
  local_cfg="$(mktemp -t cikm_local_baseline_XXXXXX.yaml)"
  write_train_override_yaml "$train_cfg"
  write_baseline_yaml "$ds" "$model" "$seed" "$local_cfg"

  echo ">>> [baseline] ds=${ds} model=${model} seed=${seed} -> ${outdir}"
  python3 -m cikm_train.run_experiment \
    --config config/cikm_base.yaml "$model_cfg" "$train_cfg" "$local_cfg" \
    --fairness-audit \
    --fairness-report-dir "$outdir" \
    --fairness-save-eval-inputs

  rm -f "$train_cfg" "$local_cfg"
  [[ -f "${outdir}/eval_inputs.json" ]] || {
    echo "ERROR: missing ${outdir}/eval_inputs.json" >&2
    exit 1
  }
}

run_alpha() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local alpha="$4"
  local beta="$5"
  local model_cfg train_cfg local_cfg outdir
  model_cfg="$(dataset_yaml "$ds" "$model")"
  outdir="$(alpha_dir "$ds" "$model" "$seed" "${alpha}_beta_${beta}")"

  [[ -f "$model_cfg" ]] || { echo "Missing config: $model_cfg" >&2; exit 1; }
  mkdir -p "$outdir"

  train_cfg="$(mktemp -t cikm_train_override_XXXXXX.yaml)"
  local_cfg="$(mktemp -t cikm_local_alpha_XXXXXX.yaml)"
  write_train_override_yaml "$train_cfg"
  write_alpha_yaml "$ds" "$model" "$seed" "$alpha" "$beta" "$local_cfg"

  echo ">>> [alpha=${alpha} beta=${beta}] ds=${ds} model=${model} seed=${seed} -> ${outdir}"
  if [[ "$SAVE_EVAL_INPUTS_ALL" == "1" ]]; then
    python3 -m cikm_train.run_experiment \
      --config config/cikm_base.yaml "$model_cfg" "$train_cfg" "$local_cfg" \
      --fairness-audit \
      --fairness-report-dir "$outdir" \
      --fairness-save-eval-inputs
  else
    python3 -m cikm_train.run_experiment \
      --config config/cikm_base.yaml "$model_cfg" "$train_cfg" "$local_cfg" \
      --fairness-audit \
      --fairness-report-dir "$outdir"
  fi

  rm -f "$train_cfg" "$local_cfg"
}

model_uses_alpha() {
  local model="$1"
  if [[ "$model" == *"weighted"* ]]; then
    return 0
  fi
  [[ "$RUN_UNWEIGHTED_ALPHAS" == "1" ]]
}

model_is_supported() {
  local model="$1"
  case "$model" in
    lightgcn|weighted_lightgcn|ngcf|weighted_ngcf|bpr|weighted_bpr|itemknn|weighted_itemknn|neumf|weighted_neumf) return 0 ;;
    *) return 1 ;;
  esac
}

experiment_id() {
  local ds="$1"
  local model="$2"
  local seed="$3"
  local run_label="$4"
  echo "${ds}__${model}__seed${seed}__${run_label}"
}

manifest_has_success() {
  local exp_id="$1"
  [[ -f "$MANIFEST" ]] || return 1
  python3 - "$MANIFEST" "$exp_id" <<'PY'
import csv, sys
manifest, exp_id = sys.argv[1], sys.argv[2]
latest = None
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("experiment_id") == exp_id:
            latest = row
print("1" if latest and latest.get("status") == "success" else "0")
PY
}

echo "=== Full fairness grid start ==="
echo "EPOCHS=${EPOCHS} STOPPING_STEP=${STOPPING_STEP} EVAL_STEP=${EVAL_STEP}"
echo "DATASETS=${DATASETS}"
echo "MODELS=${MODELS}"
echo "SEEDS=${SEEDS}"
echo "ALPHAS=${ALPHAS}"
echo "BETAS=${BETAS}"
echo "RUN_UNWEIGHTED_ALPHAS=${RUN_UNWEIGHTED_ALPHAS}"
echo "SKIP_SUCCESS=${SKIP_SUCCESS}"
echo "SAVE_EVAL_INPUTS_ALL=${SAVE_EVAL_INPUTS_ALL}"
echo "MEG_RW_MODE=${MEG_RW_MODE}"
echo "MEG_RW_GROUP_FRACS=${MEG_RW_GROUP_FRACS}"
echo "MEG_RW_NORMALIZE_PHI=${MEG_RW_NORMALIZE_PHI}"
echo "CIKM_SOTA_METHOD=${CIKM_SOTA_METHOD}"
echo "CIKM_SOTA_LAMBDA=${CIKM_SOTA_LAMBDA}"
echo "CIKM_SOTA_CANDIDATE_MULT=${CIKM_SOTA_CANDIDATE_MULT}"
echo "EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT}"
echo "MANIFEST_NAME=${MANIFEST_NAME}"
echo "ABLATION_TAG=${ABLATION_TAG}"

for ds in $DATASETS; do
  for model in $MODELS; do
    if ! model_is_supported "$model"; then
      echo "ERROR: unsupported model '${model}'" >&2
      echo "Supported: lightgcn weighted_lightgcn ngcf weighted_ngcf bpr weighted_bpr itemknn weighted_itemknn neumf weighted_neumf" >&2
      exit 1
    fi
    for seed in $SEEDS; do
      bl_id="$(experiment_id "$ds" "$model" "$seed" "baseline__${ABLATION_TAG}")"
      if [[ "$SKIP_SUCCESS" == "1" ]] && [[ "$(manifest_has_success "$bl_id")" == "1" ]]; then
        echo ">>> [skip success] ${bl_id}"
      else
        run_baseline "$ds" "$model" "$seed"
      fi
      if model_uses_alpha "$model"; then
        for beta in $BETAS; do
          for alpha in $ALPHAS; do
            run_label="alpha_${alpha}_beta_${beta}__${ABLATION_TAG}"
            a_id="$(experiment_id "$ds" "$model" "$seed" "$run_label")"
            if [[ "$SKIP_SUCCESS" == "1" ]] && [[ "$(manifest_has_success "$a_id")" == "1" ]]; then
              echo ">>> [skip success] ${a_id}"
            else
              run_alpha "$ds" "$model" "$seed" "$alpha" "$beta"
            fi
          done
        done
      else
        echo ">>> [skip alpha] ds=${ds} model=${model} seed=${seed} (unweighted backbone)"
      fi
    done
  done
done

echo "=== Full fairness grid done ==="
echo "Manifest: ${MANIFEST}"
