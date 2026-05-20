# MEG-RW: Multi-Group Exposure-calibrated Graph Reweighting for Fair Recommendation

Code and paper results for **MEG-RW** (CIKM 2026): a **preprocessing-only**, model-agnostic reweighting of the training user–item graph that reduces multi-group **item-side exposure imbalance** before learning, paired with a **dual-sided multi-group fairness audit** (item exposure, user utility, transfer matrix).

## Repository layout

```
CIKM2026/
├── src/                 # Core implementation
│   ├── meg_rw/          # Grouping, dominance, edge weights (MEG-RW)
│   ├── recbole_ext/     # Weighted LightGCN / NGCF / BPR / NeuMF / ItemKNN
│   ├── cikm_train/      # RecBole training + experiment runner
│   └── cikm_eval/       # Fairness audit, transfer matrix, metrics
├── scripts/             # Grid runners, export, tables, figures
├── config/              # RecBole YAML (ml-1m, lastfm, per model)
├── tests/               # Unit tests for grouping, transfer, audit
└── results/             # Paper figures + CSV summaries (committed)
    ├── figures/
    └── tables/
```

## Method

Items are split into popularity groups (default: 10% / 20% / 30% / 40% by train degree). For each group \(g\), catalog share \(C_g\) and interaction mass \(M_g\) define dominance \(D_g = M_g/(C_g+\epsilon)\). Training edges are weighted \(\tilde w_{ui} = (D_{g(i)}+\epsilon)^{-\alpha}\). **Only weighted backbones** consume these weights; vanilla LightGCN is an α-invariant control.

**Evaluation:** NDCG@10 / Recall@10 (RecBole); item-side exposure vs catalog targets; user-side NDCG gaps over mainstreamness quartiles; transfer matrix \(T_{h,g}\) and Δ vs baseline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Datasets are loaded by RecBole on first run (`dataset/ml-1m`, Last.fm, etc.).

## Run one experiment

```bash
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
cikm-train --config config/cikm_base.yaml config/cikm_ml1m_lightgcn.yaml \
  --fairness-audit --fairness-report-dir experiments/cikm2026/ml1m/weighted_lightgcn/seed_0/baseline
```

Set `cikm_backbone: weighted_lightgcn` (or `weighted_ngcf`, etc.) in YAML.

## Full grid (server)

```bash
chmod +x scripts/run_full_fairness_grid.sh
DATASETS="ml1m lastfm" MODELS="weighted_lightgcn weighted_ngcf weighted_bpr weighted_itemknn weighted_neumf" \
  SEEDS="0 1 2 3 4 5 6 7 8 9" ALPHAS="0.1 0.2 0.4 0.8" \
  MEG_RW_MODE="dominance" bash scripts/run_full_fairness_grid.sh
```

Export flat CSV from all `metrics.json`:

```bash
python scripts/export_all_runs_metrics_csv.py \
  --roots experiments/cikm2026,experiments/cikm2026_SOTA \
  --out experiments/cikm2026/all_runs_metrics_export.csv
```

Refresh **`results/`** for GitHub:

```bash
chmod +x scripts/prepare_github_results.sh
./scripts/prepare_github_results.sh
python scripts/plot_paper_figures_from_export.py \
  --csv experiments/cikm2026/all_runs_metrics_export.csv \
  --out-dir results/figures
```

## Paper results

See [`results/`](results/): Pareto plots (`fig_pareto_*`), user-group heatmaps (`fig_user_heatmap_*`), and tables (`all_runs_metrics_export.csv`, etc.).


