# Results (paper artifacts)

Precomputed outputs for the CIKM 2026 MEG-RW paper. Regenerate from raw runs on a cluster, then refresh this folder:

```bash
python scripts/plot_paper_figures_from_export.py \
  --csv experiments/cikm2026/all_runs_metrics_export.csv \
  --out-dir results/figures
```

## Layout

| Path | Description |
|------|-------------|
| `figures/` | Pareto fairness–utility curves and user-group ΔNDCG heatmaps (PDF + PNG) |
| `tables/all_runs_metrics_export.csv` | Flat export of all successful runs (metrics + fairness scalars) |
| `tables/paper_runs_cleaned.csv` | Filtered runs for paper tables (if present) |
| `tables/sota_baseline_only_metrics_fixed.csv` | SOTA reranking baseline comparison export |

Raw training outputs (`metrics.json`, checkpoints) stay under `experiments/` on the server and are **not** committed to Git.
