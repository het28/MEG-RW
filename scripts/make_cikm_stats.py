#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["ndcg@10", "expdev", "tail_ratio", "ndcg_gap_max_min", "worst_group_ndcg", "transfer_l1_shift"]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # non-param effect size: P(X>Y)-P(X<Y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for a in x:
        gt += np.sum(a > y)
        lt += np.sum(a < y)
    return float((gt - lt) / (len(x) * len(y)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build statistical summary tables for MEG-RW paper")
    ap.add_argument("--runs-csv", default="experiments/cikm2026/paper_tables/paper_runs_cleaned.csv")
    ap.add_argument("--out-dir", default="experiments/cikm2026/paper_tables")
    ap.add_argument("--datasets", default="ml1m,lastfm")
    ap.add_argument(
        "--models",
        default="weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    models = {x.strip() for x in args.models.split(",") if x.strip()}

    df = pd.read_csv(args.runs_csv)
    df = df[df["dataset"].isin(datasets) & df["model"].isin(models)].copy()
    df = df[df["alpha"].isin([0.0, 0.1, 0.2, 0.4, 0.8])].copy()

    # per (dataset,model,alpha,metric) summary with bootstrap CIs
    rows = []
    for (ds, model, sm, sl, alpha), g in df.groupby(["dataset", "model", "sota_method", "sota_lambda", "alpha"]):
        for m in METRICS:
            vals = g[m].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "sota_method": sm,
                    "sota_lambda": sl,
                    "alpha": alpha,
                    "metric": m,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    stats = pd.DataFrame(rows)
    if stats.empty:
        stats = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "sota_method",
                "sota_lambda",
                "alpha",
                "metric",
                "n",
                "mean",
                "std",
                "ci95_lo",
                "ci95_hi",
            ]
        )
    else:
        stats = stats.sort_values(["dataset", "model", "sota_method", "metric", "alpha"])
    stats.to_csv(out_dir / "paper_stats_ci.csv", index=False)

    # paired deltas vs baseline, with effect sizes
    deltas = []
    for (ds, model, sm, sl), gm in df.groupby(["dataset", "model", "sota_method", "sota_lambda"]):
        # Deduplicate possible repeated runs per (seed, alpha) by averaging metric values.
        gm_seed = gm.groupby(["alpha", "seed"], as_index=False)[METRICS].mean(numeric_only=True)
        base = gm_seed[gm_seed["alpha"] == 0.0].set_index("seed")
        for alpha in [0.1, 0.2, 0.4, 0.8]:
            cur = gm_seed[gm_seed["alpha"] == alpha].set_index("seed")
            common = sorted(set(base.index) & set(cur.index))
            if not common:
                continue
            b = base.loc[common]
            c = cur.loc[common]
            for m in METRICS:
                vb = b[m].to_numpy(dtype=float)
                vc = c[m].to_numpy(dtype=float)
                if np.isnan(vb).all() or np.isnan(vc).all():
                    continue
                d = vc - vb
                lo, hi = bootstrap_ci(d[~np.isnan(d)])
                deltas.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "sota_method": sm,
                        "sota_lambda": sl,
                        "alpha": alpha,
                        "metric": m,
                        "n_pairs": len(common),
                        "delta_mean": float(np.nanmean(d)),
                        "delta_std": float(np.nanstd(d, ddof=1)) if len(d) > 1 else 0.0,
                        "delta_ci95_lo": lo,
                        "delta_ci95_hi": hi,
                        "cliffs_delta": cliffs_delta(vc[~np.isnan(vc)], vb[~np.isnan(vb)]),
                    }
                )
    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        delta_df = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "sota_method",
                "sota_lambda",
                "alpha",
                "metric",
                "n_pairs",
                "delta_mean",
                "delta_std",
                "delta_ci95_lo",
                "delta_ci95_hi",
                "cliffs_delta",
            ]
        )
    else:
        delta_df = delta_df.sort_values(["dataset", "model", "sota_method", "metric", "alpha"])
    delta_df.to_csv(out_dir / "paper_stats_delta_vs_baseline.csv", index=False)

    print(f"Wrote: {out_dir / 'paper_stats_ci.csv'}")
    print(f"Wrote: {out_dir / 'paper_stats_delta_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
