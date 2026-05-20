#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "ndcg_mean": "NDCG@10 (mean)",
    "expdev_mean": "Exposure Deviation (mean)",
    "tail_ratio_mean": "Tail Ratio (mean)",
    "ndcg_gap_max_min_mean": "User NDCG Gap (mean)",
    "transfer_l1_shift_mean": "Transfer L1 Shift (mean)",
}


def _safe_model_name(name: str) -> str:
    return name.replace("weighted_", "w-")


def plot_pareto(agg: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = agg[(agg["dataset"] == ds) & (agg["model"].isin(models))]
        for model in models:
            m = sub[sub["model"] == model].sort_values("alpha")
            if m.empty:
                continue
            ax.plot(
                m["ndcg_mean"],
                m["expdev_mean"],
                marker="o",
                label=_safe_model_name(model),
            )
            for _, r in m.iterrows():
                ax.annotate(f"a={r['alpha']:.1f}", (r["ndcg_mean"], r["expdev_mean"]), fontsize=7)
        ax.set_xlabel(METRIC_LABELS["ndcg_mean"])
        ax.set_ylabel(METRIC_LABELS["expdev_mean"])
        ax.set_title(f"{ds}: Fairness-Utility Pareto")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_{ds}.png", dpi=220)
        plt.close(fig)


def plot_alpha_sensitivity(agg: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    metrics = [
        ("ndcg_mean", "ndcg_std"),
        ("expdev_mean", "expdev_std"),
        ("tail_ratio_mean", "tail_ratio_std"),
        ("ndcg_gap_max_min_mean", "ndcg_gap_max_min_std"),
        ("transfer_l1_shift_mean", "transfer_l1_shift_std"),
    ]
    for ds in datasets:
        for model in models:
            m = agg[(agg["dataset"] == ds) & (agg["model"] == model)].sort_values("alpha")
            if m.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            axes = axes.flatten()
            for i, (mean_col, std_col) in enumerate(metrics):
                ax = axes[i]
                y = m[mean_col]
                yerr = m[std_col].fillna(0.0)
                ax.errorbar(m["alpha"], y, yerr=yerr, marker="o", capsize=3)
                ax.set_title(METRIC_LABELS.get(mean_col, mean_col), fontsize=9)
                ax.set_xlabel("alpha")
                ax.grid(True, alpha=0.25)
            axes[-1].axis("off")
            fig.suptitle(f"{ds} / {_safe_model_name(model)}: alpha sensitivity", fontsize=12)
            fig.tight_layout()
            fig.savefig(out_dir / f"alpha_sensitivity_{ds}_{model}.png", dpi=220)
            plt.close(fig)


def plot_best_alpha_bars(best: pd.DataFrame, out_dir: Path) -> None:
    for ds in sorted(best["dataset"].unique()):
        d = best[best["dataset"] == ds].copy()
        if d.empty:
            continue
        x = range(len(d))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([i - width / 2 for i in x], d["best_utility_alpha"], width=width, label="Best utility alpha")
        ax.bar([i + width / 2 for i in x], d["best_fairness_alpha"], width=width, label="Best fairness alpha")
        ax.set_xticks(list(x))
        ax.set_xticklabels([_safe_model_name(m) for m in d["model"]], rotation=25, ha="right")
        ax.set_ylim(0, 0.9)
        ax.set_ylabel("alpha")
        ax.set_title(f"{ds}: best alpha by objective")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"best_alpha_{ds}.png", dpi=220)
        plt.close(fig)


def plot_exposure_distribution(agg: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    gcols = ["exp_head_mean", "exp_upper_mid_mean", "exp_lower_mid_mean", "exp_tail_mean"]
    glabs = ["Head", "UpperMid", "LowerMid", "Tail"]
    for ds in datasets:
        for model in models:
            m = agg[(agg["dataset"] == ds) & (agg["model"] == model)].sort_values("alpha")
            if m.empty or any(c not in m.columns for c in gcols):
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8))
            x = range(len(m))
            bottom = [0.0] * len(m)
            for c, lab in zip(gcols, glabs):
                vals = m[c].fillna(0.0).to_list()
                ax.bar(x, vals, bottom=bottom, label=lab)
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"a={a:.1f}" for a in m["alpha"]])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Exposure share")
            ax.set_title(f"{ds} / {_safe_model_name(model)}: exposure distribution by alpha")
            ax.legend(ncols=4, fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"exposure_distribution_{ds}_{model}.png", dpi=220)
            plt.close(fig)


def plot_user_group_ndcg(agg: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    gcols = [
        "ndcg_niche_mean",
        "ndcg_semi_niche_mean",
        "ndcg_semi_mainstream_mean",
        "ndcg_mainstream_mean",
    ]
    glabs = ["niche", "semi_niche", "semi_mainstream", "mainstream"]
    for ds in datasets:
        for model in models:
            m = agg[(agg["dataset"] == ds) & (agg["model"] == model)].sort_values("alpha")
            if m.empty:
                continue
            if all(c in m.columns for c in gcols):
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for c, lab in zip(gcols, glabs):
                    ax.plot(m["alpha"], m[c], marker="o", label=lab)
                ax.set_xlabel("alpha")
                ax.set_ylabel("Group NDCG@10 (mean)")
                ax.set_title(f"{ds} / {_safe_model_name(model)}: user-group utility by alpha")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_dir / f"user_group_ndcg_{ds}_{model}.png", dpi=220)
                plt.close(fig)


def plot_seed_variance_boxplots(cleaned: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    for ds in datasets:
        for model in models:
            m = cleaned[
                (cleaned["dataset"] == ds)
                & (cleaned["model"] == model)
                & (cleaned["sota_method"] == "none")
            ].copy()
            if m.empty:
                continue
            alphas = sorted(m["alpha"].dropna().unique().tolist())
            if not alphas:
                continue
            data = [m[m["alpha"] == a]["ndcg@10"].dropna().to_numpy(dtype=float) for a in alphas]
            if all(len(x) == 0 for x in data):
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.boxplot(data, labels=[f"a={a:.1f}" for a in alphas], showmeans=True)
            ax.set_title(f"{ds} / {_safe_model_name(model)}: seed variance (NDCG@10)")
            ax.set_ylabel("NDCG@10")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"seed_variance_boxplot_{ds}_{model}.png", dpi=220)
            plt.close(fig)


def plot_sota_comparison(agg: pd.DataFrame, out_dir: Path, datasets: list[str], models: list[str]) -> None:
    methods = ["pop_inverse", "head_penalty", "xquad_pop", "mmr_pop", "calib_pop"]
    for ds in datasets:
        for model in models:
            sub = agg[(agg["dataset"] == ds) & (agg["model"] == model)].copy()
            if sub.empty:
                continue
            # MEG-RW best fairness point among sota_method=none.
            megrw = sub[sub["sota_method"] == "none"]
            if megrw.empty:
                continue
            megrw_best = megrw.loc[megrw["expdev_mean"].idxmin()]

            rows = []
            for method in methods:
                m = sub[(sub["sota_method"] == method) & (sub["alpha"] == 0.0)]
                if m.empty:
                    continue
                rows.append(
                    {
                        "method": method,
                        "ndcg_mean": float(m["ndcg_mean"].mean()),
                        "expdev_mean": float(m["expdev_mean"].mean()),
                    }
                )
            if not rows:
                continue
            rows.append(
                {
                    "method": "megrw_best",
                    "ndcg_mean": float(megrw_best["ndcg_mean"]),
                    "expdev_mean": float(megrw_best["expdev_mean"]),
                }
            )
            rdf = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(8, 5))
            for _, r in rdf.iterrows():
                ax.scatter(r["ndcg_mean"], r["expdev_mean"], s=70)
                ax.annotate(str(r["method"]), (r["ndcg_mean"], r["expdev_mean"]), fontsize=8)
            ax.set_xlabel(METRIC_LABELS["ndcg_mean"])
            ax.set_ylabel(METRIC_LABELS["expdev_mean"])
            ax.set_title(f"{ds} / {_safe_model_name(model)}: SOTA comparison")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"sota_comparison_{ds}_{model}.png", dpi=220)
            plt.close(fig)


def plot_transfer_heatmaps(cleaned: pd.DataFrame, experiments_roots: list[str], out_dir: Path, datasets: list[str], models: list[str]) -> None:
    def _load_matrix(path: Path):
        try:
            obj = json.loads(path.read_text())
            tr = (obj or {}).get("transfer") or {}
            m = tr.get("matrix")
            ug = tr.get("user_groups")
            ig = tr.get("item_groups")
            if isinstance(m, list) and isinstance(ug, list) and isinstance(ig, list):
                arr = np.array(m, dtype=float)
                if arr.ndim == 2 and arr.size > 0:
                    return arr, [str(x) for x in ug], [str(x) for x in ig]
        except Exception:
            return None
        return None

    roots = [Path(r) for r in experiments_roots if r]
    if not roots:
        return

    for ds in datasets:
        for model in models:
            m = cleaned[
                (cleaned["dataset"] == ds)
                & (cleaned["model"] == model)
                & (cleaned["sota_method"] == "none")
            ].copy()
            if m.empty:
                continue
            base = m[m["alpha"] == 0.0]
            if base.empty:
                continue
            best_row = m.loc[m["expdev"].idxmin()]
            targets = [
                ("baseline", base.iloc[0]),
                ("megrw_best_fairness", best_row),
            ]
            for label, row in targets:
                run_label = str(row["run_label"])
                seed = int(row["seed"])
                found = None
                for root in roots:
                    candidates = list(root.rglob(f"{ds}/{model}/seed_{seed}/{run_label}/fairness_report.json"))
                    if candidates:
                        found = candidates[0]
                        break
                if found is None:
                    continue
                loaded = _load_matrix(found)
                if loaded is None:
                    continue
                mat, ug, ig = loaded
                fig, ax = plt.subplots(figsize=(6.2, 5.0))
                im = ax.imshow(mat, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(ig)))
                ax.set_xticklabels(ig, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(len(ug)))
                ax.set_yticklabels(ug, fontsize=8)
                ax.set_title(f"{ds} / {_safe_model_name(model)}: transfer {label}")
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(out_dir / f"transfer_heatmap_{ds}_{model}_{label}.png", dpi=220)
                plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CIKM-ready figures from paper tables")
    ap.add_argument("--aggregated-csv", default="experiments/cikm2026/paper_tables/paper_aggregated_alpha.csv")
    ap.add_argument("--best-alpha-csv", default="experiments/cikm2026/paper_tables/paper_best_alpha.csv")
    ap.add_argument("--runs-cleaned-csv", default="experiments/cikm2026/paper_tables/paper_runs_cleaned.csv")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--experiments-roots", default="experiments/cikm2026,experiments/cikm2026_SOTA")
    ap.add_argument("--datasets", default="ml1m,lastfm")
    ap.add_argument(
        "--models",
        default="weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    agg = pd.read_csv(args.aggregated_csv)
    best = pd.read_csv(args.best_alpha_csv)
    cleaned = pd.read_csv(args.runs_cleaned_csv)
    exp_roots = [x.strip() for x in args.experiments_roots.split(",") if x.strip()]

    plot_pareto(agg, out_dir, datasets, models)
    plot_alpha_sensitivity(agg, out_dir, datasets, models)
    plot_best_alpha_bars(best, out_dir)
    plot_exposure_distribution(agg, out_dir, datasets, models)
    plot_user_group_ndcg(agg, out_dir, datasets, models)
    plot_seed_variance_boxplots(cleaned, out_dir, datasets, models)
    plot_sota_comparison(agg, out_dir, datasets, models)
    plot_transfer_heatmaps(cleaned, exp_roots, out_dir, datasets, models)
    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
