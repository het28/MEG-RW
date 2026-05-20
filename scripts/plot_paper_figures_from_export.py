#!/usr/bin/env python3
"""
Paper figures from experiments/cikm2026/all_runs_metrics_export.csv

Plot 1: Fairness–utility trade-off (two figures: ml1m, lastfm)
  Default layout matches common ML-fairness practice (Fairlearn / mlr3fairness style):
  **x = ExpDev** (exposure / item-side cost; **smaller to the left is fairer**),
  **y = NDCG@10** (**higher up is better utility**). Desirable region: **upper-left**.
  Each colored line is one model with markers along increasing α.

Plot 2: User-group NDCG delta vs baseline heatmaps (alpha 0.4 and 0.8)

Run from repo root:
  python scripts/plot_paper_figures_from_export.py \\
    --csv experiments/cikm2026/all_runs_metrics_export.csv \\
    --out-dir experiments/cikm2026/paper_figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODELS_ORDER = [
    "weighted_itemknn",
    "weighted_bpr",
    "weighted_neumf",
    "weighted_ngcf",
    "weighted_lightgcn",
]

DISPLAY_NAMES = {
    "weighted_itemknn": "ItemKNN",
    "weighted_bpr": "BPR",
    "weighted_neumf": "NeuMF",
    "weighted_ngcf": "NGCF",
    "weighted_lightgcn": "LightGCN",
}

ALPHA_MARKERS = {
    0.0: "o",
    0.1: "s",
    0.2: "^",
    0.4: "D",
    0.8: "*",
}

USER_ROWS = ["niche", "semi_niche", "semi_mainstream", "mainstream"]
USER_COLS_CSV = ["ndcg_niche", "ndcg_semi_niche", "ndcg_semi_mainstream", "ndcg_mainstream"]

DATASETS_PARETO = ["ml1m", "lastfm"]
ALPHAS = [0.0, 0.1, 0.2, 0.4, 0.8]


def parse_alpha_row(row: pd.Series) -> float:
    mr = row.get("meg_rw_alpha")
    if mr is not None and pd.notna(mr) and str(mr).strip() != "":
        try:
            return float(mr)
        except (TypeError, ValueError):
            pass
    ac = str(row.get("alpha_cell", "")).strip().lower()
    if ac in ("baseline", "0", "0.0"):
        return 0.0
    try:
        return float(ac)
    except ValueError:
        pass
    rl = str(row.get("run_label", ""))
    m = re.search(r"alpha_([0-9.]+)", rl)
    if m:
        return float(m.group(1))
    return float("nan")


def parse_beta_run_label(row: pd.Series) -> float:
    rl = str(row.get("run_label", ""))
    if "_beta_" not in rl:
        return 0.0
    m = re.search(r"_beta_([0-9.]+)", rl)
    return float(m.group(1)) if m else 0.0


def filter_megrw_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """Dominance MEG-RW only: reranking baselines excluded; beta == 0.

    Includes canonical tags with ``__sota_none__`` and legacy dominance paths
    that omit ``sota_*`` (but excludes any ``sota_<reranker>`` substring).
    """
    out = df[
        (df["status"].astype(str).str.lower() == "success")
        & (df["dataset"].isin(DATASETS_PARETO))
        & (df["model"].isin(MODELS_ORDER))
    ].copy()
    tag = out["ablation_tag"].fillna("").astype(str)
    mask_dom = tag.str.contains("rwmode_dominance", case=False, na=False)
    rerank_tokens = (
        "sota_pop_inverse",
        "sota_xquad_pop",
        "sota_mmr_pop",
        "sota_calib_pop",
        "sota_head_penalty",
    )
    pat = "|".join(rerank_tokens)
    mask_no_rerank = ~tag.str.contains(pat, case=False, na=False)
    out = out[mask_dom & mask_no_rerank]
    out["_alpha"] = out.apply(parse_alpha_row, axis=1)
    out["_beta"] = out.apply(parse_beta_run_label, axis=1)
    out = out[out["_alpha"].isin(ALPHAS)]
    out = out[np.isclose(out["_beta"], 0.0)]
    # Snap to grid (avoids float drift vs ALPHAS / marker map)
    out["_alpha"] = np.round(out["_alpha"].astype(float), 4)
    return out


def dedupe_one_row_per_run(out: pd.DataFrame) -> pd.DataFrame:
    """Same (dataset, model, seed, alpha) can appear under legacy + canonical ablation dirs.

    Prefer ``__sota_none__`` path, then latest ``finished_at``, so means are not averaged
    across duplicate logical runs.
    """
    d = out.copy()
    tag = d["ablation_tag"].astype(str)
    d["_pref"] = tag.str.contains("__sota_none__", na=False).astype(int)
    d["_ft"] = pd.to_datetime(d["finished_at"], utc=True, errors="coerce")
    d = d.sort_values(
        ["dataset", "model", "seed", "_alpha", "_pref", "_ft"],
        ascending=[True, True, True, True, False, False],
    )
    d = d.drop_duplicates(subset=["dataset", "model", "seed", "_alpha"], keep="first")
    return d.drop(columns=["_pref", "_ft"], errors="ignore")


def aggregate_mean_seed(df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["dataset", "model", "_alpha"]
    agg = (
        df.groupby(gcols, as_index=False)
        .agg(
            ndcg_mean=("ndcg@10", "mean"),
            expdev_mean=("expdev", "mean"),
        )
        .sort_values(["dataset", "model", "_alpha"])
    )
    return agg


def plot_pareto_dataset(
    agg: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    *,
    ndcg_on_x: bool,
    invert_fairness_axis: bool,
) -> None:
    """α-sweep polylines: markers = α, color = model.

    Default (``ndcg_on_x=False``): **ExpDev on x**, **NDCG@10 on y** — same idea as
    fairness–accuracy trade-off plots (accuracy ↑, unfairness often →): **upper-left**
    is the good corner without axis tricks.

    Legacy (``ndcg_on_x=True``): NDCG on x, ExpDev on y; ``invert_fairness_axis`` flips
    the fairness axis when it is vertical (old experiment).
    """
    sub = agg[agg["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.tab10
    model_legend_handles: list[Line2D] = []
    for mi, model in enumerate(MODELS_ORDER):
        m = sub[sub["model"] == model].sort_values("_alpha", kind="mergesort")
        if m.empty:
            continue
        color = cmap(mi % 10)
        ndcg = m["ndcg_mean"].astype(float).values
        expd = m["expdev_mean"].astype(float).values
        alphas = m["_alpha"].astype(float).values
        if ndcg_on_x:
            xs, ys = ndcg, expd
        else:
            xs, ys = expd, ndcg
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            zorder=1,
        )
        model_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.5,
                label=DISPLAY_NAMES[model],
            )
        )
        ox, oy = 4 + mi * 10, 4 + mi * 6
        for a, x, y in zip(alphas, xs, ys):
            af = float(a)
            mk = ALPHA_MARKERS.get(round(af, 2), "o")
            ax.scatter(
                [x],
                [y],
                marker=mk,
                s=7**2,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
            # Only label baseline α=0 (not α=0.8)
            if np.isclose(af, 0.0):
                ax.annotate(
                    "α=0",
                    (x, y),
                    textcoords="offset points",
                    xytext=(ox, oy),
                    fontsize=8,
                    color="0.2",
                )

    if ndcg_on_x:
        ax.set_xlabel("NDCG@10 (↑ better →)")
        ax.set_ylabel("ExpDev (mean |exposure−catalog| ; ↓ better)")
        if invert_fairness_axis:
            ax.invert_yaxis()
    else:
        ax.set_xlabel("ExpDev — exposure deviation vs catalog (↓ fairer to the left)")
        ax.set_ylabel("NDCG@10 (↑ better)")
        if invert_fairness_axis:
            ax.invert_xaxis()
    ax.text(
        0.02,
        0.02,
        "Desirable: upper-left. Each line: one model; marker = α (see legend).",
        transform=ax.transAxes,
        fontsize=7,
        color="0.35",
        va="bottom",
    )
    ax.grid(True, alpha=0.3)

    leg_m = ax.legend(
        handles=model_legend_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )
    ax.add_artist(leg_m)
    alpha_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=ALPHA_MARKERS[a],
            linestyle="None",
            markersize=9,
            markerfacecolor="0.92",
            markeredgecolor="0.25",
            markeredgewidth=0.8,
            label=f"{a:g}",
        )
        for a in sorted(ALPHA_MARKERS.keys())
    ]
    ax.legend(
        handles=alpha_handles,
        title=r"$\alpha$ (marker)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.38),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )

    short = "MovieLens-1M" if dataset == "ml1m" else "LastFM"
    ax.set_title(f"Pareto trade-off — {short}")
    fig.tight_layout(rect=[0.0, 0.03, 0.72, 0.98])
    stem = "fig_pareto_ml1m_all_models" if dataset == "ml1m" else "fig_pareto_lastfm_all_models"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _heatmap_norm_and_cmap(mat: np.ndarray) -> tuple[Normalize | TwoSlopeNorm, str]:
    """Diverging scale around 0 with *asymmetric* limits when data are skewed.

    Symmetric ``±max(|Δ|)`` maps tiny positives next to large negatives into the
    washed-out middle of ``coolwarm`` (looks grey). Use ``TwoSlopeNorm`` with
    ``vmin=min(Δ), vmax=max(Δ)`` so each sign gets half the colormap (matplotlib
    behaviour; cf. seaborn heatmap ``center=`` discussion).

    Colormap: ``RdBu_r`` — saturated blue (loss) vs red (gain); clearer than
    ``coolwarm`` for print/PDF (matplotlib colormap guide: diverging RdBu).
    """
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return TwoSlopeNorm(-1e-6, 0.0, 1e-6), "RdBu_r"
    d_min = float(np.nanmin(mat))
    d_max = float(np.nanmax(mat))
    eps = 1e-9
    if d_min < 0 < d_max:
        # Pull bounds slightly past extrema so 0 stays inside
        return TwoSlopeNorm(vmin=d_min - eps, vcenter=0.0, vmax=d_max + eps), "RdBu_r"
    if d_max <= 0:
        # All non-positive: sequential (no meaningful white centre)
        return Normalize(vmin=d_min - eps, vmax=max(d_max, d_min * 1e-6)), "Blues_r"
    # All non-negative
    return Normalize(vmin=min(d_min, d_max * 1e-6) - eps, vmax=d_max + eps), "Reds"


def _heatmap_colorbar_ticks(norm: Normalize | TwoSlopeNorm, mat: np.ndarray) -> list[float]:
    """Ticks that always include 0, vmin, vmax so the red (gain) side is labeled."""
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return [-1e-6, 0.0, 1e-6]
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if isinstance(norm, TwoSlopeNorm) and lo < 0 < hi:
        ticks: list[float] = [lo, lo / 2, 0.0, hi / 2, hi]
    elif isinstance(norm, TwoSlopeNorm):
        ticks = [norm.vmin, 0.0, norm.vmax]
    else:
        ticks = list(np.linspace(lo, hi, num=5))
    out: list[float] = []
    for t in sorted(ticks):
        if not out or abs(t - out[-1]) > 1e-12 * max(1.0, abs(t)):
            out.append(float(t))
    return out


def heatmap_user_delta(
    df: pd.DataFrame,
    dataset: str,
    alpha_target: float,
    out_dir: Path,
    *,
    cmap_override: str | None = None,
) -> None:
    """Per-seed delta vs alpha=0, then mean over seeds."""
    d = df[(df["dataset"] == dataset) & (df["_alpha"].isin([0.0, alpha_target]))].copy()
    if d.empty:
        return
    base = d[np.isclose(d["_alpha"], 0.0)]
    cand = d[np.isclose(d["_alpha"], alpha_target)]
    rows_out: list[dict[str, float | str]] = []
    for model in MODELS_ORDER:
        b = base[base["model"] == model][["seed"] + USER_COLS_CSV].copy()
        c = cand[cand["model"] == model][["seed"] + USER_COLS_CSV].copy()
        b["seed"] = b["seed"].astype(int)
        c["seed"] = c["seed"].astype(int)
        merged = c.merge(b, on="seed", suffixes=("_c", "_b"))
        if merged.empty:
            continue
        for ug, col in zip(USER_ROWS, USER_COLS_CSV):
            cc, cb = f"{col}_c", f"{col}_b"
            if cc not in merged.columns or cb not in merged.columns:
                continue
            delta = merged[cc].astype(float) - merged[cb].astype(float)
            rows_out.append(
                {
                    "user_group": ug,
                    "model": model,
                    "delta": float(delta.mean()),
                }
            )
    if not rows_out:
        return
    long_df = pd.DataFrame(rows_out)
    mat = np.full((len(USER_ROWS), len(MODELS_ORDER)), np.nan, dtype=float)
    for i, ug in enumerate(USER_ROWS):
        for j, model in enumerate(MODELS_ORDER):
            sub = long_df[(long_df["user_group"] == ug) & (long_df["model"] == model)]
            if not sub.empty:
                mat[i, j] = sub["delta"].iloc[0]

    norm, cmap_name = _heatmap_norm_and_cmap(mat)
    if cmap_override:
        cmap_name = cmap_override
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#bbbbbb")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )
    ny, nx = mat.shape
    for j in range(nx):
        for i in range(ny):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="0.35")
                continue
            # 3 decimals typical for NDCG deltas in paper
            s = f"{v:+.3f}".replace("+-", "-")
            txt = ax.text(j, i, s, ha="center", va="center", fontsize=8.5, color="0.12")
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])

    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODELS_ORDER])
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(USER_ROWS)
    ax.set_xlabel("Model")
    ax.set_ylabel("User group")
    ax.set_xticks(np.arange(-0.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ny, 1), minor=True)
    ax.grid(which="minor", color="0.45", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    a_tag = str(alpha_target).replace(".", "")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ NDCG@10 (mean over seeds)")
    tickvals = _heatmap_colorbar_ticks(norm, mat)
    cbar.set_ticks(tickvals)
    cbar.set_ticklabels([f"{t:.4g}" for t in tickvals])
    cbar.ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", clip_on=False)
    fig.tight_layout()
    ds_stem = "ml1m" if dataset == "ml1m" else "lastfm"
    stem = f"fig_user_heatmap_{ds_stem}_alpha{a_tag}"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/cikm2026/all_runs_metrics_export.csv")
    ap.add_argument("--out-dir", default="experiments/cikm2026/paper_figures")
    ap.add_argument(
        "--pareto-ndcg-x",
        action="store_true",
        help="Legacy layout: NDCG@10 on x, ExpDev on y (default is ExpDev on x, NDCG on y).",
    )
    ap.add_argument(
        "--invert-fairness-axis",
        action="store_true",
        help="Flip the fairness axis (y if NDCG on x, x if NDCG on y). Default: no flip.",
    )
    ap.add_argument(
        "--heatmap-cmap",
        default="",
        help="Override heatmap colormap (default: RdBu_r when mixed sign, else Blues_r/Reds).",
    )
    ap.add_argument(
        "--ablation-require",
        choices=("auto", "canonical", "legacy"),
        default="auto",
        help="auto: dominance, no rerank tokens, dedupe legacy vs __sota_none__. "
        "canonical: only tags containing __sota_none__. legacy: dominance tag without sota_* substring.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    filt = filter_megrw_dominance(df)
    if filt.empty:
        raise SystemExit("No rows after filter (dominance + sota_none + beta==0). Check ablation_tag in CSV.")

    tag = filt["ablation_tag"].fillna("").astype(str)
    if args.ablation_require == "canonical":
        filt = filt[tag.str.contains("__sota_none__", na=False)].copy()
    elif args.ablation_require == "legacy":
        filt = filt[~tag.str.contains("sota_", na=False)].copy()

    n_dup = int(
        filt.groupby(["dataset", "model", "seed", "_alpha"]).size().gt(1).sum()
    )
    filt = dedupe_one_row_per_run(filt)
    if n_dup:
        print(f"Deduped overlapping ablation paths for {n_dup} (dataset,model,seed,alpha) groups.")

    # Coverage warning (paper table uses 10 seeds × 5 α = 50 rows per model per dataset)
    for ds in DATASETS_PARETO:
        for model in MODELS_ORDER:
            sub = filt[(filt["dataset"] == ds) & (filt["model"] == model)]
            if sub.empty:
                continue
            ns = sub["seed"].nunique()
            if ns < 10:
                print(
                    f"Warning: {ds}/{model} has only {ns} distinct seeds in CSV after filters; "
                    "Pareto means will not match a full 10-seed table."
                )

    agg = aggregate_mean_seed(filt)
    print(
        "Pareto default: x=ExpDev (fairness cost), y=NDCG@10 (utility); "
        "upper-left is better (Fairlearn-style fairness–performance plots)."
    )
    for ds in DATASETS_PARETO:
        plot_pareto_dataset(
            agg,
            ds,
            out_dir,
            ndcg_on_x=args.pareto_ndcg_x,
            invert_fairness_axis=args.invert_fairness_axis,
        )

    hmap_cmap = args.heatmap_cmap.strip() or None
    for ds in DATASETS_PARETO:
        for a in (0.4, 0.8):
            heatmap_user_delta(filt, ds, a, out_dir, cmap_override=hmap_cmap)

    print(f"Wrote figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
