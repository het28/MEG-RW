#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ALPHA_GRID = {0.0, 0.1, 0.2, 0.4, 0.8}


def _f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pick(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _group_ndcg(uf: dict[str, Any], group: str) -> float | None:
    # Common flattened variants
    direct = _pick(uf, f"ndcg_{group}", f"{group}_ndcg")
    if direct is not None:
        return _f(direct)
    # Nested variants
    for container_key in ("per_group", "group_metrics", "by_group", "groups", "ndcg_by_group"):
        c = uf.get(container_key)
        if isinstance(c, dict) and group in c:
            g = c[group]
            if isinstance(g, dict):
                return _f(_pick(g, "ndcg", "NDCG@10", "ndcg@10"))
            return _f(g)
    # Sometimes user_fairness is directly keyed by group name
    g = uf.get(group)
    if isinstance(g, dict):
        return _f(_pick(g, "ndcg", "NDCG@10", "ndcg@10"))
    return None


def _uf_scalar(uf: dict[str, Any], *keys: str) -> float | None:
    v = _pick(uf, *keys)
    if v is not None:
        return _f(v)
    # nested style in dataclass-json form
    for k in ("summary", "overall"):
        s = uf.get(k)
        if isinstance(s, dict):
            v2 = _pick(s, *keys)
            if v2 is not None:
                return _f(v2)
    return None


def _transfer_colsum(transfer: dict[str, Any], group_name: str) -> float | None:
    # preferred: explicit col_sums
    cs = transfer.get("col_sums")
    if isinstance(cs, dict):
        return _f(_pick(cs, group_name, group_name.lower(), group_name.replace("_", ""), group_name.replace("_", "-")))
    # fallback: compute from matrix + item_groups
    m = transfer.get("matrix")
    ig = transfer.get("item_groups")
    if isinstance(m, list) and isinstance(ig, list) and len(ig) > 0:
        try:
            idx = None
            aliases = {group_name, group_name.lower(), group_name.capitalize(), group_name.replace("_", "")}
            for j, g in enumerate(ig):
                if str(g) in aliases:
                    idx = j
                    break
            if idx is None:
                return None
            arr = [float(row[idx]) for row in m if isinstance(row, list) and len(row) > idx]
            return float(sum(arr))
        except Exception:
            return None
    return None


def _transfer_cell(tf: dict[str, Any], row: str, col: str) -> float | None:
    # Nested row->col
    if isinstance(tf.get(row), dict):
        return _f(_pick(tf[row], col, col.replace("_", ""), col.replace("_", "-")))
    # Flattened row_col
    flat_key = f"{row}_{col}"
    v = _f(_pick(tf, flat_key, flat_key.replace("_mid", "mid")))
    if v is not None:
        return v
    # Matrix style: transfer = {user_groups, item_groups, matrix}
    ug = tf.get("user_groups")
    ig = tf.get("item_groups")
    m = tf.get("matrix")
    if isinstance(ug, list) and isinstance(ig, list) and isinstance(m, list):
        try:
            row_alias = {row, row.lower(), row.capitalize()}
            col_alias = {col, col.lower(), col.capitalize(), col.replace("_", ""), col.replace("_", "-")}
            ri = next((i for i, x in enumerate(ug) if str(x) in row_alias), None)
            ci = next((j for j, x in enumerate(ig) if str(x) in col_alias), None)
            if ri is not None and ci is not None and ri < len(m) and ci < len(m[ri]):
                return _f(m[ri][ci])
        except Exception:
            return None
    return None


def parse_alpha(run_label: str) -> float | None:
    if run_label == "baseline" or run_label.startswith("baseline__"):
        return 0.0
    if not run_label.startswith("alpha_"):
        return None
    # supports alpha_0.4 and alpha_0.4_beta_0.3
    try:
        val = run_label.split("alpha_", 1)[1].split("_", 1)[0]
        return float(val)
    except Exception:
        return None


def collect_runs(exp_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in exp_root.rglob("metrics.json"):
        try:
            obj = json.loads(metrics_path.read_text())
        except Exception:
            continue
        if obj.get("status") != "success":
            continue

        parts = metrics_path.parts
        if "seed_" not in "".join(parts):
            # best-effort structural guard; continue on unknown layouts
            pass
        run_dir = metrics_path.parent
        seed_dir = run_dir.parent
        model_dir = seed_dir.parent
        dataset_dir = model_dir.parent

        run_label = run_dir.name
        alpha = parse_alpha(run_label)
        if alpha is None:
            continue

        tr = obj.get("test_result") or {}
        fa = obj.get("fairness_audit") or {}
        md = fa.get("metadata") or {}
        it = fa.get("item_fairness") or {}
        uf = fa.get("user_fairness") or {}
        td = fa.get("transfer_delta") or {}
        tf = fa.get("transfer") or {}
        al = fa.get("alignment") or {}
        # dataclass-json style user_fairness uses group_ndcg dict
        gndcg = uf.get("group_ndcg") if isinstance(uf.get("group_ndcg"), dict) else {}

        rows.append(
            {
                "experiments_root": str(exp_root),
                "dataset": dataset_dir.name,
                "model": model_dir.name,
                "seed": int(seed_dir.name.replace("seed_", "")),
                "run_label": run_label,
                "alpha": alpha,
                "sota_method": str(md.get("sota_method", "none")),
                "sota_lambda": _f(md.get("sota_lambda")),
                "ndcg@10": _f(tr.get("ndcg@10")),
                "recall@10": _f(tr.get("recall@10")),
                "mrr@10": _f(tr.get("mrr@10")),
                "expdev": _f(it.get("exposure_deviation")),
                "tail_ratio": _f(it.get("tail_ratio")),
                "tail_head_ratio": _f(it.get("tail_head_ratio")),
                "exposure_ratio": _f(it.get("exposure_ratio")),
                "ndcg_gap_max_min": _uf_scalar(uf, "ndcg_gap_max_min"),
                "ndcg_std": _uf_scalar(uf, "ndcg_std"),
                "worst_group_ndcg": _uf_scalar(uf, "worst_group_ndcg"),
                "ndcg_niche": _f(_pick(gndcg, "niche")) if gndcg else _group_ndcg(uf, "niche"),
                "ndcg_semi_niche": _f(_pick(gndcg, "semi_niche")) if gndcg else _group_ndcg(uf, "semi_niche"),
                "ndcg_semi_mainstream": _f(_pick(gndcg, "semi_mainstream")) if gndcg else _group_ndcg(uf, "semi_mainstream"),
                "ndcg_mainstream": _f(_pick(gndcg, "mainstream")) if gndcg else _group_ndcg(uf, "mainstream"),
                "transfer_l1_shift": _f(td.get("l1_shift")),
                "transfer_max_abs_shift": _f(td.get("max_abs_shift")),
                "align_pearson": _f(al.get("pearson_alignment")),
                "align_spearman": _f(al.get("spearman_alignment")),
                # item-group exposure shares from transfer col sums (row-normalized transfer)
                "exp_head": _transfer_colsum(tf if isinstance(tf, dict) else {}, "Head"),
                "exp_upper_mid": _transfer_colsum(tf if isinstance(tf, dict) else {}, "UpperMid"),
                "exp_lower_mid": _transfer_colsum(tf if isinstance(tf, dict) else {}, "LowerMid"),
                "exp_tail": _transfer_colsum(tf if isinstance(tf, dict) else {}, "Tail"),
                "transfer_niche_head": _transfer_cell(tf if isinstance(tf, dict) else {}, "niche", "head"),
                "transfer_niche_upper_mid": _transfer_cell(tf if isinstance(tf, dict) else {}, "niche", "upper_mid"),
                "transfer_niche_lower_mid": _transfer_cell(tf if isinstance(tf, dict) else {}, "niche", "lower_mid"),
                "transfer_niche_tail": _transfer_cell(tf if isinstance(tf, dict) else {}, "niche", "tail"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MEG-RW paper tables from run artifacts")
    ap.add_argument("--experiments-root", default="experiments/cikm2026")
    ap.add_argument(
        "--experiments-roots",
        default="",
        help="Comma-separated experiment roots to merge (overrides --experiments-root)",
    )
    ap.add_argument("--out-dir", default="experiments/cikm2026/paper_tables")
    ap.add_argument("--datasets", default="ml1m,lastfm")
    ap.add_argument("--models", default="weighted_lightgcn,weighted_ngcf,weighted_bpr,weighted_itemknn,weighted_neumf")
    ap.add_argument("--alphas", default="0.0,0.1,0.2,0.4,0.8")
    args = ap.parse_args()

    roots = [x.strip() for x in str(args.experiments_roots).split(",") if x.strip()]
    if roots:
        exp_roots = [Path(x) for x in roots]
    else:
        exp_roots = [Path(args.experiments_root)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    models = {x.strip() for x in args.models.split(",") if x.strip()}
    alphas = {float(x.strip()) for x in args.alphas.split(",") if x.strip()}

    frames = [collect_runs(r) for r in exp_roots]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No successful runs found.")
    df = pd.concat(frames, ignore_index=True)

    df = df[df["dataset"].isin(datasets) & df["model"].isin(models) & df["alpha"].isin(alphas)].copy()
    if df.empty:
        raise SystemExit("No runs match selected datasets/models/alphas.")

    # 1) final cleaned per-run table
    cleaned_cols = [
        "experiments_root",
        "dataset",
        "model",
        "seed",
        "run_label",
        "alpha",
        "sota_method",
        "sota_lambda",
        "ndcg@10",
        "expdev",
        "tail_ratio",
        "ndcg_niche",
        "ndcg_semi_niche",
        "ndcg_semi_mainstream",
        "ndcg_mainstream",
        "ndcg_gap_max_min",
        "ndcg_std",
        "worst_group_ndcg",
        "align_pearson",
        "align_spearman",
        "exp_head",
        "exp_upper_mid",
        "exp_lower_mid",
        "exp_tail",
        "transfer_l1_shift",
        "transfer_max_abs_shift",
        "transfer_niche_head",
        "transfer_niche_upper_mid",
        "transfer_niche_lower_mid",
        "transfer_niche_tail",
    ]
    df[cleaned_cols].sort_values(["dataset", "model", "sota_method", "alpha", "seed"]).to_csv(
        out_dir / "paper_runs_cleaned.csv", index=False
    )

    # 2) multi-seed aggregated mean±std for key metrics (+ user-side + transfer)
    agg = (
        df.groupby(["experiments_root", "dataset", "model", "sota_method", "sota_lambda", "alpha"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            ndcg_mean=("ndcg@10", "mean"),
            ndcg_std=("ndcg@10", "std"),
            expdev_mean=("expdev", "mean"),
            expdev_std=("expdev", "std"),
            tail_ratio_mean=("tail_ratio", "mean"),
            tail_ratio_std=("tail_ratio", "std"),
            ndcg_niche_mean=("ndcg_niche", "mean"),
            ndcg_semi_niche_mean=("ndcg_semi_niche", "mean"),
            ndcg_semi_mainstream_mean=("ndcg_semi_mainstream", "mean"),
            ndcg_mainstream_mean=("ndcg_mainstream", "mean"),
            ndcg_gap_max_min_mean=("ndcg_gap_max_min", "mean"),
            ndcg_gap_max_min_std=("ndcg_gap_max_min", "std"),
            ndcg_user_std_mean=("ndcg_std", "mean"),
            worst_group_ndcg_mean=("worst_group_ndcg", "mean"),
            align_pearson_mean=("align_pearson", "mean"),
            align_spearman_mean=("align_spearman", "mean"),
            exp_head_mean=("exp_head", "mean"),
            exp_upper_mid_mean=("exp_upper_mid", "mean"),
            exp_lower_mid_mean=("exp_lower_mid", "mean"),
            exp_tail_mean=("exp_tail", "mean"),
            transfer_l1_shift_mean=("transfer_l1_shift", "mean"),
            transfer_l1_shift_std=("transfer_l1_shift", "std"),
            transfer_max_abs_shift_mean=("transfer_max_abs_shift", "mean"),
            transfer_max_abs_shift_std=("transfer_max_abs_shift", "std"),
            transfer_niche_head_mean=("transfer_niche_head", "mean"),
            transfer_niche_upper_mid_mean=("transfer_niche_upper_mid", "mean"),
            transfer_niche_lower_mid_mean=("transfer_niche_lower_mid", "mean"),
            transfer_niche_tail_mean=("transfer_niche_tail", "mean"),
        )
        .sort_values(["dataset", "model", "sota_method", "alpha"])
    )
    agg.to_csv(out_dir / "paper_aggregated_alpha.csv", index=False)

    # 3) best alpha per model for utility and fairness
    utility_idx = agg.groupby(["experiments_root", "dataset", "model", "sota_method", "sota_lambda"])["ndcg_mean"].idxmax()
    fair_idx = agg.groupby(["experiments_root", "dataset", "model", "sota_method", "sota_lambda"])["expdev_mean"].idxmin()
    util = agg.loc[utility_idx, ["experiments_root", "dataset", "model", "sota_method", "sota_lambda", "alpha", "ndcg_mean", "expdev_mean"]].rename(
        columns={"alpha": "best_utility_alpha", "ndcg_mean": "best_utility_ndcg", "expdev_mean": "best_utility_expdev"}
    )
    fair = agg.loc[fair_idx, ["experiments_root", "dataset", "model", "sota_method", "sota_lambda", "alpha", "ndcg_mean", "expdev_mean"]].rename(
        columns={"alpha": "best_fairness_alpha", "ndcg_mean": "best_fairness_ndcg", "expdev_mean": "best_fairness_expdev"}
    )
    best = util.merge(
        fair,
        on=["experiments_root", "dataset", "model", "sota_method", "sota_lambda"],
        how="inner",
    ).sort_values(["dataset", "model", "sota_method"])
    best.to_csv(out_dir / "paper_best_alpha.csv", index=False)

    # 4) compact table for baseline + alpha grid
    compact = agg[
        [
            "experiments_root",
            "dataset",
            "model",
            "sota_method",
            "sota_lambda",
            "alpha",
            "ndcg_mean",
            "ndcg_std",
            "expdev_mean",
            "expdev_std",
            "tail_ratio_mean",
            "tail_ratio_std",
        ]
    ].copy()
    compact.to_csv(out_dir / "paper_compact_table.csv", index=False)

    # 5) per-group exposure table
    exp_cols = [
        "experiments_root",
        "dataset",
        "model",
        "seed",
        "alpha",
        "run_label",
        "sota_method",
        "sota_lambda",
        "exp_head",
        "exp_upper_mid",
        "exp_lower_mid",
        "exp_tail",
    ]
    df[exp_cols].sort_values(["dataset", "model", "sota_method", "alpha", "seed"]).to_csv(
        out_dir / "paper_exposure_per_group.csv", index=False
    )

    # 6) full transfer long table (niche row currently explicit; can expand later)
    transfer_cols = [
        "dataset",
        "model",
        "seed",
        "alpha",
        "run_label",
        "sota_method",
        "sota_lambda",
        "transfer_niche_head",
        "transfer_niche_upper_mid",
        "transfer_niche_lower_mid",
        "transfer_niche_tail",
        "transfer_l1_shift",
        "transfer_max_abs_shift",
    ]
    df[transfer_cols].sort_values(["dataset", "model", "sota_method", "alpha", "seed"]).to_csv(
        out_dir / "paper_transfer_summary.csv", index=False
    )

    # 7) delta vs baseline per seed for selected metrics
    delta_rows: list[dict[str, Any]] = []
    delta_metrics = [
        "ndcg@10",
        "expdev",
        "tail_ratio",
        "ndcg_gap_max_min",
        "ndcg_std",
        "worst_group_ndcg",
        "exp_head",
        "exp_upper_mid",
        "exp_lower_mid",
        "exp_tail",
    ]
    for (ds, model, sm, sl, seed), g in df.groupby(["dataset", "model", "sota_method", "sota_lambda", "seed"]):
        base = g[g["alpha"] == 0.0]
        if base.empty:
            continue
        b = base.iloc[-1]
        for _, r in g[g["alpha"] != 0.0].iterrows():
            row = {
                "dataset": ds,
                "model": model,
                "sota_method": sm,
                "sota_lambda": sl,
                "seed": int(seed),
                "alpha": float(r["alpha"]),
                "run_label": r["run_label"],
            }
            for m in delta_metrics:
                bv = _f(b.get(m))
                rv = _f(r.get(m))
                row[f"delta_{m}"] = None if bv is None or rv is None else (rv - bv)
            delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        delta_df = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "sota_method",
                "sota_lambda",
                "seed",
                "alpha",
                "run_label",
            ]
        )
    else:
        delta_df = delta_df.sort_values(["dataset", "model", "sota_method", "seed", "alpha"])
    delta_df.to_csv(out_dir / "paper_deltas_vs_baseline.csv", index=False)

    # 8) structured run JSONL
    jsonl_path = out_dir / "paper_runs_structured.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, r in df.sort_values(["dataset", "model", "sota_method", "seed", "alpha"]).iterrows():
            obj = {
                "experiments_root": r["experiments_root"],
                "dataset": r["dataset"],
                "model": r["model"],
                "sota_method": r["sota_method"],
                "sota_lambda": _f(r.get("sota_lambda")),
                "seed": int(r["seed"]),
                "run_label": r["run_label"],
                "alpha": float(r["alpha"]),
                "metrics": {
                    "ndcg@10": _f(r.get("ndcg@10")),
                    "expdev": _f(r.get("expdev")),
                    "tail_ratio": _f(r.get("tail_ratio")),
                    "ndcg_gap_max_min": _f(r.get("ndcg_gap_max_min")),
                    "ndcg_std": _f(r.get("ndcg_std")),
                    "worst_group_ndcg": _f(r.get("worst_group_ndcg")),
                    "transfer_l1_shift": _f(r.get("transfer_l1_shift")),
                    "transfer_max_abs_shift": _f(r.get("transfer_max_abs_shift")),
                },
                "exp_per_group": {
                    "Head": _f(r.get("exp_head")),
                    "UpperMid": _f(r.get("exp_upper_mid")),
                    "LowerMid": _f(r.get("exp_lower_mid")),
                    "Tail": _f(r.get("exp_tail")),
                },
                "ndcg_per_user_group": {
                    "niche": _f(r.get("ndcg_niche")),
                    "semi_niche": _f(r.get("ndcg_semi_niche")),
                    "semi_mainstream": _f(r.get("ndcg_semi_mainstream")),
                    "mainstream": _f(r.get("ndcg_mainstream")),
                },
                "transfer_niche_row": {
                    "Head": _f(r.get("transfer_niche_head")),
                    "UpperMid": _f(r.get("transfer_niche_upper_mid")),
                    "LowerMid": _f(r.get("transfer_niche_lower_mid")),
                    "Tail": _f(r.get("transfer_niche_tail")),
                },
            }
            f.write(json.dumps(obj) + "\n")

    print(f"Wrote: {out_dir / 'paper_runs_cleaned.csv'}")
    print(f"Wrote: {out_dir / 'paper_aggregated_alpha.csv'}")
    print(f"Wrote: {out_dir / 'paper_best_alpha.csv'}")
    print(f"Wrote: {out_dir / 'paper_compact_table.csv'}")
    print(f"Wrote: {out_dir / 'paper_exposure_per_group.csv'}")
    print(f"Wrote: {out_dir / 'paper_transfer_summary.csv'}")
    print(f"Wrote: {out_dir / 'paper_deltas_vs_baseline.csv'}")
    print(f"Wrote: {out_dir / 'paper_runs_structured.jsonl'}")


if __name__ == "__main__":
    main()
