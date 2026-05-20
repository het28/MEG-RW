#!/usr/bin/env python3
"""
Flatten every successful (or all) metrics.json under experiment roots into one CSV,
similar spirit to experiments/cikm2026_SOTA/sota_baseline_only_metrics_fixed.csv.

Run on the server (repo root, venv activated):

  export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
  python scripts/export_all_runs_metrics_csv.py \\
    --roots experiments/cikm2026,experiments/cikm2026_SOTA \\
    --out experiments/cikm2026/all_runs_metrics_export.csv

Then scp only the CSV to your Mac if you like.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

KNOWN_DATASETS = {"ml1m", "lastfm", "amazonbooks", "ml-1m"}


def _f(x: Any) -> str:
    try:
        if x is None:
            return ""
        v = float(x)
        if v != v:  # nan
            return ""
        return f"{v:.12g}"
    except Exception:
        return ""


def _s(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def parse_run_path(metrics_path: Path) -> dict[str, str]:
    """Infer ablation_tag, dataset, model, seed, run_label.

    With ablation:
      .../cikm2026[_SOTA]/<rwmode_*__...>/<dataset>/<model>/seed_<n>/<run>/metrics.json
    Legacy (no ablation folder):
      .../cikm2026/<dataset>/<model>/seed_<n>/<run>/metrics.json
    """
    run_dir = metrics_path.parent
    run_label = run_dir.name
    seed_dir = run_dir.parent
    model_dir = seed_dir.parent
    dataset_dir = model_dir.parent
    parent_of_dataset = dataset_dir.parent

    ds_norm = dataset_dir.name.replace("-", "").lower()
    known = {k.replace("-", "").lower() for k in KNOWN_DATASETS}

    if parent_of_dataset.name.startswith("rwmode_") or parent_of_dataset.name.startswith("fracs_"):
        ablation_tag = parent_of_dataset.name
        dataset = dataset_dir.name
    elif ds_norm in known:
        ablation_tag = ""
        dataset = dataset_dir.name
    else:
        # Fallback: treat unknown middle segment as ablation tag
        ablation_tag = parent_of_dataset.name
        dataset = dataset_dir.name

    model = model_dir.name
    seed = seed_dir.name.replace("seed_", "")

    return {
        "ablation_tag": ablation_tag,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "run_label": run_label,
        "run_dir": str(run_dir.resolve()),
    }


def row_from_metrics(metrics_path: Path, obj: dict[str, Any]) -> dict[str, str]:
    loc = parse_run_path(metrics_path)
    tr = obj.get("test_result") or {}
    fa = obj.get("fairness_audit") or {}
    if isinstance(fa, dict) and fa.get("error"):
        # still export row with error string
        md = {}
        it = {}
        uf: dict[str, Any] = {}
        td = {}
    else:
        md = fa.get("metadata") or {}
        it = fa.get("item_fairness") or {}
        uf = fa.get("user_fairness") or {}
        td = fa.get("transfer_delta") or {}

    gndcg = uf.get("group_ndcg") if isinstance(uf.get("group_ndcg"), dict) else {}

    exp_id = _s(obj.get("experiment_id") or md.get("experiment_id"))
    alpha_cell = _s(md.get("alpha", ""))

    return {
        "experiment_id": exp_id,
        "ablation_tag": loc["ablation_tag"],
        "dataset": loc["dataset"],
        "model": loc["model"],
        "seed": loc["seed"],
        "run_label": loc["run_label"],
        "run_dir": loc["run_dir"],
        "status": _s(obj.get("status")),
        "started_at": _s(obj.get("started_at")),
        "finished_at": _s(obj.get("finished_at")),
        "sota_method": _s(md.get("sota_method", "none")),
        "sota_lambda": _s(md.get("sota_lambda", "")),
        "alpha_cell": alpha_cell,
        "meg_rw_alpha": _s(md.get("meg_rw_alpha", "")),
        "ndcg@10": _f(tr.get("ndcg@10")),
        "recall@10": _f(tr.get("recall@10")),
        "mrr@10": _f(tr.get("mrr@10")),
        "expdev": _f(it.get("exposure_deviation")),
        "tail_ratio": _f(it.get("tail_ratio")),
        "tail_head_ratio": _f(it.get("tail_head_ratio")),
        "exposure_ratio": _f(it.get("exposure_ratio")),
        "ndcg_gap_max_min": _f(uf.get("ndcg_gap_max_min")),
        "ndcg_std": _f(uf.get("ndcg_std")),
        "worst_group_ndcg": _f(uf.get("worst_group_ndcg")),
        "ndcg_niche": _f(gndcg.get("niche")),
        "ndcg_semi_niche": _f(gndcg.get("semi_niche")),
        "ndcg_semi_mainstream": _f(gndcg.get("semi_mainstream")),
        "ndcg_mainstream": _f(gndcg.get("mainstream")),
        "transfer_l1_shift": _f(td.get("l1_shift")),
        "transfer_max_abs_shift": _f(td.get("max_abs_shift")),
        "fairness_error": _s(fa.get("error")) if isinstance(fa, dict) else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export flat CSV from all metrics.json under experiment roots")
    ap.add_argument(
        "--roots",
        default="experiments/cikm2026,experiments/cikm2026_SOTA",
        help="Comma-separated roots (relative to cwd or absolute)",
    )
    ap.add_argument("--out", default="experiments/cikm2026/all_runs_metrics_export.csv")
    ap.add_argument(
        "--only-success",
        action="store_true",
        help="Skip rows where status != success",
    )
    args = ap.parse_args()

    roots = [Path(x.strip()) for x in str(args.roots).split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "ablation_tag",
        "dataset",
        "model",
        "seed",
        "run_label",
        "run_dir",
        "status",
        "started_at",
        "finished_at",
        "sota_method",
        "sota_lambda",
        "alpha_cell",
        "meg_rw_alpha",
        "ndcg@10",
        "recall@10",
        "mrr@10",
        "expdev",
        "tail_ratio",
        "tail_head_ratio",
        "exposure_ratio",
        "ndcg_gap_max_min",
        "ndcg_std",
        "worst_group_ndcg",
        "ndcg_niche",
        "ndcg_semi_niche",
        "ndcg_semi_mainstream",
        "ndcg_mainstream",
        "transfer_l1_shift",
        "transfer_max_abs_shift",
        "fairness_error",
    ]

    rows: list[dict[str, str]] = []
    seen_run_dir: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for metrics_path in sorted(root.rglob("metrics.json")):
            try:
                obj = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if args.only_success and str(obj.get("status", "")).lower() != "success":
                continue
            rd = str(metrics_path.parent.resolve())
            if rd in seen_run_dir:
                continue
            seen_run_dir.add(rd)
            rows.append(row_from_metrics(metrics_path, obj))

    rows.sort(key=lambda r: (r["ablation_tag"], r["dataset"], r["model"], int(r["seed"] or 0), r["run_label"]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")
    print(f"Roots: {[str(r.resolve()) for r in roots if r.is_dir()]}")


if __name__ == "__main__":
    main()
