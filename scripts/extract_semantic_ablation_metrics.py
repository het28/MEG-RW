#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from cikm_eval.io import load_eval_inputs
from meg_rw.semantic import semantic_profiles_from_train_dataset


RUN_RE = re.compile(r"^alpha_(?P<alpha>[-+]?\d*\.?\d+)(?:_beta_(?P<beta>[-+]?\d*\.?\d+))?$")


def parse_run_label(run_label: str) -> tuple[float, float]:
    if run_label == "baseline":
        return 0.0, 0.0
    m = RUN_RE.match(run_label)
    if not m:
        return math.nan, math.nan
    alpha = float(m.group("alpha"))
    beta = float(m.group("beta")) if m.group("beta") is not None else 0.0
    return alpha, beta


def dataset_cfg_for_semantics(dataset_short: str) -> str:
    # Field mappings are identical across model cfgs for a dataset;
    # use lightgcn cfg as canonical.
    return f"config/cikm_{dataset_short}_lightgcn.yaml"


def recbole_dataset_name(dataset_short: str) -> str:
    # Map project shorthand to RecBole dataset identifiers used in configs.
    return {
        "ml1m": "ml-1m",
        "lastfm": "lastfm",
        "amazonbooks": "amazon-books",
    }.get(dataset_short, dataset_short)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def diagnostics_vs_baseline(cur_eval_inputs_path: Path, baseline_eval_inputs_path: Path) -> tuple[float | None, float | None]:
    if not cur_eval_inputs_path.exists() or not baseline_eval_inputs_path.exists():
        return None, None
    cur = load_eval_inputs(cur_eval_inputs_path)
    base = load_eval_inputs(baseline_eval_inputs_path)

    users = sorted(set(base.recommendations.keys()) & set(cur.recommendations.keys()))
    if not users:
        return None, None

    changed = 0
    jacc = []
    for u in users:
        a = list(map(int, base.recommendations[u]))
        b = list(map(int, cur.recommendations[u]))
        sa, sb = set(a), set(b)
        if sa != sb:
            changed += 1
        den = len(sa | sb)
        jacc.append((len(sa & sb) / den) if den > 0 else 1.0)
    return changed / len(users), float(np.mean(jacc))


def semantic_coherence(eval_inputs_path: Path, item_vec: np.ndarray, user_vec: np.ndarray) -> float | None:
    if not eval_inputs_path.exists():
        return None
    ev = load_eval_inputs(eval_inputs_path)
    vals = []
    for u, recs in ev.recommendations.items():
        uu = int(u)
        if uu < 0 or uu >= user_vec.shape[0]:
            continue
        uv = user_vec[uu]
        for i in recs:
            ii = int(i)
            if 0 <= ii < item_vec.shape[0]:
                vals.append(cosine(uv, item_vec[ii]))
    if not vals:
        return None
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", default="experiments/cikm2026")
    ap.add_argument("--output-csv", default="experiments/cikm2026/semantic_ablation_metrics.csv")
    ap.add_argument("--datasets", default="ml1m lastfm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem-dim", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.experiments_root)
    datasets = args.datasets.split()
    seed_dir = f"seed_{args.seed}"

    # Precompute semantic profiles per dataset once.
    sem_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds in datasets:
        cfgs = ["config/cikm_base.yaml", dataset_cfg_for_semantics(ds)]
        cfg = Config(
            model="LightGCN",
            dataset=recbole_dataset_name(ds),
            config_file_list=cfgs,
            config_dict={},
        )
        dset = create_dataset(cfg)
        train_data, _, _ = data_preparation(cfg, dset)
        item_v, user_v = semantic_profiles_from_train_dataset(train_data._dataset, sem_dim=args.sem_dim)
        sem_cache[ds] = (item_v, user_v)

    rows: list[dict[str, Any]] = []
    for ds in datasets:
        for model_dir in sorted((root / ds).glob("*")):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            run_root = model_dir / seed_dir
            if not run_root.exists():
                continue
            baseline_eval = run_root / "baseline" / "eval_inputs.json"
            item_v, user_v = sem_cache[ds]
            base_sem = semantic_coherence(baseline_eval, item_v, user_v) if baseline_eval.exists() else None

            for run_dir in sorted(run_root.glob("*")):
                if not run_dir.is_dir():
                    continue
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists():
                    continue

                run_label = run_dir.name
                alpha, beta = parse_run_label(run_label)
                obj = json.loads(metrics_path.read_text())
                tr = obj.get("test_result") or {}
                fa = obj.get("fairness_audit") or {}
                uf = fa.get("user_fairness") or {}
                it = fa.get("item_fairness") or {}
                td = fa.get("transfer_delta") or {}
                al = fa.get("alignment") or {}

                cur_eval = run_dir / "eval_inputs.json"
                changed_pct, jacc = diagnostics_vs_baseline(cur_eval, baseline_eval)
                sem_mean = semantic_coherence(cur_eval, item_v, user_v)
                sem_delta = None if sem_mean is None or base_sem is None else (sem_mean - base_sem)

                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "seed": args.seed,
                        "alpha": alpha,
                        "beta": beta,
                        "run_label": run_label,
                        "status": obj.get("status"),
                        "ndcg@10": tr.get("ndcg@10"),
                        "recall@10": tr.get("recall@10"),
                        "mrr@10": tr.get("mrr@10"),
                        "exposure_deviation": it.get("exposure_deviation"),
                        "exposure_ratio": it.get("exposure_ratio"),
                        "tail_ratio": it.get("tail_ratio"),
                        "tail_head_ratio": it.get("tail_head_ratio"),
                        "ndcg_gap_max_min": uf.get("ndcg_gap_max_min"),
                        "recall_gap_max_min": uf.get("recall_gap_max_min"),
                        "transfer_l1_shift": td.get("l1_shift"),
                        "transfer_max_abs_shift": td.get("max_abs_shift"),
                        "align_pearson": al.get("pearson_alignment"),
                        "align_spearman": al.get("spearman_alignment"),
                        "topk_changed_user_pct": changed_pct,
                        "topk_mean_jaccard_vs_baseline": jacc,
                        "semantic_topk_mean_cosine": sem_mean,
                        "semantic_topk_delta_vs_baseline": sem_delta,
                        "fairness_audit_error": fa.get("error"),
                    }
                )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
