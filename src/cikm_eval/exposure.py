"""Item-side exposure fairness vs catalog (from EvalInputs only)."""

from __future__ import annotations

from collections import Counter, defaultdict

from cikm_eval.types import EvalInputs, GroupName, ordered_item_groups


def catalog_share_from_item_map(item_group: dict[int, GroupName]) -> dict[GroupName, float]:
    c = Counter(item_group.values())
    n = sum(c.values())
    if n <= 0:
        return {}
    return {g: c[g] / n for g in ordered_item_groups(set(c.keys()))}


def compute_item_fairness_metrics(inputs: EvalInputs) -> dict[str, float]:
    """
    Returns stable scalar keys for logging:

    - ``exposure_deviation``: mean abs deviation of global exposure share from catalog share
    - ``exposure_ratio``: min group exposure / max group exposure (disparity)
    - ``tail_ratio``: global fraction of top-k slots exposing Tail items
    - ``tail_head_ratio``: tail exposure / head exposure (Head = first in ITEM_GROUP_ORDER)
    """
    seen_items: set[GroupName] = set()
    for i in inputs.recommendations.values():
        for j in i:
            seen_items.add(inputs.item_group[j])

    catalog = catalog_share_from_item_map(inputs.item_group)
    exp_mass: dict[GroupName, float] = defaultdict(float)
    total_slots = 0

    for u, recs in inputs.recommendations.items():
        for it in recs[: inputs.k]:
            g = inputs.item_group[it]
            exp_mass[g] += 1.0
            total_slots += 1

    if total_slots <= 0:
        return {
            "exposure_deviation": 0.0,
            "exposure_ratio": 0.0,
            "tail_ratio": 0.0,
            "tail_head_ratio": 0.0,
        }

    exp_share = {g: exp_mass[g] / total_slots for g in exp_mass}
    all_g = ordered_item_groups(set(catalog.keys()) | set(exp_share.keys()))
    cat_vec = [catalog.get(g, 0.0) for g in all_g]
    exp_vec = [exp_share.get(g, 0.0) for g in all_g]
    mad = sum(abs(exp_vec[i] - cat_vec[i]) for i in range(len(all_g))) / max(
        len(all_g), 1
    )

    ev = list(exp_share.values())
    exposure_ratio = min(ev) / max(ev) if ev else 0.0

    # For canonical 4-group setup use Tail/Head; for dynamic Gxx groups use last/first.
    tail_g: GroupName = "Tail" if "Tail" in exp_share else (all_g[-1] if all_g else "Tail")
    head_g: GroupName = "Head" if "Head" in exp_share else (all_g[0] if all_g else "Head")
    tail_ratio = exp_share.get(tail_g, 0.0)
    tail_head = tail_ratio / (exp_share.get(head_g, 0.0) + 1e-12)

    return {
        "exposure_deviation": float(mad),
        "exposure_ratio": float(exposure_ratio),
        "tail_ratio": float(tail_ratio),
        "tail_head_ratio": float(tail_head),
    }
