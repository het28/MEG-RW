"""Post-hoc reranking baselines for popularity fairness."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rerank_topk(
    *,
    method: str,
    scores: np.ndarray,
    candidate_items: Sequence[int],
    topk: int,
    item_groups: np.ndarray,
    item_popularity: np.ndarray,
    lam: float,
) -> list[int]:
    """Return top-k item ids after a configurable reranking policy.

    Supported methods:
    - none: no reranking
    - pop_inverse: additive boost for less-popular items
    - head_penalty: subtract popularity-based penalty
    - xquad_pop: xQuAD-style group coverage reranking
    - mmr_pop: MMR-style diversity reranking over popularity groups
    - calib_pop: calibration to a target popularity-group distribution
    """
    m = (method or "none").strip().lower()
    if topk <= 0:
        return []

    cand = np.asarray(candidate_items, dtype=np.int64)
    if cand.size == 0:
        return []

    k = int(min(topk, cand.size))
    s = scores[cand].astype(np.float64, copy=False)

    if m == "none":
        order = np.argsort(-s, kind="mergesort")
        return cand[order[:k]].astype(np.int64).tolist()

    if m == "pop_inverse":
        # IPS-like popularity correction in score space.
        adj = s + float(lam) * (1.0 - item_popularity[cand])
        order = np.argsort(-adj, kind="mergesort")
        return cand[order[:k]].astype(np.int64).tolist()

    if m == "head_penalty":
        # Popularity-aware penalty favoring tail items.
        adj = s - float(lam) * np.log1p(item_popularity[cand] * 1000.0)
        order = np.argsort(-adj, kind="mergesort")
        return cand[order[:k]].astype(np.int64).tolist()

    if m == "xquad_pop":
        return _xquad_pop(
            s=s, cand=cand, k=k, groups=item_groups[cand], lam=float(lam)
        )

    if m == "mmr_pop":
        return _mmr_pop(
            s=s, cand=cand, k=k, groups=item_groups[cand], lam=float(lam)
        )

    if m == "calib_pop":
        return _calib_pop(
            s=s,
            cand=cand,
            k=k,
            groups=item_groups[cand],
            item_groups=item_groups,
            lam=float(lam),
        )

    # Fallback: treat unknown method as "none" to avoid crashing long sweeps.
    order = np.argsort(-s, kind="mergesort")
    return cand[order[:k]].astype(np.int64).tolist()


def _xquad_pop(
    *, s: np.ndarray, cand: np.ndarray, k: int, groups: np.ndarray, lam: float
) -> list[int]:
    uniq = np.unique(groups)
    g2i = {int(g): i for i, g in enumerate(uniq)}
    covered = np.zeros(len(uniq), dtype=np.float64)
    selected: list[int] = []
    selected_mask = np.zeros(cand.size, dtype=bool)

    for _ in range(k):
        best_idx = -1
        best_val = -1e30
        for i in range(cand.size):
            if selected_mask[i]:
                continue
            gi = g2i[int(groups[i])]
            novelty = 1.0 / (1.0 + covered[gi])
            val = (1.0 - lam) * s[i] + lam * novelty
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected_mask[best_idx] = True
        selected.append(int(cand[best_idx]))
        covered[g2i[int(groups[best_idx])]] += 1.0
    return selected


def _mmr_pop(
    *, s: np.ndarray, cand: np.ndarray, k: int, groups: np.ndarray, lam: float
) -> list[int]:
    selected: list[int] = []
    selected_idx: list[int] = []
    selected_mask = np.zeros(cand.size, dtype=bool)

    for _ in range(k):
        best_idx = -1
        best_val = -1e30
        for i in range(cand.size):
            if selected_mask[i]:
                continue
            # Group-level dissimilarity proxy.
            if selected_idx:
                sim = max(1.0 if groups[i] == groups[j] else 0.0 for j in selected_idx)
            else:
                sim = 0.0
            val = (1.0 - lam) * s[i] - lam * sim
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected_mask[best_idx] = True
        selected_idx.append(best_idx)
        selected.append(int(cand[best_idx]))
    return selected


def _calib_pop(
    *,
    s: np.ndarray,
    cand: np.ndarray,
    k: int,
    groups: np.ndarray,
    item_groups: np.ndarray,
    lam: float,
) -> list[int]:
    # Target = training-group prior (calibration baseline).
    n_groups = int(item_groups.max()) + 1 if item_groups.size else 1
    prior = np.bincount(item_groups.astype(np.int64), minlength=n_groups).astype(np.float64)
    prior = prior / max(prior.sum(), 1.0)

    counts = np.zeros(n_groups, dtype=np.float64)
    selected: list[int] = []
    selected_mask = np.zeros(cand.size, dtype=bool)

    for t in range(k):
        best_idx = -1
        best_val = -1e30
        denom = float(t + 1)
        for i in range(cand.size):
            if selected_mask[i]:
                continue
            gi = int(groups[i])
            curr = counts / max(float(t), 1.0) if t > 0 else np.zeros_like(counts)
            next_dist = curr.copy()
            next_dist[gi] = (counts[gi] + 1.0) / denom
            calib_gain = -np.abs(next_dist - prior).sum()
            val = (1.0 - lam) * s[i] + lam * calib_gain
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected_mask[best_idx] = True
        gi = int(groups[best_idx])
        counts[gi] += 1.0
        selected.append(int(cand[best_idx]))
    return selected
