# RecBole integration (MEG-RW)

## What RecBole gives you

- **Datasets:** MovieLens-1M, Amazon reviews, and many others via `recbole` dataset URLs + atomic `.inter` format.
- **Models:** `BPR`, `LightGCN`, `NGCF` with shared config (YAML) and `run_recbole`.
- **Splits / negatives:** configurable; align with the research plan (80/10/10 temporal vs LOO).

## What we add

### 1. Weighted graph for LightGCN / NGCF

RecBole’s `LightGCN.get_norm_adj_mat()` builds an unweighted bipartite adjacency (all ones). **MEG-RW** needs \(\tilde w_{ui}>0\) on each train edge.

**Approach:** `WeightedLightGCN` subclasses `LightGCN`, overrides `get_norm_adj_mat()` to:

- Use `self.interaction_matrix` as SciPy COO **with `data` = edge weights** (symmetric user→item and item→user copies share the same weight).
- Compute \( \hat A = D^{-1/2} A D^{-1/2} \) with **weighted** row sums in \(D\).

**Feeding weights:** After `Dataset` loads, replace the training COO:

```python
import scipy.sparse as sp
# train_mat: sp.coo_matrix, shape (n_users, n_items), nnz = train edges
train_mat.data[:] = meg_rw_weights_aligned.astype(np.float32)
```

Order of `(row, col)` **must** match `dataset.inter_matrix(form="coo")` exactly. Easiest: compute MEG-RW from the same COO rows/cols RecBole emits.

### 2. BPR-MF

Weighted BPR is not the same as weighted adjacency. Options:

- **Sample positives** proportional to \(\tilde w_{ui}\) (custom sampler), or
- **Scale per-pair loss** by \(\tilde w_{ui}\) in a thin `Trainer` subclass.

RecBole may expose `weight` in interactions for some losses — verify for your version; otherwise implement explicit loss weighting.

### 3. NGCF

Same pattern as LightGCN: subclass and build weighted normalized adjacency (copy `NGCF` graph construction, replace ones with `inter_matrix.data`).

## Fairness metrics

RecBole metrics are mostly **global** NDCG/Recall. **Item/user groups, transfer matrix, Pareto over α** live in `src/cikm_eval/` (see `fairness_audit.run_fairness_audit`).

## Baselines (FDA, FA4GCF)

Treat as **separate** code paths or repos; normalize to the same `.inter` splits and our eval module for apples-to-apples fairness tables.
