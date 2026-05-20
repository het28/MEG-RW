# MEG-RW: Formal Research Plan (CIKM 2026)

**Working title:** *Before the Recommender Learns: Dual-Sided Multi-Group Fairness via User–Item Graph Preprocessing*

**Method name:** **MEG-RW** — Multi-group Exposure-calibrated Graph ReWeighting

**Design principle:** **Simple method, rich evaluation**—one preprocessing intervention (item-side structural reweighting); **no** user-side loss. All user-side and cross-side structure is an **audit** aligned with joint fairness–relevance and multi-sided fairness trends in the literature.

**Scope:** Intervene **only** before training; evaluate **item-side exposure** and **user-side utility** jointly, plus **cross-side** transfer and consistency.

---

## 1. Problem statement & paper twist

Let \(\mathcal{U}\), \(\mathcal{I}\) be user and item sets; \(\mathcal{E}\) implicit positives; \(G = (\mathcal{U}\cup\mathcal{I}, \mathcal{E})\) bipartite. **Task:** top-\(K\) recommendation.

**Item-side (exposure fairness):** \(G\) groups \(\{\mathcal{I}_g\}_{g=1}^G\) by **training** popularity (§4). **User-side (utility fairness):** \(L\) groups \(\{\mathcal{U}_h\}_{h=1}^L\) by **training** mainstreamness (§5).

**Twist:** A **preprocessing-only** graph reweighting is **motivated and computed from item-side structure** (MEG-RW). Downstream objectives are **unchanged**. We **audit** how this reshapes **both** item exposure allocation **and** user-group utility—not a second debiasing objective for users.

**Novelty (positioning):** Fairness is **two-sided and multi-group**; we show **when** item-side correction **transfers**, **conflicts**, or **hides** in user-side metrics (transfer matrix + joint reporting).

---

## 2. Final research questions

| ID | Question |
|----|----------|
| **RQ1** | How does preprocessing the user–item graph affect **multi-group item-side exposure fairness** in top-\(K\) recommendation? |
| **RQ2** | How do item-side fairness-oriented preprocessing changes **propagate** to **user-side utility fairness** across **behavioral** user groups? |
| **RQ3** | Do item-side fairness gains and user-side fairness gains **align**, **conflict**, or **trade off** across datasets and recommenders? |
| **RQ4** | How **robust** are fairness–utility patterns across **seeds**, **group granularities** (item/user ablations), and **fairness target** definitions (§8)? |

---

## 3. Final hypotheses (testable)

| ID | Hypothesis |
|----|------------|
| **H1** | Raw user–item graphs yield **systematically imbalanced** exposure across item popularity groups (head dominates vs. tail). |
| **H2** | MEG-RW **reduces item-side exposure disparity** (vs. targets in §8) **without** changing the downstream recommendation **objective** (only input weights). |
| **H3** | **Niche** users (low mainstreamness) **benefit more** from tail-aware preprocessing than **mainstream** users (tail exposure / utility gains skew toward niche groups). |
| **H4** | Item-side and user-side fairness are **not perfectly aligned**: some settings improve exposure balance while **worst-group user utility** is flat or worse (honest conflict). |
| **H5** | Conclusions are **more stable** under **multi-group audits** and **Pareto-style** fairness–relevance reporting than under a **single** aggregate fairness number. |

---

## 4. Exact item groups (locked)

**Signal:** item interaction count \(d_i = |\{u : (u,i)\in\mathcal{E}_{\mathrm{train}}\}|\).

**Main partition (\(G=4\))**—by **descending** \(d_i\), **by count of items** (not equal mass per group):

| Label | Definition |
|--------|------------|
| **Head** | Top **10%** of items by \(d_i\) |
| **Upper-mid** | Next **20%** |
| **Lower-mid** | Next **30%** |
| **Tail** | Bottom **40%** |

**Rationale:** Heavy-tailed catalogs; small dominant head; broad tail—more meaningful than flat quartiles for exposure fairness.

**Ablations (item):**

- **\(G=2\):** head vs. tail (e.g. same 10% / bottom 40% merge middle, or median split—**fix one rule** in appendix).
- **\(G=5\):** very-head, head, mid, low, tail (define **exact percentage cuts** in appendix when reporting).

**Implementation note:** Sort items by \(d_i\); assign \(g(i)\) by rank bands on the **item** axis. MEG-RW uses these same \(g(i)\) for \(\phi(g)\).

---

## 5. Exact user groups (locked)

**Behavioral only in the main paper**—no demographics as **primary** groups (optional supplementary if labels are clean).

### 5.1 Mainstreamness (main user partition, \(L=4\))

Let \(\mathcal{I}_u = \{ i : (u,i)\in\mathcal{E}_{\mathrm{train}} \}\) (same as \(I_u\) in prose).

**Normalized popularity** (train only). Let \(d_i\) be train degrees. Define, for example, **min–max**:

\[
\tilde{p}(i) \;=\; \frac{d_i - d_{\min}}{d_{\max} - d_{\min} + \epsilon} \;\in [0,1],
\]

\(d_{\min}, d_{\max}\) over items with \(d_i>0\). (**Alternative** logged counts or rank percentile—**pick one** and freeze for all experiments.)

**Mainstreamness score:**

\[
M(u) \;=\; \frac{1}{|\mathcal{I}_u|} \sum_{i \in \mathcal{I}_u} \tilde{p}(i).
\]

**User strata:** sort users by \(M(u)\); split into **4 quantile buckets** (equal user counts):

1. **Niche** — lowest \(M(u)\) quartile  
2. **Semi-niche**  
3. **Semi-mainstream**  
4. **Mainstream** — highest \(M(u)\) quartile  

Denote user-group index \(h(u)\in\{1,\ldots,L\}\).

### 5.2 Secondary user partitions (ablations / appendix)

**A. Activity:** tertiles or quartiles of \(|\mathcal{I}_u|\) (train): low / medium / high activity.

**B. Diversity (optional):** e.g. entropy of group labels \(\{g(i): i\in\mathcal{I}_u\}\) or of \(\tilde{p}(i)\) bins → narrow / medium / broad taste.

---

## 6. Item-side target shares (evaluation targets)

Exposure fairness is defined **relative to a target** vector \((T_1,\ldots,T_G)\) over item groups.

### 6.1 Main target: **catalog-share parity**

\[
T_g^{\mathrm{cat}} \;=\; \frac{|\mathcal{I}_g|}{|\mathcal{I}|}.
\]

*Interpretation:* each group should receive exposure proportional to **how many items** it contains.

### 6.2 Secondary target: **relevance-aware parity** (defense vs. “forcing irrelevant tail”)

Using **test** relevance only for **evaluation** (not for MEG-RW weights): let \(R_g\) be the **fraction of test positive interactions** that involve items in \(\mathcal{I}_g\) (aggregated over users, or per-user normalized—**choose one** and document).

\[
T_g^{\mathrm{rel}} \;=\; \frac{R_g}{\sum_{g'} R_{g'}}.
\]

Report item-side deviation metrics (§9.B) under **both** \(T^{\mathrm{cat}}\) and \(T^{\mathrm{rel}}\) in appendix or one main table row; **primary** paper figure uses \(T^{\mathrm{cat}}\) unless stated.

---

## 7. Notation (extended)

| Symbol | Meaning |
|--------|--------|
| \(\mathcal{E}_{\mathrm{train}}\), \(\mathcal{E}_{\mathrm{val}}\), \(\mathcal{E}_{\mathrm{test}}\) | Splits |
| \(d_i\), \(\tilde{p}(i)\) | Train degree; normalized popularity |
| \(g(i)\), \(h(u)\) | Item / user group indices |
| \(\mathcal{I}_g\), \(\mathcal{U}_h\) | Item / user group sets |
| \(C_g = |\mathcal{I}_g|/|\mathcal{I}|\) | Catalog share |
| \(\mathrm{TopK}(u)\) | Top-\(K\) items for \(u\) at evaluation |
| \(\mathrm{Exp}_g\) | Global exposure share of group \(g\) in top-\(K\) lists (§9.B) |
| \(T_g\) | Target share (\(T^{\mathrm{cat}}\) or \(T^{\mathrm{rel}}\)) |
| \(T_{h,g}\) | Fairness transfer matrix (cross-side; §9.D) — **not** the same as \(T_g\) |

**Leakage:** \(g(i)\), \(h(u)\), \(\tilde{p}(i)\), MEG-RW statistics use **train** only. Targets \(T^{\mathrm{rel}}\) use **test** labels **only inside evaluation code**, never for training or preprocessing.

---

## 8. MEG-RW mechanism (summary)

Unchanged from prior versions:

\[
D_g = \frac{M_g}{C_g+\epsilon},\quad M_g=\frac{\sum_{i\in\mathcal{I}_g}d_i}{\sum_j d_j},\quad
\phi(g)=(D_g+\epsilon)^{-\alpha},\quad
\tilde{w}_{ui}=\phi(g(i))\,c_{ui},
\]

then symmetric normalization \(\hat{A}=D^{-1/2}\tilde{A}D^{-1/2}\); BPR uses weighted sampling / pair weights. **No** fairness term in the loss.

---

## 9. Formal evaluation protocol — metric definitions

### 9.A Utility (global)

| Metric | Definition |
|--------|------------|
| **NDCG@10** | Standard per-user NDCG@10 on test positives; report **mean over users** (and global SE / seed std). |
| **Recall@10** | Mean Recall@10 over users. |
| **Optional** | NDCG@20, Recall@20. |

---

### 9.B Item-side fairness (exposure)

Let \(|\mathrm{TopK}(u)|=K\). Define **indicator** \(\mathbb{1}[i\in \mathrm{TopK}(u)]\) from the final ranking.

**1. Exposure share** (group \(g\)):

\[
\mathrm{Exp}_g \;=\; \frac{1}{|\mathcal{U}|} \sum_{u\in\mathcal{U}} \frac{1}{K} \sum_{i\in \mathrm{TopK}(u)} \mathbb{1}[g(i)=g].
\]

**2. Exposure share deviation** (vs. target \(T_g\in\{T_g^{\mathrm{cat}},T_g^{\mathrm{rel}}\}\)):

\[
\Delta^{\mathrm{item}}_g \;=\; \bigl| \mathrm{Exp}_g - T_g \bigr|.
\]

**Aggregates:**

- **MAD:** \(\mathrm{MAD}^{\mathrm{item}} = \frac{1}{G}\sum_g \Delta^{\mathrm{item}}_g\).
- **Max:** \(\mathrm{MaxDev}^{\mathrm{item}} = \max_g \Delta^{\mathrm{item}}_{g}\).
- **Weighted:** \(\sum_g w_g \Delta^{\mathrm{item}}_g\) with \(w_g\propto T_g\) (optional).

**3. Exposure ratio (balance):**

\[
\mathrm{ER}^{\mathrm{item}} \;=\; \frac{\min_g \mathrm{Exp}_g}{\max_g \mathrm{Exp}_g + \epsilon}.
\]

(Higher \(\Rightarrow\) more balanced across groups.)

**4. Tail / head exposure ratio (TailExp@\(K\)):**

\[
\mathrm{TailExp}(K) \;=\; \frac{\mathrm{Exp}_{\mathrm{Tail}}}{\mathrm{Exp}_{\mathrm{Head}} + \epsilon}
\]

**5. Group-conditioned NDCG**

For each \(g\), compute **NDCG@K** using only **relevant items that lie in \(\mathcal{I}_g\)** (binary relevance on test positives), **or** restrict the ranked list to positions of items in \(\mathcal{I}_g\)—**choose one standard** (e.g. RecBole-style per-group NDCG) and **fix in code**. Report \(\mathrm{NDCG}^{(g)}\) for \(g\in\{\mathrm{Head},\ldots,\mathrm{Tail}\}\) to separate “tail shown more” vs. “tail shown **relevantly**.”

---

### 9.C User-side fairness (utility)

For each user group \(h\), evaluate **test** users \(u\in\mathcal{U}_h\):

\[
U_h^{\mathrm{NDCG}} = \frac{1}{|\mathcal{U}_h|} \sum_{u\in\mathcal{U}_h} \mathrm{NDCG@10}(u), \quad
U_h^{\mathrm{Rec}} = \frac{1}{|\mathcal{U}_h|} \sum_{u\in\mathcal{U}_h} \mathrm{Recall@10}(u).
\]

| Metric | Formula |
|--------|---------|
| **User NDCG gap** | \(\mathrm{Gap}^{\mathrm{user}}_{\mathrm{NDCG}} = \max_h U_h^{\mathrm{NDCG}} - \min_h U_h^{\mathrm{NDCG}}\) |
| **User Recall gap** | \(\mathrm{Gap}^{\mathrm{user}}_{\mathrm{Rec}} = \max_h U_h^{\mathrm{Rec}} - \min_h U_h^{\mathrm{Rec}}\) |
| **Std (NDCG)** | \(\mathrm{Std}^{\mathrm{user}}_{\mathrm{NDCG}} = \mathrm{std}_h(U_h^{\mathrm{NDCG}})\) |
| **Worst-group NDCG** | \(\mathrm{WorstUser} = \min_h U_h^{\mathrm{NDCG}}\) |

**Novelty / tail in recommendations to users:** per user \(u\), fraction of \(\mathrm{TopK}(u)\) in \(\mathcal{I}_{\mathrm{Tail}}\); average within \(\mathcal{U}_h\) → **tail-received exposure** \(\mathrm{TailRecv}_h\). Report \(\mathrm{TailRecv}_h\) alongside \(U_h^{\mathrm{NDCG}}\).

---

### 9.D Cross-side coupling

**1. Fairness transfer matrix** \(T \in \mathbb{R}^{L\times G}\) (reuse symbol only here; do not confuse with target \(T_g\)):

\[
T_{h,g} \;=\; \frac{1}{|\mathcal{U}_h|} \sum_{u\in\mathcal{U}_h} \frac{1}{K} \sum_{i\in\mathrm{TopK}(u)} \mathbb{1}[g(i)=g].
\]

Compare \(T^{\mathrm{raw}}\) vs. \(T^{\mathrm{MEG}}\); plot \(\Delta T = T^{\mathrm{MEG}}-T^{\mathrm{raw}}\).

**2. Exposure–utility alignment (per user group)**

Let \(\tau_h(\alpha)\) = **tail-received** share for group \(h\) at fairness strength \(\alpha\) (or \(\mathrm{Exp}_{\mathrm{Tail}}\) **within** group \(h\): average over \(u\in\mathcal{U}_h\) of tail fraction in \(\mathrm{TopK}(u)\)). Let \(\upsilon_h(\alpha)=U_h^{\mathrm{NDCG}}\) at \(\alpha\).

Over the grid \(\mathcal{A}=\{\alpha_1,\ldots,\alpha_m\}\), define **alignment** (Pearson correlation):

\[
A_h \;=\; \mathrm{corr}\bigl( (\tau_h(\alpha_1),\ldots,\tau_h(\alpha_m)),\, (\upsilon_h(\alpha_1),\ldots,\upsilon_h(\alpha_m)) \bigr).
\]

*Interpretation:* if higher \(\alpha\) increases tail exposure for group \(h\) **together** with NDCG, \(A_h\) is positive; decoupling \(\Rightarrow\) low/negative \(A_h\).

**3. Cross-side fairness consistency (CSFC) — secondary summary**

Normalize item and user disparity to \([0,1]\) per run (e.g. divide by raw-graph value at \(\alpha=0\) on same seed, clip).

Let \(D_I\) = normalized item disparity (e.g. \(\mathrm{MAD}^{\mathrm{item}}\) or \(\max_g \Delta^{\mathrm{item}}_g\)), \(D_U\) = normalized user NDCG gap (vs. raw).

\[
\mathrm{CSFC} \;=\; 1 - \frac{D_I + D_U}{2}
\]

(optional **weighted** \(\beta D_I + (1-\beta)D_U\)). **Secondary** headline only—primary evidence = Pareto curves + \(T\) + \(A_h\).

---

## 10. Master evaluation protocol table (paper-ready)

| Stage | Rule |
|--------|------|
| **Data filter** | 5-core users/items on train graph |
| **Split** | Temporal 80/10/10 per user if timestamps; else LOO + negatives |
| **Item groups** | 10% / 20% / 30% / 40% by train \(d_i\); ablations: 2 groups, 5 groups |
| **User groups** | \(M(u)\) from \(\tilde{p}(i)\) on train interactions; 4 quantiles → niche → mainstream |
| **Preprocessor** | MEG-RW on **train** edges only; \(\alpha\) sweep |
| **Training** | BPR-MF / LightGCN / NGCF; **same** hyperparams where possible; early stop on **val NDCG@10** (global) |
| **Evaluation** | Test users; **NDCG@10**, **Recall@10**; **§9.B–D**; targets \(T^{\mathrm{cat}}\) (main), \(T^{\mathrm{rel}}\) (secondary) |
| **Seeds** | ≥5; target **10** |
| **Figures** | Pareto: NDCG vs. \(\mathrm{MAD}^{\mathrm{item}}\) / MaxDev; NDCG vs. \(\mathrm{Gap}^{\mathrm{user}}_{\mathrm{NDCG}}\); heatmaps \(T\), \(\Delta T\) |
| **Baselines** | Raw + MEG-RW + 2–3 fairness baselines (same audit on all) |

---

## 11. Draft: Method + Experimental Setup (for paper body)

**Method (MEG-RW).** We partition items into \(G=4\) popularity strata on the **training** graph (10/20/30/40). We compute group dominance \(D_g\) and assign nonnegative train-edge weights \(\tilde{w}_{ui}=\phi(g(i))c_{ui}\) with \(\phi(g)=(D_g+\epsilon)^{-\alpha}\), then build the normalized adjacency for graph models or sampling weights for BPR. **No** fairness term is added to the recommender loss.

**Experimental setup.** Datasets: MovieLens-1M, Last.fm HetRec 2011, Amazon Books 5-core (details in supplement). We compare **raw** vs. **MEG-RW** graphs under identical optimizers. **Item-side** metrics compare \(\mathrm{Exp}_g\) to **catalog-share** targets (primary) and **relevance-aware** targets (secondary). **User-side** metrics report \(U_h^{\mathrm{NDCG}}\), \(\mathrm{Gap}^{\mathrm{user}}\), and **worst-group** utility over **mainstreamness** quartiles. **Cross-side:** we visualize the **transfer matrix** \(T_{h,g}\) and report **exposure–utility alignment** \(A_h\) along \(\alpha\), plus optional **CSFC**. We repeat all runs over multiple random seeds and report mean \(\pm\) std and Pareto curves, testing **H1–H5**.

---

## 12. Baselines, tables, figures, repo, roadmap, risks

**Baselines:** BPR-MF, LightGCN, NGCF; fairness: FDA, FA4GCF, FairGap (optional). **Same dual-sided audit** for every method.

**Table 2 (main)** columns: NDCG@10, Recall@10, \(\mathrm{MAD}^{\mathrm{item}}\) (or MaxDev), \(\mathrm{ER}^{\mathrm{item}}\), TailExp@10, \(\mathrm{Gap}^{\mathrm{user}}_{\mathrm{NDCG}}\), \(\mathrm{Gap}^{\mathrm{user}}_{\mathrm{Rec}}\), WorstUser, \(\mathrm{Std}^{\mathrm{user}}_{\mathrm{NDCG}}\); optional CSFC.

**Figures:** (1) Pareto NDCG–item deviation, (2) Pareto NDCG–user gap, (3) \(T\) and \(\Delta T\), (4) optional \(A_h\) vs. \(h\) bar plot.

**Repo (`src/cikm_eval/`):** `grouping.py`, `fairness_audit.py` (implemented); planned: `targets.py`, `transfer_matrix.py`, `alignment.py`, `csfc.py`, `pareto.py`, `bootstrap.py`.

**Roadmap / risks / next actions:** unchanged spirit from v2.0; prioritize implementing §9 definitions **exactly** once in code and freezing \(\tilde{p}\) and group-conditioned NDCG convention.

---

## 13. One-paragraph story (introduction draft)

We partition items into multi-group popularity strata and users into multi-group **mainstreamness** strata from **training** behavior. We apply **MEG-RW**, a **preprocessing-only** reweighting of the user–item graph aimed at reducing structural **head dominance** before training, **without** modifying downstream objectives. We evaluate not only whether **item-side** exposure moves toward **catalog-share** (and relevance-aware) targets, but also whether **user-side** utility gaps widen or narrow—using a **transfer matrix** and **exposure–utility alignment** across fairness strengths. This frames fairness as a **two-sided multi-group** phenomenon rather than a single aggregate score.

---

## 14. Novelty statement (abstract)

*Fairness-aware recommendation often intervenes during training, data generation, or reranking. We instead preprocess the **user–item graph** to reduce **multi-group item exposure imbalance** before learning, leaving objectives unchanged. We argue this is insufficient to assess in isolation: we audit **user-group utility** and **cross-side** exposure flows via a **transfer matrix** and alignment metrics, showing when gains **transfer**, **conflict**, or **obscure** worst-group outcomes—supporting joint, multi-group fairness evaluation alongside relevance.*

---

*Document version: **3.0** — locked item/user groups, dual targets, full metric block (9.A–D), RQ1–4 & H1–5, master protocol table, draft Method+Setup.*

---

## 15. Empirical Insight Snapshot (ML-1M, seed 0, LightGCN family)

This section records the current high-confidence observations from completed seed-0 runs for paper drafting. These are provisional until multi-seed confirmation.

### 15.1 Control result: standard LightGCN is alpha-invariant

- For `lightgcn`, `baseline`, `alpha_0.1`, `alpha_0.2`, `alpha_0.4`, and `alpha_0.8` are numerically identical (including transfer delta L1 shift = 0.0).
- Interpretation: MEG-RW does not affect the unweighted propagation path; this is expected by design.
- Paper-safe claim: observed fairness shifts arise from weighted graph propagation, not incidental data perturbation.

Suggested sentence for paper:
"Standard LightGCN is invariant to MEG-RW alpha in our setup, confirming that the downstream effects are driven by weighted propagation rather than accidental data changes."

### 15.2 Weighted LightGCN shows controllable structural redistribution

Observed for `weighted_lightgcn` (ML-1M, seed 0):

- Utility remains stable across alpha:
  - baseline: Recall@10 = 0.1479, NDCG@10 = 0.2393
  - alpha=0.1: Recall@10 = 0.1481, NDCG@10 = 0.2394
  - alpha=0.2: Recall@10 = 0.1482, NDCG@10 = 0.2392
  - alpha=0.4: Recall@10 = 0.1481, NDCG@10 = 0.2394
  - alpha=0.8: Recall@10 = 0.1473, NDCG@10 = 0.2397
- Item exposure deviation improves monotonically (lower is better):
  - 0.39534 -> 0.39181 -> 0.38852 -> 0.38227 -> 0.37212
- Tail exposure ratio improves strongly and monotonically:
  - 0.000364 -> 0.000546 -> 0.000679 -> 0.001109 -> 0.003411
- Transfer-delta magnitude increases monotonically (structural control):
  - L1 shift: 0.0564 -> 0.1092 -> 0.2092 -> 0.3716 (for alpha 0.1 to 0.8)

### 15.3 Main interpretation to preserve

- MEG-RW provides a stable alpha-controlled mechanism for redistributing exposure mass away from head items.
- This redistribution occurs with near-constant global ranking utility.
- The mechanism is model-path dependent: effect appears in weighted models, not in standard LightGCN control runs.

### 15.4 Cautionary interpretation (important for paper framing)

- Do not claim that weighted variants are globally superior to standard LightGCN based on raw cross-model baseline values.
- Primary evidence should be within-model alpha sweeps plus control-model invariance.
- Do not overstate absolute fairness: despite large relative gains, tail exposure remains low in absolute terms (head-dominant regime persists).

### 15.5 Seed-0 working conclusions (draft language)

1. Controlled intervention: item-side fairness metrics improve smoothly and monotonically with alpha.
2. Utility robustness: NDCG@10 and Recall@10 remain effectively flat across the alpha sweep.
3. Mechanism isolation: no-alpha-effect in standard LightGCN and strong alpha-effect in weighted LightGCN support a causal mechanism tied to weighted propagation.

### 15.6 Immediate validation checklist before camera-ready claims

- Confirm identical split protocol and config parity between control and weighted runs for each seed.
- Replicate the same trends for seeds 1-4 before making stability claims.
- Keep both relative-gain and absolute-level reporting in all fairness plots/tables.

---

## 16. Semantic x Popularity Implementation Log

This section tracks concrete implementation decisions for semantic-aware MEG-RW so paper text and code stay aligned.

### 16.1 Implemented mechanism (code-level)

- Added semantic factor module: `src/meg_rw/semantic.py`
  - Builds item semantic vectors from `train_dataset.item_feat` (excluding `iid_field`).
  - Uses SVD projection (default 128d) to form compact semantics.
  - Builds user semantic profiles from train interactions.
  - Computes per-interaction cosine similarity and converts to a bounded factor:
    - z-score + clipping (`meg_sem_clip_z`)
    - exponential scaling by `meg_sem_beta`
    - optional mean-renormalization to keep average factor near 1
    - final clipping to [`meg_sem_wmin`, `meg_sem_wmax`]
- Updated `src/cikm_train/recbole_dataset.py`:
  - Existing MEG-RW popularity weight remains `w_pop`.
  - If semantic enabled, computes `w_sem`.
  - Final train interaction weight written to `meg_rw_value_field` is `w_pop * w_sem`.
- Updated `src/cikm_train/run_experiment.py`:
  - Passes semantic config knobs through `_build_model` injection path for all weighted backbones.
  - Adds semantic settings to fairness metadata (`meg_sem_enable`, `meg_sem_beta`).
- Updated runner `scripts/run_full_fairness_grid.sh`:
  - Added `BETAS` environment knob (default `0.0`).
  - Weighted-model sweeps now iterate over `(beta, alpha)` grid.
  - Run label format: `alpha_<a>_beta_<b>`.

### 16.2 New configuration knobs

Added to `config/cikm_base.yaml`:

- `meg_sem_enable` (bool)
- `meg_sem_beta` (float)
- `meg_sem_dim` (int)
- `meg_sem_clip_z` (float)
- `meg_sem_wmin` (float)
- `meg_sem_wmax` (float)
- `meg_sem_renorm_mean_one` (bool)

### 16.3 Recommended first semantic sweeps

For `weighted_lightgcn` and `weighted_ngcf` on `ml1m`, `lastfm`:

- `ALPHAS="0.1 0.2 0.4"`
- `BETAS="0.0 0.1 0.3 0.5"`
- Use seed 0 first, then seeds 0..4 on best settings.

### 16.4 Paper-facing interpretation guardrails

- Treat prior alpha-only experiments as `beta=0.0`.
- Report no-op diagnostics (top-K change vs baseline) whenever alpha/beta appears inactive.
- Prefer moderate control (`alpha/beta`) regions for fairness-utility tradeoff narrative.
