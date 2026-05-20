from __future__ import annotations

import numpy as np
import torch
from recbole.data.interaction import Interaction

from meg_rw.reweight import DEFAULT_MEG_RW_FIELD, meg_rw_weights_for_interaction_rows
from meg_rw.semantic import semantic_weights_for_interactions


def inject_meg_rw_into_train_dataset(
    train_dataset,
    alpha: float,
    fracs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    rw_mode: str = "dominance",
    normalize_phi: bool = False,
    field_name: str = DEFAULT_MEG_RW_FIELD,
    sem_enable: bool = False,
    sem_beta: float = 0.0,
    sem_dim: int = 128,
    sem_clip_z: float = 2.0,
    sem_wmin: float = 0.5,
    sem_wmax: float = 2.0,
    sem_renorm_mean_one: bool = True,
) -> None:
    """
    Add ``field_name`` to ``train_dataset.inter_feat`` (in-place).
    Uses train-only item degrees for popularity groups (MEG-RW).
    """
    feat = train_dataset.inter_feat
    iid_f = train_dataset.iid_field
    items = feat[iid_f].detach().cpu().numpy()
    n_items = train_dataset.item_num
    uids = feat[train_dataset.uid_field].detach().cpu().numpy()
    w_pop = meg_rw_weights_for_interaction_rows(
        items, n_items=n_items, alpha=alpha, fracs=fracs, mode=rw_mode, normalize_phi=normalize_phi
    )
    if sem_enable and abs(float(sem_beta)) > 1e-12:
        w_sem = semantic_weights_for_interactions(
            train_dataset=train_dataset,
            item_internal_ids=items,
            user_internal_ids=uids,
            beta=float(sem_beta),
            clip_z=float(sem_clip_z),
            sem_dim=int(sem_dim),
            w_min=float(sem_wmin),
            w_max=float(sem_wmax),
            renorm_mean_one=bool(sem_renorm_mean_one),
        )
    else:
        w_sem = np.ones_like(w_pop, dtype=np.float32)
    w = (w_pop * w_sem).astype(np.float32, copy=False)
    new_d = {k: feat[k] for k in feat.interaction}
    new_d[field_name] = torch.as_tensor(w, dtype=torch.float32)
    train_dataset.inter_feat = Interaction(new_d)
