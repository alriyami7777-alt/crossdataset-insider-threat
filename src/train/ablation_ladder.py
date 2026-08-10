"""
STEP A ablation ladder (in-distribution only).

Variants
  A1  MLP head on x_ud ONLY (counts+deviation). No graph / memory.
  A2  Memory ONLY: r = s_u(end of day), same MLP head.
  A3  Full:        r = [standardize(x_ud) || s_u], same head.

Leakage policy (documented choice)
  Build ``user_day_features(df, deviation=True)`` on the FULL event stream so
  causal deviation baselines see earlier days. Then:
    * A1 — split the user-day frame (temporal by day quantile / user-disjoint),
      matching the honest RF diagonal in ``supervised_transfer``.
    * A2/A3 — split raw events with ``temporal_split`` / ``user_disjoint_split``
      for memory streaming; join full-stream feature rows by (user, day).
  Scaler always fit on TRAIN rows only (base-count columns).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, user_day_features
from ..data.schema import LABEL, USER
from .day_supervised_gnn import DaySupervisedGNN, _with_lift
from .evaluate import compute_metrics
from .splits import temporal_split, user_disjoint_split
from .supervised_transfer import _split_user_day_frame

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A1: feature-only MLP
# ---------------------------------------------------------------------------
class FeatureOnlyMLP:
    """2-layer MLP on standardized x_ud (no TGN)."""

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 3e-4,
        hidden=(128, 64),
        dropout: float = 0.1,
        patience: int = 20,
        min_epochs: int = 40,
        val_frac: float = 0.15,
        batch_size: int = 4096,
        seed: int = 7,
        device=None,
        val_mode: str = "temporal",  # temporal day holdout | user
        max_pos_weight: float = 100.0,
    ):
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.dropout = dropout
        self.patience = patience
        self.min_epochs = min_epochs
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.seed = seed
        self._device = device
        self.val_mode = val_mode
        self.max_pos_weight = max_pos_weight
        self.scaler: Optional[StandardScaler] = None
        self.head = None
        self._feat_cols = list(FEATURE_COLUMNS)
        self._base_cols = list(BASE_FEATURE_COLUMNS)
        self.best_epoch = -1
        self.best_val_pr = float("nan")

    def _torch(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def _transform(self, feat: pd.DataFrame) -> np.ndarray:
        base = feat[self._base_cols].to_numpy(dtype=np.float64)
        if self.scaler is not None:
            base = self.scaler.transform(base)
        # clip extreme z-scores / deviation spikes for stable MLP training
        base = np.clip(base, -8.0, 8.0)
        dev_cols = [c for c in self._feat_cols if c.startswith("dev_")]
        dev = np.clip(feat[dev_cols].to_numpy(dtype=np.float32), -8.0, 8.0)
        return np.concatenate([base.astype(np.float32), dev], axis=1)

    def _val_split_idx(self, feat: pd.DataFrame):
        n = len(feat)
        if self.val_mode == "temporal" and "day" in feat.columns:
            f = feat.reset_index(drop=True)
            order = np.argsort(f["day"].to_numpy(), kind="mergesort")
            n_val = max(1, int(round(n * self.val_frac)))
            if n >= 5:
                n_val = min(max(n_val, 1), n - 2)
            va = order[-n_val:]
            tr = order[:-n_val]
            return tr, va
        rng = np.random.default_rng(self.seed)
        user_pos = feat.groupby(USER)[LABEL].max()
        pos_u = user_pos[user_pos == 1].index.to_numpy()
        neg_u = user_pos[user_pos == 0].index.to_numpy()
        rng.shuffle(pos_u)
        rng.shuffle(neg_u)

        def _val(users):
            nv = int(round(len(users) * self.val_frac))
            if len(users) >= 2:
                nv = min(max(nv, 1), len(users) - 1)
            else:
                nv = 0
            return set(users[:nv])

        val_users = _val(pos_u) | _val(neg_u)
        is_val = feat[USER].isin(val_users).to_numpy()
        tr, va = np.where(~is_val)[0], np.where(is_val)[0]
        if len(va) == 0:
            perm = rng.permutation(n)
            cut = max(1, int(0.2 * n))
            return perm[cut:], perm[:cut]
        return tr, va

    def fit(self, feat_train: pd.DataFrame):
        torch = self._torch()
        import torch.nn.functional as F
        from ..models.ud_head import UserDayMLP
        from ..utils.seed import set_seed
        set_seed(self.seed)

        feat_train = feat_train.reset_index(drop=True)
        self.scaler = StandardScaler()
        self.scaler.fit(feat_train[self._base_cols].to_numpy(dtype=np.float64))
        X = self._transform(feat_train)
        y = feat_train[LABEL].to_numpy(dtype=np.float32)
        tr_idx, va_idx = self._val_split_idx(feat_train)

        n_pos = float(y[tr_idx].sum())
        n_neg = float(len(tr_idx) - n_pos)
        raw_pw = n_neg / max(n_pos, 1.0)
        pw_val = float(min(raw_pw, self.max_pos_weight))
        pw = torch.tensor([pw_val], device=self._device)
        prior = max(n_pos / max(n_pos + n_neg, 1.0), 1e-6)
        log.info(
            "A1 MLP pos_weight=%.1f (raw=%.1f) n_pos=%.0f n_neg=%.0f prior=%.5f",
            pw_val, raw_pw, n_pos, n_neg, prior,
        )

        self.head = UserDayMLP(
            X.shape[1], hidden=self.hidden, dropout=self.dropout,
        ).to(self._device)
        # imbalance-aware bias: start near the positive prior
        with torch.no_grad():
            last = self.head.net[-1]
            last.bias.fill_(float(np.log(prior / (1.0 - prior))))

        opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        X_t = torch.tensor(X, dtype=torch.float, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float, device=self._device)

        best_pr, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            self.head.train()
            perm = np.random.default_rng(self.seed + epoch).permutation(tr_idx)
            total = 0.0
            n_batches = 0
            for i in range(0, len(perm), self.batch_size):
                sl = perm[i:i + self.batch_size]
                opt.zero_grad()
                logit = self.head(X_t[sl])
                loss = F.binary_cross_entropy_with_logits(
                    logit, y_t[sl], pos_weight=pw,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                opt.step()
                total += float(loss.detach().cpu())
                n_batches += 1
            sched.step()

            self.head.eval()
            with torch.no_grad():
                vlogit = self.head(X_t[va_idx])
                vs = torch.sigmoid(vlogit).cpu().numpy()
            vy = y[va_idx]
            if len(np.unique(vy)) < 2:
                vpr = float("nan")
            else:
                vpr = float(compute_metrics(vy, vs)["pr_auc"])
            log.info(
                "A1 epoch %d/%d loss=%.4f val_pr=%.4f",
                epoch + 1, self.epochs, total / max(n_batches, 1), vpr,
            )
            if np.isfinite(vpr) and vpr > best_pr + 1e-4:
                best_pr = vpr
                bad = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.head.state_dict().items()
                }
                self.best_epoch = epoch + 1
            else:
                bad += 1
            if epoch + 1 >= self.min_epochs and bad >= self.patience:
                log.info("A1 early stop at %d (best=%d pr=%.4f)",
                         epoch + 1, self.best_epoch, best_pr)
                break

        if best_state is not None:
            self.head.load_state_dict(best_state)
            self.head.to(self._device)
            self.best_val_pr = best_pr
        return self

    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        torch = self._torch()
        X = self._transform(feat)
        self.head.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float, device=self._device)
            return torch.sigmoid(self.head(xt)).cpu().numpy()


def run_a1(
    feat: pd.DataFrame,
    dataset: str,
    train_frac: float = 0.7,
    seed: int = 0,
    **mlp_kw,
) -> List[dict]:
    rows = []
    for split in ("temporal", "user_disjoint"):
        tr, te = _split_user_day_frame(
            feat, split, train_frac=train_frac, seed=seed,
        )
        kw = dict(mlp_kw)
        kw.setdefault(
            "val_mode", "temporal" if split == "temporal" else "user",
        )
        clf = FeatureOnlyMLP(seed=seed, **kw)
        clf.fit(tr)
        scores = clf.predict_proba(te)
        y = te[LABEL].to_numpy()
        m = _with_lift(compute_metrics(y, scores), y)
        m.update({
            "dataset": dataset,
            "split": split,
            "variant": "A1",
            "best_epoch": clf.best_epoch,
            "best_val_pr": clf.best_val_pr,
        })
        log.info(
            "A1 %s/%s: pr_auc=%.4f lift=%.2f n=%d pos=%d",
            dataset, split, m["pr_auc"], m["lift"], m["n"], m["n_pos"],
        )
        rows.append(m)
    return rows


def run_a2_a3(
    df: pd.DataFrame,
    full_feat: pd.DataFrame,
    dataset: str,
    variant: str,
    train_frac: float = 0.7,
    seed: int = 0,
    **gnn_kw,
) -> List[dict]:
    rows = []
    for split, splitter in (
        ("temporal", lambda d: temporal_split(d, train_frac=train_frac)),
        ("user_disjoint", lambda d: user_disjoint_split(d, train_frac=train_frac, seed=seed)),
    ):
        train_df, test_df = splitter(df)
        det = DaySupervisedGNN(variant=variant, seed=seed, **gnn_kw)
        det.set_full_features(full_feat)
        det.fit(train_df)
        agg = det.score_user_day(test_df)
        y = agg["label"].to_numpy()
        m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
        m.update({
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "best_epoch": det.best_epoch,
            "best_val_pr": det.best_val_pr,
            "loss": det.loss,
        })
        log.info(
            "%s %s/%s: pr_auc=%.4f lift=%.2f n=%d pos=%d best_ep=%d",
            variant, dataset, split, m["pr_auc"], m["lift"],
            m["n"], m["n_pos"], det.best_epoch,
        )
        rows.append(m)
    return rows


def run_ablation_ladder(
    dfs: Dict[str, pd.DataFrame],
    *,
    variants=("A1", "A2", "A3"),
    train_frac: float = 0.7,
    seed: int = 0,
    a1_kw=None,
    gnn_kw=None,
) -> pd.DataFrame:
    """Run A1/A2/A3 on every dataset under both ID splits → tidy DataFrame."""
    a1_kw = dict(a1_kw or {})
    gnn_kw = dict(gnn_kw or {})
    rows: List[dict] = []
    full_feats = {
        name: user_day_features(df, deviation=True) for name, df in dfs.items()
    }
    for name, df in dfs.items():
        feat = full_feats[name]
        if "A1" in variants:
            rows.extend(run_a1(feat, name, train_frac=train_frac, seed=seed, **a1_kw))
        for v in variants:
            if v in ("A2", "A3"):
                rows.extend(run_a2_a3(
                    df, feat, name, v,
                    train_frac=train_frac, seed=seed, **gnn_kw,
                ))
    return pd.DataFrame(rows)


def run_bc_id(
    dfs: Dict[str, pd.DataFrame],
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    losses=("bce", "focal"),
    **gnn_kw,
) -> pd.DataFrame:
    """STEP B/C: day-supervised full model under both losses × both splits."""
    rows = []
    full_feats = {
        name: user_day_features(df, deviation=True) for name, df in dfs.items()
    }
    for name, df in dfs.items():
        feat = full_feats[name]
        for loss in losses:
            kw = dict(gnn_kw)
            kw["loss"] = loss
            kw.setdefault("epochs", 80)
            kw.setdefault("ssl_epochs", 3)
            batch = run_a2_a3(
                df, feat, name, "BC",
                train_frac=train_frac, seed=seed, **kw,
            )
            for r in batch:
                r["variant"] = f"BC_{loss}"
                r["loss_tag"] = loss
            rows.extend(batch)
    return pd.DataFrame(rows)
