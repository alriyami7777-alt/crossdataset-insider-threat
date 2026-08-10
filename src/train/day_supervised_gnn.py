"""
Day-level supervised Temporal GNN (STEPS A2/A3/B/C).

Train/eval match:
  * Stream events day-by-day to update TGN memory (no per-edge BCE).
  * At end of day d, score users with an MLP head on
        A2: r = s_u
        A3: r = [standardize(x_ud) || s_u]
    against y_{u,d} directly.

Leakage-safe features:
  * ``user_day_features(full_df, deviation=True)`` is built on the FULL event
    stream (causal expanding baselines need earlier days).
  * Event splits (temporal / user-disjoint) select which edges update memory
    and which user-days enter the loss / metrics; feature rows are joined by
    (user, day) from the full-stream table.

Scaler: StandardScaler fit on TRAIN rows, BASE_FEATURE_COLUMNS only; deviation
columns are left as-is (already ~standardized).
"""
from __future__ import annotations

import logging
from collections import namedtuple
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, user_day_features
from ..data.graph_builder import edge_stream
from ..data.schema import ACTIONS, LABEL, TIMESTAMP, USER
from .evaluate import compute_metrics

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "u h a dt y")


def focal_bce_with_logits(
    logits, targets, pos_weight=None, gamma: float = 2.0,
):
    """Focal loss on top of BCE-with-logits (gamma=2 default)."""
    import torch
    import torch.nn.functional as F

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none",
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return (((1.0 - p_t) ** gamma) * bce).mean()


def _with_lift(m: dict, y_true) -> dict:
    y = np.asarray(y_true)
    p = float(y.mean()) if len(y) else float("nan")
    m = dict(m)
    m["base_rate"] = p
    pr = m.get("pr_auc", float("nan"))
    m["lift"] = float(pr / p) if (p and p > 0 and np.isfinite(pr)) else float("nan")
    return m


class DaySupervisedGNN:
    """TGN memory + day-level MLP head (no edge BCE).

    Parameters
    ----------
    variant : ``"A2"`` (memory only) or ``"A3"`` / ``"BC"`` (features || memory)
    loss : ``"bce"`` or ``"focal"``
    use_memory : if False, reset memory each day (static within-day aggregation;
        no cross-day TGN state). Default True.
    use_ssl_pretrain : if False, skip next-action SSL. Default True.
    use_time_encoding : if False, zero the Time2Vec channel. Default True.
    use_deviation_features : if False, base counts only (no ``dev_*``). Default True.
    """

    def __init__(
        self,
        variant: str = "A3",
        mem_dim: int = 64,
        epochs: int = 80,
        ssl_epochs: int = 3,
        lr: float = 3e-4,
        batch_edges: int = 8192,
        hidden=(128, 64),
        dropout: float = 0.1,
        pos_weight: bool = True,
        max_pos_weight: float = 100.0,
        loss: str = "bce",
        focal_gamma: float = 2.0,
        patience: int = 12,
        min_epochs: int = 25,
        val_frac: float = 0.15,
        lr_schedule: str = "cosine",
        val_every: int = 2,
        use_dann: bool = False,
        use_memory: bool = True,
        use_ssl_pretrain: bool = True,
        use_time_encoding: bool = True,
        use_deviation_features: bool = True,
        seed: int = 7,
        device=None,
    ):
        if variant not in ("A2", "A3", "BC"):
            raise ValueError(f"variant must be A2/A3/BC, got {variant}")
        self.variant = variant
        self.use_features = variant != "A2"
        self.mem_dim = mem_dim
        self.epochs = epochs
        self.ssl_epochs = ssl_epochs
        self.lr = lr
        self.batch_edges = batch_edges
        self.hidden = hidden
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.max_pos_weight = max_pos_weight
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.patience = patience
        self.min_epochs = min_epochs
        self.val_frac = val_frac
        self.lr_schedule = lr_schedule
        self.val_every = max(1, int(val_every))
        self.use_dann = use_dann
        # Component toggles (defaults preserve full-model behaviour elsewhere).
        self.use_memory = bool(use_memory)
        self.use_ssl_pretrain = bool(use_ssl_pretrain)
        self.use_time_encoding = bool(use_time_encoding)
        self.use_deviation_features = bool(use_deviation_features)
        self.seed = seed
        self._device = device
        self.model = None
        self.head = None
        self.proj = None
        self.adv = None
        self.scaler: Optional[StandardScaler] = None
        self._base_cols = list(BASE_FEATURE_COLUMNS)
        if self.use_deviation_features:
            self._feat_cols = list(FEATURE_COLUMNS)
        else:
            self._feat_cols = list(BASE_FEATURE_COLUMNS)
        self._full_feat: Optional[pd.DataFrame] = None
        self.best_val_pr = float("nan")
        self.best_epoch = -1

    def _torch(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def set_full_features(self, feat: pd.DataFrame):
        """Attach full-stream user-day features (leakage-safe causal deviation)."""
        self._full_feat = feat

    def _features_for(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._full_feat is not None:
            # Restrict full-stream rows to (user, day) pairs present in df
            d = df.copy()
            d["day"] = d[TIMESTAMP].dt.floor("D")
            keys = d[[USER, "day"]].drop_duplicates()
            feat = self._full_feat.merge(keys, on=[USER, "day"], how="inner")
            return feat
        return user_day_features(df, deviation=True)

    def _fit_scaler(self, feat_train: pd.DataFrame):
        if not self.use_features:
            self.scaler = None
            return
        self.scaler = StandardScaler()
        self.scaler.fit(feat_train[self._base_cols].to_numpy(dtype=np.float64))

    def _transform_x(self, feat: pd.DataFrame) -> np.ndarray:
        """Return x_ud with base cols scaled, deviation cols clipped (if enabled)."""
        base = feat[self._base_cols].to_numpy(dtype=np.float64)
        if self.scaler is not None:
            base = self.scaler.transform(base)
        base = np.clip(base, -8.0, 8.0)
        if not self.use_deviation_features:
            return base.astype(np.float32)
        dev_cols = [c for c in self._feat_cols if c.startswith("dev_")]
        if not dev_cols:
            return base.astype(np.float32)
        dev = np.clip(feat[dev_cols].to_numpy(dtype=np.float32), -8.0, 8.0)
        return np.concatenate([base.astype(np.float32), dev], axis=1)

    def _tensors(self, df: pd.DataFrame, feat: Optional[pd.DataFrame] = None):
        torch = self._torch()
        e, uidx, hidx = edge_stream(df)
        t = e["t"].to_numpy().astype(np.float64)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        d_sorted = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
        day = d_sorted[TIMESTAMP].dt.floor("D").to_numpy()
        dev = self._device
        out = {
            "u": torch.tensor(e["u"].to_numpy(), dtype=torch.long, device=dev),
            "h": torch.tensor(e["h"].to_numpy(), dtype=torch.long, device=dev),
            "a": torch.tensor(e["a"].to_numpy(), dtype=torch.long, device=dev),
            "dt": torch.tensor(t, dtype=torch.float, device=dev).unsqueeze(-1),
            "y": torch.tensor(e["y"].to_numpy(), dtype=torch.long, device=dev),
            "day": day,
            "n_users": len(uidx),
            "n_hosts": len(hidx),
            "n_edges": len(e),
            "uidx": uidx,
        }
        if feat is None:
            feat = self._features_for(df)
        feat = feat.copy()
        feat["_u"] = feat[USER].map(uidx)
        feat = feat.dropna(subset=["_u"]).reset_index(drop=True)
        feat["_u"] = feat["_u"].astype(int)
        out["feat"] = feat
        out["x_mat"] = self._transform_x(feat) if self.use_features else None
        return out

    def _batches(self, ts, idx):
        for i in range(0, len(idx), self.batch_edges):
            sl = idx[i:i + self.batch_edges]
            yield Batch(
                ts["u"][sl], ts["h"][sl], ts["a"][sl], ts["dt"][sl], ts["y"][sl],
            )

    def _ud_for_day(self, ts, day_val, user_mask=None):
        torch = self._torch()
        feat = ts["feat"]
        mask = feat["day"].to_numpy() == day_val
        if user_mask is not None:
            mask = mask & feat[USER].isin(user_mask).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        fday = feat.iloc[idx]
        u = torch.tensor(fday["_u"].to_numpy(), dtype=torch.long, device=self._device)
        y = torch.tensor(fday[LABEL].to_numpy(), dtype=torch.float, device=self._device)
        x = None
        if self.use_features and ts["x_mat"] is not None:
            x = torch.tensor(
                ts["x_mat"][idx], dtype=torch.float, device=self._device,
            )
        return {
            "u": u, "x": x, "y": y,
            "users": fday[USER].to_numpy(),
            "day": fday["day"].to_numpy(),
            "label": fday[LABEL].to_numpy(),
        }

    def _split_train_val_users(self, feat: pd.DataFrame):
        """Hold out val_frac of train users (stratified on any-positive)."""
        rng = np.random.default_rng(self.seed)
        user_pos = feat.groupby(USER)[LABEL].max()
        pos_u = user_pos[user_pos == 1].index.to_numpy()
        neg_u = user_pos[user_pos == 0].index.to_numpy()
        rng.shuffle(pos_u)
        rng.shuffle(neg_u)

        def _take_val(users):
            n_val = int(round(len(users) * self.val_frac))
            if len(users) >= 2:
                n_val = min(max(n_val, 1), len(users) - 1)
            else:
                n_val = 0
            return set(users[:n_val]), set(users[n_val:])

        val_pos, tr_pos = _take_val(pos_u)
        val_neg, tr_neg = _take_val(neg_u)
        return tr_pos | tr_neg, val_pos | val_neg

    def _day_loss(self, logits, y, pw):
        import torch.nn.functional as F
        if self.loss == "focal":
            return focal_bce_with_logits(
                logits, y, pos_weight=pw, gamma=self.focal_gamma,
            )
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

    def _repr(self, s_u, x):
        import torch
        if self.variant == "A2":
            r = s_u
        else:
            r = torch.cat([x, s_u], dim=-1)
        if self.proj is not None:
            return self.proj(r)
        return r

    def _score_vecs(self, s_u, x):
        return self.head(self._repr(s_u, x))

    def _stream_day(self, ts, day_idx):
        """Fold one day's edges into TGN memory (standard forward; no edge BCE)."""
        for b in self._batches(ts, day_idx):
            self.model(b.u, b.h, b.a, b.dt)

    def _maybe_reset_memory_for_day(self):
        """When memory is OFF: static (per-day) aggregation — no cross-day state."""
        if not self.use_memory and self.model is not None:
            self.model.reset_memory()

    def fit(self, df_train: pd.DataFrame, df_tgt=None):
        """Day-level supervised fit. ``df_tgt`` enables DANN-UDA when use_dann."""
        torch = self._torch()
        import torch.nn as nn
        import torch.nn.functional as F
        from ..models.temporal_gnn import TemporalHeteroGNN, self_supervised_loss
        from ..models.ud_head import UserDayMLP
        from ..models.dann import DomainAdversary, lambda_schedule
        from ..utils.seed import set_seed
        set_seed(self.seed)

        feat_all = self._features_for(df_train)
        train_users, val_users = self._split_train_val_users(feat_all)
        feat_tr = feat_all[feat_all[USER].isin(train_users)]
        feat_va = feat_all[feat_all[USER].isin(val_users)]
        self._fit_scaler(feat_tr)

        ts = self._tensors(df_train, feat=feat_all)
        feat_dim = len(self._feat_cols) if self.use_features else 0
        self.model = TemporalHeteroGNN(
            ts["n_users"], ts["n_hosts"], n_actions=len(ACTIONS),
            mem_dim=self.mem_dim, feat_dim=0,  # head is external
            use_time_encoding=self.use_time_encoding,
        ).to(self._device)
        # use_memory=False → static (per-day) aggregation: still read s_u after
        # streaming that day's edges, but never carry memory across days.
        raw_dim = self.mem_dim + (feat_dim if self.use_features else 0)
        # Learnable projection so DANN can push domain-invariant day reps.
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, raw_dim), nn.ReLU(),
        ).to(self._device)
        self.head = UserDayMLP(
            raw_dim, hidden=self.hidden, dropout=self.dropout,
        ).to(self._device)
        self.adv = None
        tt = None
        if self.use_dann and df_tgt is not None:
            # Target features: join target full-stream table if set, else build;
            # transform with SOURCE scaler (no target label leakage).
            prev = self._full_feat
            # caller should set_full_features(source) before fit; for target
            # scoring we swap later — here build tgt feat from df_tgt events
            # using whatever full feat is currently attached if it covers tgt.
            tt_feat = user_day_features(df_tgt, deviation=True)
            # temporarily use source scaler + tgt feat frame
            saved = self._full_feat
            self._full_feat = tt_feat
            tt = self._tensors(df_tgt, feat=tt_feat)
            self._full_feat = saved
            self.adv = DomainAdversary(raw_dim).to(self._device)
            log.info("DANN-UDA: target edges=%d users=%d", tt["n_edges"], tt["n_users"])

        n_ssl = int(self.ssl_epochs) if self.use_ssl_pretrain else 0
        log.info(
            "DaySupervisedGNN variant=%s in_dim=%d loss=%s epochs=%d dann=%s "
            "memory=%s ssl=%s(epochs=%d) time_enc=%s deviation=%s",
            self.variant, raw_dim, self.loss, self.epochs, bool(self.adv),
            self.use_memory, self.use_ssl_pretrain, n_ssl,
            self.use_time_encoding, self.use_deviation_features,
        )

        # SSL pretrain (label-free next-action); skipped when use_ssl_pretrain=False
        if n_ssl > 0:
            opt_ssl = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=1e-4,
            )
            for _ in range(n_ssl):
                self.model.set_graph(ts["n_users"], ts["n_hosts"])
                self.model.reset_memory()
                n = ts["n_edges"]
                for i in range(0, n, self.batch_edges):
                    j = min(i + self.batch_edges, n)
                    sl = np.arange(i, j)
                    b = Batch(
                        ts["u"][sl], ts["h"][sl], ts["a"][sl],
                        ts["dt"][sl], ts["y"][sl],
                    )
                    opt_ssl.zero_grad()
                    loss = self_supervised_loss(self.model, b.u, b.h, b.a, b.dt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt_ssl.step()

        pw = None
        prior = 0.5
        if self.pos_weight:
            y_ud = feat_tr[LABEL].to_numpy()
            n_pos = float(y_ud.sum())
            n_neg = float(len(y_ud) - n_pos)
            raw = n_neg / max(n_pos, 1.0)
            ratio = float(min(raw, self.max_pos_weight))
            prior = max(n_pos / max(n_pos + n_neg, 1.0), 1e-6)
            pw = torch.tensor([ratio], device=self._device)
            log.info(
                "day-BCE pos_weight=%.1f (raw=%.1f n_pos=%.0f n_neg=%.0f)",
                ratio, raw, n_pos, n_neg,
            )
        with torch.no_grad():
            self.head.net[-1].bias.fill_(float(np.log(prior / (1.0 - prior))))

        # Freeze TGN during day-supervision: memory is a (SSL) feature; head learns
        # the day-level decision. Avoids AdamW decay wiping SSL weights with no grad.
        for p in self.model.parameters():
            p.requires_grad_(False)
        params = list(self.head.parameters()) + list(self.proj.parameters())
        if self.adv is not None:
            params += list(self.adv.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        if self.lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        elif self.lr_schedule == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        else:
            sched = None

        unique_days = pd.unique(ts["day"])
        tgt_days = pd.unique(tt["day"]) if tt is not None else None
        best_state = None
        best_pr = -1.0
        bad = 0
        total_steps = max(1, self.epochs * max(1, len(unique_days)))
        step = 0

        for epoch in range(self.epochs):
            self.model.eval()  # frozen encoder; still updates memory buffers
            self.head.train()
            self.proj.train()
            if self.adv is not None:
                self.adv.train()
            self.model.set_graph(ts["n_users"], ts["n_hosts"])
            self.model.reset_memory()
            tgt_snap = None
            if tt is not None:
                # isolated target memory bank
                self.model.set_graph(tt["n_users"], tt["n_hosts"])
                self.model.reset_memory()
                tgt_snap = (
                    self.model.user_mem.memory.detach().clone(),
                    self.model.user_mem.last_t.detach().clone(),
                    self.model.host_mem.memory.detach().clone(),
                    self.model.host_mem.last_t.detach().clone(),
                )
                self.model.set_graph(ts["n_users"], ts["n_hosts"])
                self.model.reset_memory()
            epoch_loss = 0.0
            n_steps = 0
            tgt_day_i = 0
            for day_val in unique_days:
                day_idx = np.flatnonzero(ts["day"] == day_val)
                if day_idx.size == 0:
                    continue
                # Stream day edges (memory updates detach per TGN). Day-level
                # loss trains the MLP head (+ SSL already trained the encoder).
                # Memory OFF: reset before each day → static within-day aggregate.
                self._maybe_reset_memory_for_day()
                with torch.no_grad():
                    self._stream_day(ts, day_idx)
                ud = self._ud_for_day(ts, day_val, user_mask=train_users)
                if ud is None:
                    continue
                opt.zero_grad()
                s_u = self.model.user_mem.memory[ud["u"]].detach()
                z = self._repr(s_u, ud["x"])
                logits = self.head(z)
                loss = self._day_loss(logits, ud["y"], pw)
                if self.adv is not None and tt is not None:
                    lam = lambda_schedule(step, total_steps)
                    dsrc = self.adv(z, lam)
                    loss = loss + F.cross_entropy(
                        dsrc,
                        torch.zeros(z.size(0), dtype=torch.long, device=self._device),
                    )
                    # one target day step under isolated memory
                    td = tgt_days[tgt_day_i % len(tgt_days)]
                    tgt_day_i += 1
                    src_snap = (
                        self.model.user_mem.memory.detach().clone(),
                        self.model.user_mem.last_t.detach().clone(),
                        self.model.host_mem.memory.detach().clone(),
                        self.model.host_mem.last_t.detach().clone(),
                    )
                    um, ut, hm, ht = tgt_snap
                    self.model.set_graph(tt["n_users"], tt["n_hosts"])
                    self.model.user_mem.memory = um
                    self.model.user_mem.last_t = ut
                    self.model.host_mem.memory = hm
                    self.model.host_mem.last_t = ht
                    with torch.no_grad():
                        t_idx = np.flatnonzero(tt["day"] == td)
                        if t_idx.size:
                            self._stream_day(tt, t_idx)
                    tud = self._ud_for_day(tt, td)
                    if tud is not None:
                        ts_u = self.model.user_mem.memory[tud["u"]].detach()
                        zt = self._repr(ts_u, tud["x"])
                        dtg = self.adv(zt, lam)
                        loss = loss + F.cross_entropy(
                            dtg,
                            torch.ones(zt.size(0), dtype=torch.long, device=self._device),
                        )
                    tgt_snap = (
                        self.model.user_mem.memory.detach().clone(),
                        self.model.user_mem.last_t.detach().clone(),
                        self.model.host_mem.memory.detach().clone(),
                        self.model.host_mem.last_t.detach().clone(),
                    )
                    um, ut, hm, ht = src_snap
                    self.model.set_graph(ts["n_users"], ts["n_hosts"])
                    self.model.user_mem.memory = um
                    self.model.user_mem.last_t = ut
                    self.model.host_mem.memory = hm
                    self.model.host_mem.last_t = ht
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                n_steps += 1
                step += 1
            if sched is not None:
                sched.step()

            # validation PR-AUC (same streaming, val users only) — every val_every epochs
            do_val = ((epoch + 1) % self.val_every == 0) or (epoch + 1 == self.epochs)
            if do_val:
                val_pr = self._eval_pr(ts, unique_days, val_users)
            else:
                val_pr = best_pr if best_pr >= 0 else float("nan")
            log.info(
                "epoch %d/%d loss=%.4f val_pr=%.4f",
                epoch + 1, self.epochs,
                epoch_loss / max(n_steps, 1), val_pr,
            )
            if do_val and np.isfinite(val_pr) and val_pr > best_pr + 1e-4:
                best_pr = val_pr
                bad = 0
                best_state = {
                    "model": {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()},
                    "head": {k: v.detach().cpu().clone()
                             for k, v in self.head.state_dict().items()},
                    "proj": {k: v.detach().cpu().clone()
                             for k, v in self.proj.state_dict().items()},
                    "epoch": epoch + 1,
                }
            elif do_val:
                bad += self.val_every
            if epoch + 1 >= self.min_epochs and bad >= self.patience:
                log.info("early stop at epoch %d (best=%d pr=%.4f)",
                         epoch + 1, best_state["epoch"] if best_state else -1, best_pr)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.head.load_state_dict(best_state["head"])
            if "proj" in best_state and self.proj is not None:
                self.proj.load_state_dict(best_state["proj"])
            self.best_epoch = best_state["epoch"]
            self.best_val_pr = best_pr
            self.model.to(self._device)
            self.head.to(self._device)
            if self.proj is not None:
                self.proj.to(self._device)
        return self

    def _eval_pr(self, ts, unique_days, users) -> float:
        torch = self._torch()
        self.model.eval()
        self.head.eval()
        # Fresh memory stream for val (do not touch training-time memory mid-epoch:
        # we reset at the start of every training epoch anyway).
        self.model.set_graph(ts["n_users"], ts["n_hosts"])
        self.model.reset_memory()
        ys, ss = [], []
        with torch.no_grad():
            for day_val in unique_days:
                day_idx = np.flatnonzero(ts["day"] == day_val)
                self._maybe_reset_memory_for_day()
                for b in self._batches(ts, day_idx):
                    self.model(b.u, b.h, b.a, b.dt)
                ud = self._ud_for_day(ts, day_val, user_mask=users)
                if ud is None:
                    continue
                s_u = self.model.user_mem.memory[ud["u"]]
                logits = self._score_vecs(s_u, ud["x"])
                ys.append(ud["label"])
                ss.append(torch.sigmoid(logits).cpu().numpy())
        self.model.train()
        self.head.train()
        if not ys:
            return float("nan")
        y = np.concatenate(ys)
        s = np.concatenate(ss)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(compute_metrics(y, s)["pr_auc"])

    def score_user_day(self, df: pd.DataFrame) -> pd.DataFrame:
        torch = self._torch()
        feat = self._features_for(df)
        ts = self._tensors(df, feat=feat)
        self.model.set_graph(ts["n_users"], ts["n_hosts"])
        self.model.reset_memory()
        self.model.eval()
        self.head.eval()
        if self.proj is not None:
            self.proj.eval()
        rows = []
        with torch.no_grad():
            for day_val in pd.unique(ts["day"]):
                day_idx = np.flatnonzero(ts["day"] == day_val)
                self._maybe_reset_memory_for_day()
                for b in self._batches(ts, day_idx):
                    self.model(b.u, b.h, b.a, b.dt)
                ud = self._ud_for_day(ts, day_val)
                if ud is None:
                    continue
                s_u = self.model.user_mem.memory[ud["u"]]
                logits = self._score_vecs(s_u, ud["x"])
                sc = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(sc)):
                    rows.append({
                        USER: ud["users"][i],
                        "day": ud["day"][i],
                        "score": float(sc[i]),
                        "label": int(ud["label"][i]),
                    })
        return pd.DataFrame(rows)


def eval_day_gnn_split(
    df: pd.DataFrame,
    full_feat: pd.DataFrame,
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    **kw,
) -> dict:
    """Fit DaySupervisedGNN on train_df, score test_df, return metrics+lift."""
    det = DaySupervisedGNN(variant=variant, **kw)
    det.set_full_features(full_feat)
    det.fit(train_df)
    agg = det.score_user_day(test_df)
    y = agg["label"].to_numpy()
    m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
    m.update({
        "split": split_name,
        "variant": variant,
        "best_epoch": det.best_epoch,
        "best_val_pr": det.best_val_pr,
        "loss": det.loss,
    })
    return m
