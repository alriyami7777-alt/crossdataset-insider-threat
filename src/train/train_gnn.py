"""
Trainer for the temporal-GNN detector: self-supervised pretraining -> supervised
fine-tuning -> optional domain-adversarial (DANN) generalization, plus a cross-
dataset runner that produces the source x target PR-AUC matrix (Table II in the
paper).

Requires torch. Runs on the GPU box / Cursor -- NOT in the network-locked cloud
sandbox (which has no torch). Everything starts from a canonical dataframe, so
swapping datasets is a one-line change and the model is inductive across datasets
(memory is re-sized per graph; learned weights transfer).

Usage (in Cursor):
    from src.data.loaders import load
    from src.train.train_gnn import run_cross_dataset_gnn, run_cross_dataset_gnn_settings
    dfs = {"spedia": load("spedia","logs_SPEDIA.csv"),
           "synthB": load("synthB")}
    # single setting (backward-compatible)
    matrix, details = run_cross_dataset_gnn(dfs, epochs=15, ssl_epochs=5)
    # locked design: report BOTH zero-shot and DANN-UDA
    both = run_cross_dataset_gnn_settings(dfs, epochs=15, ssl_epochs=5)
"""
from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import pandas as pd

from ..data.features import (
    BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, user_day_features,
)
from ..data.graph_builder import edge_stream
from ..data.schema import ACTIONS, USER, TIMESTAMP, LABEL
from .domain_concat import tag_and_concat
from .evaluate import compute_metrics
from .id_eval import id_eval_table

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "u h a dt y")


class TemporalGNNDetector:
    """Wraps the temporal-hetero GNN with SSL + DANN training and user-day scoring.

    When ``use_ud_features=True`` (default), the user-day readout is
    ``r_{u,d} = [s_u || x_{u,d}]`` with causal deviation-augmented ``x_{u,d}``.
    """

    def __init__(self, mem_dim=64, epochs=15, ssl_epochs=5, lr=1e-3,
                 batch_edges=4096, use_dann=False, pos_weight=True,
                 use_ud_features=True, deviation=True,
                 device=None, seed=7):
        self.mem_dim = mem_dim
        self.epochs = epochs
        self.ssl_epochs = ssl_epochs
        self.lr = lr
        self.batch_edges = batch_edges
        self.use_dann = use_dann
        # If True, BCE uses pos_weight = n_neg/n_pos on the source stream
        # (critical for CERT-scale imbalance ~0.4% positives).
        self.pos_weight = pos_weight
        self.use_ud_features = use_ud_features
        self.deviation = deviation
        self.seed = seed
        self._device = device
        self.model = None
        self._feat_cols = (
            list(FEATURE_COLUMNS) if deviation else list(BASE_FEATURE_COLUMNS)
        )

    # -- lazy torch setup (import here so the module loads without torch) -------
    def _torch(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def _tensors(self, df):
        torch = self._torch()
        e, uidx, hidx = edge_stream(df)
        t = e["t"].to_numpy().astype(np.float64)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)     # normalized time in [0,1]
        # Edge calendar days in the same mergesort order as edge_stream
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
            "n_users": len(uidx), "n_hosts": len(hidx), "n_edges": len(e),
            "uidx": uidx,
        }
        if self.use_ud_features:
            feat = user_day_features(df, deviation=self.deviation)
            feat = feat.copy()
            feat["_u"] = feat[USER].map(uidx)
            n_drop = int(feat["_u"].isna().sum())
            if n_drop:
                log.warning("user-day features: dropping %d rows with unknown user", n_drop)
                feat = feat.dropna(subset=["_u"])
            feat["_u"] = feat["_u"].astype(int)
            out["feat"] = feat
            out["feat_cols"] = list(self._feat_cols)
        return out

    def _batches(self, ts):
        n = ts["n_edges"]
        for i in range(0, n, self.batch_edges):
            j = min(i + self.batch_edges, n)
            yield Batch(ts["u"][i:j], ts["h"][i:j], ts["a"][i:j],
                        ts["dt"][i:j], ts["y"][i:j])

    def _snapshot_memory(self):
        """Clone current user/host memory buffers (source or target bank)."""
        return (
            self.model.user_mem.memory.detach().clone(),
            self.model.user_mem.last_t.detach().clone(),
            self.model.host_mem.memory.detach().clone(),
            self.model.host_mem.last_t.detach().clone(),
        )

    def _restore_memory(self, snap):
        um, ut, hm, ht = snap
        self.model.set_graph(um.shape[0], hm.shape[0])
        self.model.user_mem.memory = um
        self.model.user_mem.last_t = ut
        self.model.host_mem.memory = hm
        self.model.host_mem.last_t = ht

    def _target_edge_emb(self, tb, tgt_n_users, tgt_n_hosts, tgt_snap):
        """Forward a target batch under an isolated memory bank.

        Cross-dataset DANN cannot share the source memory table: target node ids
        are a different index space and may be larger (OOB) or collide (silent
        corruption). We stash source memory, swap in the target bank, embed, then
        restore source memory so the TGN stream stays consistent.
        """
        src_snap = self._snapshot_memory()
        self._restore_memory(tgt_snap)
        # Ensure sizes match the target graph even if snap was empty/fresh
        if (self.model.user_mem.memory.shape[0] != tgt_n_users
                or self.model.host_mem.memory.shape[0] != tgt_n_hosts):
            self.model.set_graph(tgt_n_users, tgt_n_hosts)
            self.model.reset_memory()
        _, temb = self.model(tb.u, tb.h, tb.a, tb.dt)
        new_tgt_snap = self._snapshot_memory()
        self._restore_memory(src_snap)
        return temb, new_tgt_snap

    def _stream_day_edges(self, ts, day_idx):
        """Yield edge batches for one calendar day (integer positions)."""
        for i in range(0, len(day_idx), self.batch_edges):
            sl = day_idx[i:i + self.batch_edges]
            yield Batch(
                ts["u"][sl], ts["h"][sl], ts["a"][sl], ts["dt"][sl], ts["y"][sl],
            )

    def _ud_tensors_for_day(self, ts, day_val):
        """User-day feature/label tensors for one calendar day."""
        torch = self._torch()
        feat = ts["feat"]
        fday = feat[feat["day"] == day_val]
        if len(fday) == 0:
            return None
        cols = ts["feat_cols"]
        return {
            "u": torch.tensor(fday["_u"].to_numpy(), dtype=torch.long, device=self._device),
            "x": torch.tensor(
                fday[cols].to_numpy(dtype=np.float32), dtype=torch.float, device=self._device
            ),
            "y": torch.tensor(
                fday[LABEL].to_numpy(), dtype=torch.float, device=self._device
            ),
            "users": fday[USER].to_numpy(),
            "day": fday["day"].to_numpy(),
            "label": fday[LABEL].to_numpy(),
        }

    # -- training --------------------------------------------------------------
    def fit(self, df_src, df_tgt=None):
        torch = self._torch()
        import torch.nn.functional as F
        from ..models.temporal_gnn import TemporalHeteroGNN, self_supervised_loss
        from ..models.dann import DomainAdversary, lambda_schedule
        from ..utils.seed import set_seed
        set_seed(self.seed)

        ts = self._tensors(df_src)
        feat_dim = len(self._feat_cols) if self.use_ud_features else 0
        self.model = TemporalHeteroGNN(
            ts["n_users"], ts["n_hosts"], n_actions=len(ACTIONS),
            mem_dim=self.mem_dim, feat_dim=feat_dim,
        ).to(self._device)
        log.info(
            "TemporalGNN: use_ud_features=%s deviation=%s feat_dim=%d",
            self.use_ud_features, self.deviation, feat_dim,
        )

        # 1) self-supervised pretraining (label-free, edge-level)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.ssl_epochs):
            self.model.set_graph(ts["n_users"], ts["n_hosts"]); self.model.reset_memory()
            for b in self._batches(ts):
                opt.zero_grad()
                loss = self_supervised_loss(self.model, b.u, b.h, b.a, b.dt)
                loss.backward(); opt.step()

        # 2) supervised fine-tuning (+ optional DANN)
        adv, tt = None, None
        tgt_snap = None
        params = list(self.model.parameters())
        if self.use_dann and df_tgt is not None:
            adv = DomainAdversary(2 * self.mem_dim).to(self._device)
            params += list(adv.parameters())
            tt = self._tensors(df_tgt)
            log.info(
                "DANN: source nodes u=%d h=%d | target nodes u=%d h=%d "
                "(isolated target memory bank)",
                ts["n_users"], ts["n_hosts"], tt["n_users"], tt["n_hosts"],
            )
        opt = torch.optim.Adam(params, lr=self.lr)

        # pos_weight from user-day labels when using ud readout, else edge labels
        pw = None
        if self.pos_weight:
            if self.use_ud_features:
                y_ud = ts["feat"][LABEL].to_numpy()
                n_pos = float(y_ud.sum()); n_neg = float(len(y_ud) - n_pos)
            else:
                n_pos = float(ts["y"].float().sum().item())
                n_neg = float(ts["n_edges"] - n_pos)
            ratio = n_neg / max(n_pos, 1.0)
            pw = torch.tensor([ratio], device=self._device)
            log.info(
                "BCE pos_weight=%.1f (n_pos=%.0f n_neg=%.0f, level=%s)",
                ratio, n_pos, n_neg,
                "user-day" if self.use_ud_features else "edge",
            )

        unique_days = pd.unique(ts["day"])  # chronological for datetime64
        n_day_steps = max(1, len(unique_days))
        total_steps = self.epochs * n_day_steps
        step = 0
        for _ in range(self.epochs):
            self.model.set_graph(ts["n_users"], ts["n_hosts"]); self.model.reset_memory()
            tgt_iter = iter(self._batches(tt)) if tt is not None else None
            if tt is not None:
                self.model.set_graph(tt["n_users"], tt["n_hosts"])
                self.model.reset_memory()
                tgt_snap = self._snapshot_memory()
                self.model.set_graph(ts["n_users"], ts["n_hosts"])
                self.model.reset_memory()

            for day_val in unique_days:
                day_idx = np.flatnonzero(ts["day"] == day_val)
                if day_idx.size == 0:
                    continue
                opt.zero_grad()
                loss = torch.zeros((), device=self._device)
                last_emb = None
                # fold today's edges into memory (+ edge BCE keeps encoder trained)
                for b in self._stream_day_edges(ts, day_idx):
                    logit_e, emb = self.model(b.u, b.h, b.a, b.dt)
                    last_emb = emb
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logit_e, b.y.float(), pos_weight=pw
                    )
                # user-day readout r = [s_u || x_{u,d}]
                if self.use_ud_features:
                    ud = self._ud_tensors_for_day(ts, day_val)
                    if ud is not None:
                        s_u = self.model.user_mem.memory[ud["u"]]
                        logit_ud = self.model.score_user_day_vec(s_u, ud["x"])
                        loss = loss + F.binary_cross_entropy_with_logits(
                            logit_ud, ud["y"], pos_weight=pw
                        )
                if adv is not None and last_emb is not None:
                    lam = lambda_schedule(step, total_steps)
                    dsrc = adv(last_emb, lam)
                    loss = loss + F.cross_entropy(
                        dsrc,
                        torch.zeros(last_emb.size(0), dtype=torch.long, device=self._device),
                    )
                    try:
                        tb = next(tgt_iter)
                    except (StopIteration, TypeError):
                        tgt_iter = iter(self._batches(tt)); tb = next(tgt_iter)
                    temb, tgt_snap = self._target_edge_emb(
                        tb, tt["n_users"], tt["n_hosts"], tgt_snap
                    )
                    dtg = adv(temb, lam)
                    loss = loss + F.cross_entropy(
                        dtg,
                        torch.ones(temb.size(0), dtype=torch.long, device=self._device),
                    )
                loss.backward(); opt.step()
                step += 1
        return self

    # -- scoring ---------------------------------------------------------------
    def score_user_day(self, df):
        """Inductive user-day scores. With ``use_ud_features``, uses r=[s_u||x_ud]."""
        torch = self._torch()
        ts = self._tensors(df)
        self.model.set_graph(ts["n_users"], ts["n_hosts"]); self.model.reset_memory()

        if not self.use_ud_features:
            scores = []
            with torch.no_grad():
                for b in self._batches(ts):
                    logit, _ = self.model(b.u, b.h, b.a, b.dt)
                    scores.append(torch.sigmoid(logit).cpu().numpy())
            edge_scores = np.concatenate(scores) if scores else np.array([])
            d = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True).copy()
            d["score"] = edge_scores
            d["day"] = d[TIMESTAMP].dt.floor("D")
            return d.groupby([USER, "day"]).agg(
                score=("score", "max"), label=(LABEL, "max")).reset_index()

        rows = []
        with torch.no_grad():
            for day_val in pd.unique(ts["day"]):
                day_idx = np.flatnonzero(ts["day"] == day_val)
                for b in self._stream_day_edges(ts, day_idx):
                    self.model(b.u, b.h, b.a, b.dt)
                ud = self._ud_tensors_for_day(ts, day_val)
                if ud is None:
                    continue
                s_u = self.model.user_mem.memory[ud["u"]]
                logit = self.model.score_user_day_vec(s_u, ud["x"])
                sc = torch.sigmoid(logit).cpu().numpy()
                for i in range(len(sc)):
                    rows.append({
                        USER: ud["users"][i],
                        "day": ud["day"][i],
                        "score": float(sc[i]),
                        "label": int(ud["label"][i]),
                    })
        return pd.DataFrame(rows)


def run_cross_dataset_gnn(dfs: dict, use_dann=False, **kw):
    """Train on each source, score every target -> (source x target) PR-AUC matrix.

    use_dann=False (default): zero-shot generalization — source labels only.
    use_dann=True: unsupervised domain adaptation — source labels + unlabeled
                   target for the domain adversary (no target insider labels).
    """
    names = list(dfs.keys())
    setting = "dann_uda" if use_dann else "zero_shot"
    log.info("run_cross_dataset_gnn: setting=%s domains=%s", setting, names)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    details = {}
    for s in names:
        for t in names:
            # Diagonal: no domain adversary (source==target has nothing to adapt).
            adapt = bool(use_dann and t != s)
            det = TemporalGNNDetector(use_dann=adapt, **kw)
            det.fit(dfs[s], df_tgt=(dfs[t] if adapt else None))
            agg = det.score_user_day(dfs[t])
            m = compute_metrics(agg["label"].to_numpy(), agg["score"].to_numpy())
            m["setting"] = setting
            details[(s, t)] = m
            matrix.loc[s, t] = m.get("pr_auc", float("nan"))
            log.info(
                "cell %s->%s [%s]: pr_auc=%.3f n=%d pos=%d",
                s, t, setting, m.get("pr_auc", float("nan")),
                m.get("n", 0), m.get("n_pos", 0),
            )
    return matrix, details


def run_cross_dataset_gnn_settings(dfs: dict, **kw):
    """Locked design: report BOTH zero-shot and DANN-UDA matrices.

    Returns
    -------
    dict with keys ``zero_shot`` and ``dann_uda``, each ``(matrix, details)``.
    Also prints both matrices and their generalization gaps.
    """
    from .crossdataset import generalization_gap

    out = {}
    for name, flag in (("zero_shot", False), ("dann_uda", True)):
        print(f"\n=== Cross-dataset GNN: {name} (use_dann={flag}) ===")
        matrix, details = run_cross_dataset_gnn(dfs, use_dann=flag, **kw)
        print(matrix.round(3).to_string())
        gap = generalization_gap(matrix)
        print(f"Generalization gap: {gap:.3f}")
        out[name] = (matrix, details)
    return out


def leave_one_domain_out_gnn(dfs: dict, use_dann=False, **kw):
    """Train on K-1 domains (concatenated), test on the held-out domain.

    When use_dann=True the held-out domain is the *unlabeled* DANN target during
    training (no target insider labels used), then scored for evaluation.
    """
    names = list(dfs.keys())
    if len(names) < 2:
        raise ValueError("leave_one_domain_out_gnn needs at least 2 domains")
    setting = "dann_uda" if use_dann else "zero_shot"
    scores = {}
    details = {}
    for held in names:
        train_dfs = {n: dfs[n] for n in names if n != held}
        train_df = tag_and_concat(train_dfs)
        test_df = tag_and_concat({held: dfs[held]})
        log.info(
            "LODO-GNN [%s] held_out=%s train_domains=%s train_rows=%d test_rows=%d",
            setting, held, list(train_dfs.keys()), len(train_df), len(test_df),
        )
        det = TemporalGNNDetector(use_dann=use_dann, **kw)
        det.fit(train_df, df_tgt=(test_df if use_dann else None))
        agg = det.score_user_day(test_df)
        m = compute_metrics(agg["label"].to_numpy(), agg["score"].to_numpy())
        m["setting"] = setting
        m["held_out"] = held
        details[held] = m
        scores[held] = m.get("pr_auc", float("nan"))
        log.info(
            "LODO-GNN [%s] held=%s pr_auc=%.3f",
            setting, held, scores[held],
        )
    series = pd.Series(scores, name="pr_auc")
    print(f"\n=== Leave-one-domain-out GNN: {setting} ===")
    print(series.round(3).to_string())
    return series, details


def leave_one_domain_out_gnn_settings(dfs: dict, **kw):
    """LODO under both zero-shot and DANN-UDA. Returns dict of (series, details)."""
    return {
        "zero_shot": leave_one_domain_out_gnn(dfs, use_dann=False, **kw),
        "dann_uda": leave_one_domain_out_gnn(dfs, use_dann=True, **kw),
    }


def gnn_id_eval(dfs: dict, **kw) -> pd.DataFrame:
    """In-distribution TemporalGNN eval under temporal + user-disjoint splits."""
    def _factory():
        return TemporalGNNDetector(**kw)

    table = id_eval_table(dfs, detector_factory=_factory)
    print("\n=== In-distribution GNN (temporal + user-disjoint) ===")
    cols = [c for c in ("dataset", "split", "pr_auc", "roc_auc",
                        "dr_at_1pct_fpr", "n", "n_pos") if c in table.columns]
    print(table[cols].round(3).to_string(index=False))
    return table
