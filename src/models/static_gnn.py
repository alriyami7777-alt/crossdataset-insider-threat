"""
Static graph baselines (GCN) and an ATHITD-inspired attention model.

Pure-torch implementations (no torch_geometric required). Both expose
detector wrappers that consume the same user-day feature table used by RF/GNN
and score at user-day granularity for the honest transfer matrix.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from ..data.schema import LABEL, USER


class StaticGCN(nn.Module):
    """Homogeneous GCN over a (normalized) adjacency — pure torch."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.weights = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layers)]
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.weights):
            h = adj @ lin(h)
            if i < len(self.weights) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, adj)).squeeze(-1)


class ATHITDStyle(nn.Module):
    """Attention temporal-heterogeneous style baseline (compact reimplementation).

    Per user-day: attend over a short sequence of (scaled) day features for that
    user, then classify. Captures the ATHITD idea of relation/temporal attention
    without requiring the original codebase.
    """

    def __init__(self, in_dim: int, emb: int = 64, nhead: int = 4, max_ctx: int = 8):
        super().__init__()
        self.max_ctx = max_ctx
        self.proj = nn.Linear(in_dim, emb)
        self.attn = nn.MultiheadAttention(emb, num_heads=nhead, batch_first=True)
        self.head = nn.Linear(emb * 2, 1)

    def forward(self, x_ctx: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        """x_ctx: [B,T,D]; x_cur: [B,D]."""
        q = self.proj(x_cur).unsqueeze(1)
        k = self.proj(x_ctx)
        ctx, _ = self.attn(q, k, k)
        h = torch.cat([ctx.squeeze(1), self.proj(x_cur)], dim=-1)
        return self.head(h).squeeze(-1)


def _scale_xy(
    feat: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = False,
):
    """StandardScaler on base counts only; deviation columns clipped as-is."""
    base = list(BASE_FEATURE_COLUMNS)
    cols = list(FEATURE_COLUMNS)
    xb = feat[base].to_numpy(dtype=np.float64)
    if fit:
        scaler = StandardScaler()
        scaler.fit(xb)
    assert scaler is not None
    xb = np.clip(scaler.transform(xb), -8.0, 8.0)
    dev = [c for c in cols if c.startswith("dev_")]
    if dev:
        xd = np.clip(feat[dev].to_numpy(dtype=np.float32), -8.0, 8.0)
        X = np.concatenate([xb.astype(np.float32), xd], axis=1)
    else:
        X = xb.astype(np.float32)
    y = feat[LABEL].to_numpy(dtype=np.float32) if LABEL in feat.columns else None
    return X, y, scaler


def _norm_adj(n: int, edges: np.ndarray, device: str) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops (dense; compact n)."""
    A = np.eye(n, dtype=np.float32)
    if edges.size:
        A[edges[:, 0], edges[:, 1]] = 1.0
        A[edges[:, 1], edges[:, 0]] = 1.0
    deg = A.sum(axis=1, keepdims=True).clip(min=1.0)
    dinv = 1.0 / np.sqrt(deg)
    A = (dinv * A) * dinv.T
    return torch.tensor(A, device=device)


class StaticGCNDetector:
    """Supervised GCN on a kNN graph of user-day features (inductive at test)."""

    def __init__(
        self,
        hidden: int = 64,
        k: int = 8,
        epochs: int = 40,
        lr: float = 1e-3,
        max_train_nodes: int = 25000,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.hidden = hidden
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.max_train_nodes = max_train_nodes
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[StaticGCN] = None
        self._nn: Optional[NearestNeighbors] = None
        self._Xtr: Optional[np.ndarray] = None

    def fit(self, feat_train: pd.DataFrame, y=None):
        from ..utils.seed import set_seed
        set_seed(self.seed)
        feat_train = feat_train.reset_index(drop=True)
        X, y_arr, self.scaler = _scale_xy(feat_train, fit=True)
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float32)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        if n > self.max_train_nodes:
            pos = np.where(y_arr == 1)[0]
            neg = np.where(y_arr == 0)[0]
            n_pos = min(len(pos), max(200, self.max_train_nodes // 20))
            n_neg = self.max_train_nodes - n_pos
            take = np.concatenate([
                rng.choice(pos, size=n_pos, replace=False) if len(pos) else pos,
                rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
            ])
            X, y_arr = X[take], y_arr[take]
            n = len(X)
        self._Xtr = X
        self._nn = NearestNeighbors(n_neighbors=min(self.k + 1, n), algorithm="auto")
        self._nn.fit(X)
        nbrs = self._nn.kneighbors(X, return_distance=False)
        edges = [(i, int(j)) for i, row in enumerate(nbrs) for j in row if j != i]
        edges_arr = (
            np.asarray(edges, dtype=np.int64) if edges
            else np.zeros((0, 2), dtype=np.int64)
        )
        adj = _norm_adj(n, edges_arr, self.device)
        xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(y_arr, device=self.device)
        self.model = StaticGCN(X.shape[1], hidden=self.hidden).to(self.device)
        n_pos = float(y_arr.sum())
        n_neg = float(len(y_arr) - n_pos)
        pw = torch.tensor([min(n_neg / max(n_pos, 1.0), 100.0)], device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            logits = self.model(xt, adj)
            loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pw)
            loss.backward()
            opt.step()
        return self

    def anomaly_scores(self, feat: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.scaler is not None and self._nn is not None
        X, _, _ = _scale_xy(feat, scaler=self.scaler, fit=False)
        self.model.eval()
        k = min(self.k, len(self._Xtr))
        nbrs = self._nn.kneighbors(X, n_neighbors=k, return_distance=False)
        scores = np.zeros(len(X), dtype=np.float64)
        bs = 2048
        with torch.no_grad():
            for i0 in range(0, len(X), bs):
                sl = slice(i0, min(i0 + bs, len(X)))
                xb = X[sl]
                agg = self._Xtr[nbrs[sl]].mean(axis=1)
                B, D = xb.shape
                nodes = np.stack([xb, agg], axis=1).reshape(B * 2, D)
                adj = torch.zeros(B * 2, B * 2, device=self.device)
                idx = torch.arange(B, device=self.device)
                i = 2 * idx
                j = i + 1
                adj[i, i] = adj[j, j] = adj[i, j] = adj[j, i] = 1.0
                deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
                adj = adj / torch.sqrt(deg) / torch.sqrt(deg.transpose(0, 1))
                xt = torch.tensor(nodes, device=self.device)
                logits = self.model(xt, adj)
                scores[sl] = torch.sigmoid(logits[0::2]).cpu().numpy()
        return scores


def _build_user_day_context(
    feat: pd.DataFrame, X: np.ndarray, max_ctx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row context window of prior+current day features within each user."""
    f = feat.reset_index(drop=True)
    f2 = f[[USER, "day"]].copy()
    f2["_i"] = np.arange(len(f2))
    f2 = f2.sort_values([USER, "day"], kind="mergesort")
    idx_sorted = f2["_i"].to_numpy()
    Xs = X[idx_sorted]
    users = f.iloc[idx_sorted][USER].to_numpy()
    ys = f.iloc[idx_sorted][LABEL].to_numpy(dtype=np.float32)
    T, D = max_ctx, X.shape[1]
    ctx_s = np.zeros((len(Xs), T, D), dtype=np.float32)
    start = 0
    while start < len(Xs):
        u = users[start]
        end = start
        while end < len(Xs) and users[end] == u:
            end += 1
        for t in range(start, end):
            lo = max(start, t - T + 1)
            w = Xs[lo:t + 1]
            ctx_s[t, -len(w):] = w
        start = end
    ctx = np.zeros_like(ctx_s)
    cur = np.zeros_like(Xs)
    y = np.zeros_like(ys)
    ctx[idx_sorted] = ctx_s
    cur[idx_sorted] = Xs
    y[idx_sorted] = ys
    return ctx, cur, y


class ATHITDDetector:
    """Supervised ATHITD-style attention over each user's recent day context."""

    def __init__(
        self,
        emb: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        max_ctx: int = 8,
        batch_size: int = 1024,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.emb = emb
        self.epochs = epochs
        self.lr = lr
        self.max_ctx = max_ctx
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[ATHITDStyle] = None

    def fit(self, feat_train: pd.DataFrame, y=None):
        from ..utils.seed import set_seed
        set_seed(self.seed)
        feat_train = feat_train.reset_index(drop=True)
        X, y_arr, self.scaler = _scale_xy(feat_train, fit=True)
        ctx, cur, y_built = _build_user_day_context(feat_train, X, self.max_ctx)
        if y is not None:
            y_built = np.asarray(y, dtype=np.float32)
        self.model = ATHITDStyle(
            X.shape[1], emb=self.emb, max_ctx=self.max_ctx,
        ).to(self.device)
        n_pos = float(y_built.sum())
        n_neg = float(len(y_built) - n_pos)
        pw = torch.tensor([min(n_neg / max(n_pos, 1.0), 100.0)], device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(y_built)
        for ep in range(self.epochs):
            perm = np.random.default_rng(self.seed + ep).permutation(n)
            self.model.train()
            for i in range(0, n, self.batch_size):
                sl = perm[i:i + self.batch_size]
                opt.zero_grad()
                logits = self.model(
                    torch.tensor(ctx[sl], device=self.device),
                    torch.tensor(cur[sl], device=self.device),
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    torch.tensor(y_built[sl], device=self.device),
                    pos_weight=pw,
                )
                loss.backward()
                opt.step()
        return self

    def anomaly_scores(self, feat: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.scaler is not None
        feat = feat.reset_index(drop=True)
        X, _, _ = _scale_xy(feat, scaler=self.scaler, fit=False)
        ctx, cur, _ = _build_user_day_context(feat, X, self.max_ctx)
        self.model.eval()
        out = np.zeros(len(X), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                sl = slice(i, i + self.batch_size)
                logits = self.model(
                    torch.tensor(ctx[sl], device=self.device),
                    torch.tensor(cur[sl], device=self.device),
                )
                out[sl] = torch.sigmoid(logits).cpu().numpy()
        return out
