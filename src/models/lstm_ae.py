"""
Compact LSTM autoencoder baseline on user-day feature sequences.

Same feature vocabulary as the tabular/GNN path (counts+deviation). Per user,
chronological day vectors form a sequence; reconstruction MSE is the anomaly
score for each day. Trained on TRAIN rows only (scaler fit on train).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from ..data.schema import LABEL, USER


class _LSTMAENet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        z, _ = self.enc(x)
        # decode from last hidden repeated (compact AE)
        h = z[:, -1:, :].expand(-1, x.size(1), -1)
        y, _ = self.dec(h)
        return self.out(y)


class LSTMAEDetector:
    """Unsupervised LSTM-AE over per-user day-feature sequences."""

    def __init__(
        self,
        hidden: int = 64,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_users: int = 64,
        max_len: int = 64,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_users = batch_users
        self.max_len = max_len
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.net: Optional[_LSTMAENet] = None
        self._base = list(BASE_FEATURE_COLUMNS)
        self._cols = list(FEATURE_COLUMNS)

    def _transform(self, feat: pd.DataFrame) -> np.ndarray:
        base = feat[self._base].to_numpy(dtype=np.float64)
        if self.scaler is not None:
            base = self.scaler.transform(base)
        base = np.clip(base, -8.0, 8.0)
        dev = [c for c in self._cols if c.startswith("dev_")]
        if not dev:
            return base.astype(np.float32)
        d = np.clip(feat[dev].to_numpy(dtype=np.float32), -8.0, 8.0)
        return np.concatenate([base.astype(np.float32), d], axis=1)

    def _user_seqs(self, feat: pd.DataFrame, X: np.ndarray):
        """Group rows into per-user chronological sequences (indices into feat)."""
        f = feat.reset_index(drop=True)
        order = np.argsort(f["day"].to_numpy(), kind="mergesort")
        f = f.iloc[order].reset_index(drop=True)
        X = X[order]
        seqs = []
        for _, idx in f.groupby(USER, sort=False).groups.items():
            idx = np.asarray(idx)
            if len(idx) > self.max_len:
                idx = idx[-self.max_len:]
            seqs.append((idx, X[idx]))
        return seqs

    def fit(self, feat_train: pd.DataFrame, y=None):
        from ..utils.seed import set_seed
        set_seed(self.seed)
        feat_train = feat_train.reset_index(drop=True)
        self.scaler = StandardScaler()
        self.scaler.fit(feat_train[self._base].to_numpy(dtype=np.float64))
        X = self._transform(feat_train)
        seqs = self._user_seqs(feat_train, X)
        # Prefer mostly-benign users for AE; fall back to all
        if y is None and LABEL in feat_train.columns:
            user_pos = feat_train.groupby(USER)[LABEL].max()
            keep = set(user_pos[user_pos == 0].index)
            seqs_b = [s for s in seqs if feat_train.iloc[s[0][0]][USER] in keep]
            if len(seqs_b) >= 8:
                seqs = seqs_b
        in_dim = X.shape[1]
        self.net = _LSTMAENet(in_dim, hidden=self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for ep in range(self.epochs):
            rng = np.random.default_rng(self.seed + ep)
            order = rng.permutation(len(seqs))
            total = 0.0
            n = 0
            for i in range(0, len(order), self.batch_users):
                batch = [seqs[j] for j in order[i:i + self.batch_users]]
                # pad to max T in batch
                T = max(len(s[1]) for s in batch)
                D = in_dim
                xb = np.zeros((len(batch), T, D), dtype=np.float32)
                mask = np.zeros((len(batch), T), dtype=np.float32)
                for bi, (_, arr) in enumerate(batch):
                    xb[bi, :len(arr)] = arr
                    mask[bi, :len(arr)] = 1.0
                xt = torch.tensor(xb, device=self.device)
                mt = torch.tensor(mask, device=self.device).unsqueeze(-1)
                opt.zero_grad()
                recon = self.net(xt)
                loss = (((recon - xt) ** 2) * mt).sum() / mt.sum().clamp_min(1.0)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())
                n += 1
        return self

    def anomaly_scores(self, feat: pd.DataFrame) -> np.ndarray:
        assert self.net is not None and self.scaler is not None
        feat = feat.reset_index(drop=True)
        X = self._transform(feat)
        seqs = self._user_seqs(feat, X)
        scores = np.zeros(len(feat), dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            for idx, arr in seqs:
                xt = torch.tensor(arr[None, ...], device=self.device)
                recon = self.net(xt).cpu().numpy()[0]
                mse = ((recon - arr) ** 2).mean(axis=1)
                scores[idx] = mse
        return scores

