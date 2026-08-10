"""
Transformer sequence encoder baseline (cf. user-based sequencing).

Two modes:
  * action-token sequences per user-day (masked LM NLL anomaly score)
  * feature-sequence fallback via a small Transformer on day-feature tokens

Requires torch. Included so the paper compares graph vs sequence models under
the SAME cross-dataset / user-day protocol.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from ..data.schema import ACTION, ACTION_TO_ID, ACTIONS, LABEL, TIMESTAMP, USER


class TransformerEncoderDetector(nn.Module):
    def __init__(self, n_actions=16, d_model=64, nhead=4, layers=2, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(n_actions + 1, d_model)  # +1 PAD
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, n_actions + 1)

    def forward(self, seq, pad_mask=None):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.tok(seq) + self.pos(pos)
        z = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(z)

    @torch.no_grad()
    def anomaly_scores(self, seq, pad_mask=None):
        """Per-sequence reconstruction NLL as anomaly score."""
        logits = self.forward(seq, pad_mask=pad_mask)
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)
        if pad_mask is not None:
            nll = nll.masked_fill(pad_mask, 0.0)
            denom = (~pad_mask).sum(dim=1).clamp_min(1).float()
            return nll.sum(dim=1) / denom
        return nll.mean(dim=1)


class TransformerUDDetector:
    """Unsupervised transformer on action tokens, scored at user-day level.

    For each (user, day), take up to ``max_len`` actions; train to reconstruct
    tokens (teacher forcing); score = mean NLL. Uses TRAIN events only.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        max_len: int = 64,
        epochs: int = 8,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[TransformerEncoderDetector] = None
        self._pad_id = len(ACTIONS)

    def _ud_sequences(self, df: pd.DataFrame):
        d = df.copy()
        d["day"] = d[TIMESTAMP].dt.floor("D")
        d["_aid"] = d[ACTION].map(ACTION_TO_ID).fillna(self._pad_id).astype(int)
        d.loc[d["_aid"] >= self._pad_id, "_aid"] = self._pad_id - 1
        rows = []
        seqs = []
        for (u, day), g in d.groupby([USER, "day"], sort=False):
            aids = g.sort_values(TIMESTAMP, kind="mergesort")["_aid"].to_numpy()
            if len(aids) > self.max_len:
                aids = aids[-self.max_len:]
            lab = int(g[LABEL].max()) if LABEL in g.columns else 0
            rows.append({USER: u, "day": day, LABEL: lab})
            seqs.append(aids)
        return pd.DataFrame(rows), seqs

    def _pad_batch(self, seqs):
        B = len(seqs)
        L = max(len(s) for s in seqs)
        L = max(L, 1)
        arr = np.full((B, L), self._pad_id, dtype=np.int64)
        mask = np.ones((B, L), dtype=bool)
        for i, s in enumerate(seqs):
            if len(s) == 0:
                continue
            arr[i, :len(s)] = s
            mask[i, :len(s)] = False
        return (
            torch.tensor(arr, device=self.device),
            torch.tensor(mask, device=self.device),
        )

    def fit(self, df_train: pd.DataFrame, y=None):
        from ..utils.seed import set_seed
        set_seed(self.seed)
        _, seqs = self._ud_sequences(df_train)
        # keep mostly benign user-days
        meta, seqs2 = self._ud_sequences(df_train)
        yud = meta[LABEL].to_numpy()
        keep = np.where(yud == 0)[0]
        if len(keep) < 16:
            keep = np.arange(len(seqs2))
        seqs = [seqs2[i] for i in keep]
        self.net = TransformerEncoderDetector(
            n_actions=len(ACTIONS),
            d_model=self.d_model,
            nhead=self.nhead,
            layers=self.layers,
            max_len=self.max_len,
        ).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            perm = np.random.default_rng(self.seed + ep).permutation(len(seqs))
            self.net.train()
            for i in range(0, len(perm), self.batch_size):
                batch = [seqs[j] for j in perm[i:i + self.batch_size]]
                xt, mask = self._pad_batch(batch)
                opt.zero_grad()
                logits = self.net(xt, pad_mask=mask)
                # CE ignoring pad
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    xt.reshape(-1),
                    ignore_index=self._pad_id,
                )
                loss.backward()
                opt.step()
        return self

    def score_user_day(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.net is not None
        meta, seqs = self._ud_sequences(df)
        scores = np.zeros(len(seqs), dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                batch = seqs[i:i + self.batch_size]
                xt, mask = self._pad_batch(batch)
                scores[i:i + len(batch)] = (
                    self.net.anomaly_scores(xt, pad_mask=mask).cpu().numpy()
                )
        meta = meta.copy()
        meta["score"] = scores
        return meta


class TransformerFeatureDetector:
    """Supervised Transformer on windows of scaled user-day features."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        max_ctx: int = 8,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 1024,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.max_ctx = max_ctx
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.in_proj: Optional[nn.Linear] = None
        self.enc: Optional[nn.TransformerEncoder] = None
        self.head: Optional[nn.Linear] = None
        self._base = list(BASE_FEATURE_COLUMNS)
        self._cols = list(FEATURE_COLUMNS)

    def _transform(self, feat: pd.DataFrame) -> np.ndarray:
        base = feat[self._base].to_numpy(dtype=np.float64)
        base = np.clip(self.scaler.transform(base), -8.0, 8.0)
        dev = [c for c in self._cols if c.startswith("dev_")]
        d = np.clip(feat[dev].to_numpy(dtype=np.float32), -8.0, 8.0)
        return np.concatenate([base.astype(np.float32), d], axis=1)

    def _contexts(self, feat, X):
        from .static_gnn import _build_user_day_context
        return _build_user_day_context(feat, X, self.max_ctx)

    def fit(self, feat_train: pd.DataFrame, y=None):
        from ..utils.seed import set_seed
        set_seed(self.seed)
        feat_train = feat_train.reset_index(drop=True)
        self.scaler = StandardScaler()
        self.scaler.fit(feat_train[self._base].to_numpy(dtype=np.float64))
        X = self._transform(feat_train)
        ctx, cur, y_arr = self._contexts(feat_train, X)
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float32)
        D = X.shape[1]
        self.in_proj = nn.Linear(D, self.d_model).to(self.device)
        layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, dim_feedforward=self.d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(layer, self.layers).to(self.device)
        self.head = nn.Linear(self.d_model, 1).to(self.device)
        params = (
            list(self.in_proj.parameters())
            + list(self.enc.parameters())
            + list(self.head.parameters())
        )
        n_pos = float(y_arr.sum())
        n_neg = float(len(y_arr) - n_pos)
        pw = torch.tensor([min(n_neg / max(n_pos, 1.0), 100.0)], device=self.device)
        opt = torch.optim.Adam(params, lr=self.lr)
        n = len(y_arr)
        for ep in range(self.epochs):
            perm = np.random.default_rng(self.seed + ep).permutation(n)
            self.in_proj.train(); self.enc.train(); self.head.train()
            for i in range(0, n, self.batch_size):
                sl = perm[i:i + self.batch_size]
                opt.zero_grad()
                # use context window; pool last token
                z = self.in_proj(torch.tensor(ctx[sl], device=self.device))
                z = self.enc(z)[:, -1, :]
                logits = self.head(z).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.tensor(y_arr[sl], device=self.device), pos_weight=pw,
                )
                loss.backward()
                opt.step()
        return self

    def anomaly_scores(self, feat: pd.DataFrame) -> np.ndarray:
        feat = feat.reset_index(drop=True)
        X = self._transform(feat)
        ctx, _, _ = self._contexts(feat, X)
        self.in_proj.eval(); self.enc.eval(); self.head.eval()
        out = np.zeros(len(X), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                sl = slice(i, i + self.batch_size)
                z = self.in_proj(torch.tensor(ctx[sl], device=self.device))
                z = self.enc(z)[:, -1, :]
                out[sl] = torch.sigmoid(self.head(z).squeeze(-1)).cpu().numpy()
        return out
