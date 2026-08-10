"""
Heterogeneous temporal graph construction from unified events.

Node types: 'user', 'host'. Edge type: (user)-[access @ t]->(host), carrying the
action id and a timestamp. This is the substrate for the temporal GNN backbone.

Two outputs:
  * summary_stats(): pure-python/pandas graph statistics -- runs anywhere, used
    by the smoke test and sanity checks.
  * to_pyg(): builds a torch_geometric HeteroData with a time-ordered edge stream
    (lazy torch import; only needed on the training box / Cursor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    TIMESTAMP, USER, SRC_HOST, DST_HOST, ACTION, LABEL, ACTION_TO_ID,
)


def build_index(df):
    users = sorted(df[USER].unique())
    hosts = sorted(pd.unique(df[[SRC_HOST, DST_HOST]].values.ravel("K")))
    uidx = {u: i for i, u in enumerate(users)}
    hidx = {h: i for i, h in enumerate(hosts)}
    return uidx, hidx


def edge_stream(df):
    """Return a time-ordered DataFrame of (t, user_idx, host_idx, action_id, label)."""
    uidx, hidx = build_index(df)
    d = df.sort_values(TIMESTAMP, kind="mergesort").copy()   # stable: aligns with scorer
    # Force nanosecond resolution before int cast: pandas 2.x may store datetimes
    # at second/minute resolution, which would silently break time-delta math.
    t_ns = d[TIMESTAMP].astype("datetime64[ns]").astype("int64").to_numpy()
    e = pd.DataFrame({
        "t": t_ns,  # ns since epoch
        "u": d[USER].map(uidx).to_numpy(),
        "h": d[DST_HOST].map(hidx).to_numpy(),
        "a": d[ACTION].map(ACTION_TO_ID).to_numpy(),
        "y": d[LABEL].to_numpy(),
    })
    return e, uidx, hidx


def summary_stats(df) -> dict:
    e, uidx, hidx = edge_stream(df)
    deg_u = e.groupby("u").size()
    deg_h = e.groupby("h").size()
    return {
        "n_users": len(uidx),
        "n_hosts": len(hidx),
        "n_edges": int(len(e)),
        "n_malicious_edges": int(e["y"].sum()),
        "mal_edge_rate": float(e["y"].mean()),
        "avg_user_degree": float(deg_u.mean()),
        "avg_host_degree": float(deg_h.mean()),
        "span_days": float((e["t"].max() - e["t"].min()) / 1e9 / 86400),
    }


def to_pyg(df, time_normalize=True):
    """Build a torch_geometric HeteroData. Requires torch + torch_geometric."""
    import torch                          # noqa: local heavy import
    from torch_geometric.data import HeteroData

    e, uidx, hidx = edge_stream(df)
    data = HeteroData()
    data["user"].num_nodes = len(uidx)
    data["host"].num_nodes = len(hidx)

    src = torch.tensor(e["u"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(e["h"].to_numpy(), dtype=torch.long)
    t = e["t"].to_numpy().astype(np.float64)
    if time_normalize and t.max() > t.min():
        t = (t - t.min()) / (t.max() - t.min())
    edge_time = torch.tensor(t, dtype=torch.float)
    edge_attr = torch.tensor(e["a"].to_numpy(), dtype=torch.long)   # action id
    edge_label = torch.tensor(e["y"].to_numpy(), dtype=torch.long)

    rel = ("user", "access", "host")
    data[rel].edge_index = torch.stack([src, dst], dim=0)
    data[rel].edge_time = edge_time
    data[rel].edge_attr = edge_attr
    data[rel].edge_label = edge_label
    # reverse relation for message passing both ways
    data["host", "accessed_by", "user"].edge_index = torch.stack([dst, src], dim=0)
    data["host", "accessed_by", "user"].edge_time = edge_time
    return data
