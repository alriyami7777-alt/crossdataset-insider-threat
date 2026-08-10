"""
Concatenate multiple canonical datasets while avoiding false identity matches.

User/host string IDs collide across datasets (e.g. "U01" on CERT vs SPEDIA).
Prefixing with the dataset name keeps the graph/feature code honest when we
train on K-1 domains for leave-one-domain-out.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..data.schema import USER, SRC_HOST, DST_HOST, DATASET

log = logging.getLogger(__name__)


def tag_and_concat(dfs: dict) -> pd.DataFrame:
    """Prefix user/host ids with dataset name, then concatenate."""
    if not dfs:
        raise ValueError("tag_and_concat requires at least one dataframe")
    parts = []
    for name, df in dfs.items():
        d = df.copy()
        d[USER] = name + "::" + d[USER].astype(str)
        d[SRC_HOST] = name + "::" + d[SRC_HOST].astype(str)
        d[DST_HOST] = name + "::" + d[DST_HOST].astype(str)
        d[DATASET] = name
        parts.append(d)
        log.info(
            "tag_and_concat: including domain=%s rows=%d", name, len(d)
        )
    out = pd.concat(parts, ignore_index=True)
    log.info(
        "tag_and_concat: total rows=%d domains=%s",
        len(out),
        list(dfs.keys()),
    )
    return out
