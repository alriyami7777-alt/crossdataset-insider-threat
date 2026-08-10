"""
User-day feature aggregation -- the shared tabular view used by the non-graph
baselines (Isolation Forest, One-Class SVM, LSTM-AE inputs) and optionally the
GNN user-day readout.

Produces one row per (user, day) with action counts, temporal stats, and host
breadth. A user-day is labelled malicious (1) if it contains any malicious event.
Feature columns are FIXED by the canonical action vocabulary so vectors from
different datasets are aligned dimension-for-dimension -- required for cross-
dataset transfer.

Optional *per-user deviation* features (``deviation=True``) express each day's
counts relative to that user's own causal baseline (expanding mean/std over
strictly earlier days). Same transform for every dataset; no content/keywords.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .schema import ACTIONS, TIMESTAMP, USER, ACTION, DST_HOST, LABEL

log = logging.getLogger(__name__)

BASE_FEATURE_COLUMNS = (
    [f"cnt_{a}" for a in ACTIONS]
    + ["n_events", "n_distinct_hosts", "frac_offhours", "hour_mean", "hour_std"]
)
DEV_FEATURE_COLUMNS = [f"dev_{c}" for c in BASE_FEATURE_COLUMNS]
# Full aligned vocabulary when deviation features are enabled (base then dev).
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DEV_FEATURE_COLUMNS


def _add_causal_deviations(out: pd.DataFrame) -> pd.DataFrame:
    """Append ``dev_<base>`` columns: (x - expanding_mean) / (expanding_std + 1e-6)
    over each user's STRICTLY EARLIER days (shift-1). Cold start -> 0.0."""
    out = out.sort_values([USER, "day"], kind="mergesort").reset_index(drop=True)
    n_cold = 0
    n_nan_filled = 0
    for col in BASE_FEATURE_COLUMNS:
        # expanding stats including current day, then shift -> stats over earlier days only
        mean_prev = (
            out.groupby(USER, sort=False)[col]
            .transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
        )
        std_prev = (
            out.groupby(USER, sort=False)[col]
            .transform(lambda s: s.expanding(min_periods=1).std(ddof=0).shift(1))
        )
        dev = (out[col] - mean_prev) / (std_prev + 1e-6)
        cold = mean_prev.isna()
        n_cold += int(cold.sum())
        n_nan_filled += int(dev.isna().sum())
        out[f"dev_{col}"] = dev.fillna(0.0)
    # cold-start count is summed over features; report per-row first-day count once
    n_first_days = int(
        out.groupby(USER, sort=False).cumcount().eq(0).sum()
    )
    log.info(
        "deviation features: users' first days (cold start)=%d | "
        "dev cells filled from NaN=%d (incl. cold start x n_features)",
        n_first_days,
        n_nan_filled,
    )
    return out


def user_day_features(df: pd.DataFrame, deviation: bool = False) -> pd.DataFrame:
    """Build per-(user, day) features. If ``deviation``, append causal ``dev_*``."""
    d = df.copy()
    d["day"] = d[TIMESTAMP].dt.floor("D")
    d["hour"] = d[TIMESTAMP].dt.hour
    d["offhours"] = ((d["hour"] < 7) | (d["hour"] > 19)).astype(int)

    # action counts pivot
    counts = (
        d.pivot_table(index=[USER, "day"], columns=ACTION, values=TIMESTAMP,
                      aggfunc="count", fill_value=0)
        .reindex(columns=ACTIONS, fill_value=0)
    )
    counts.columns = [f"cnt_{a}" for a in ACTIONS]

    agg = d.groupby([USER, "day"]).agg(
        n_events=(TIMESTAMP, "count"),
        n_distinct_hosts=(DST_HOST, "nunique"),
        frac_offhours=("offhours", "mean"),
        hour_mean=("hour", "mean"),
        hour_std=("hour", "std"),
        label=(LABEL, "max"),
    )
    out = counts.join(agg)
    out["hour_std"] = out["hour_std"].fillna(0.0)
    out = out.reset_index()

    for c in BASE_FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0

    if deviation:
        out = _add_causal_deviations(out)
        cols = FEATURE_COLUMNS
    else:
        cols = BASE_FEATURE_COLUMNS

    # guarantee fixed feature order for the active column set
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def active_feature_columns(feat: pd.DataFrame):
    """Columns to feed a model: full FEATURE_COLUMNS iff any ``dev_`` is present."""
    if any(c.startswith("dev_") for c in feat.columns):
        return list(FEATURE_COLUMNS)
    return list(BASE_FEATURE_COLUMNS)


def X_y(feat: pd.DataFrame, columns=None):
    cols = columns if columns is not None else active_feature_columns(feat)
    X = feat[cols].to_numpy(dtype=float)
    y = feat[LABEL].to_numpy(dtype=int)
    return X, y
