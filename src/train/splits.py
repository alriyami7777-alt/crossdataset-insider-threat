"""
In-distribution evaluation splits.

Cross-dataset transfer is only meaningful if the *in*-distribution numbers are
honest. Two complementary protocols (both required by the paper design):

  * temporal_split  — train on the past, test on the future (no future leakage).
  * user_disjoint_split — train users and test users are disjoint (no identity
    leakage across the split).

Both operate on canonical event dataframes and log the resulting subset sizes.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from ..data.schema import TIMESTAMP, USER, LABEL

log = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split events by time: earliest ``train_frac`` → train, remainder → test.

    Cut is at a timestamp quantile so the split is chronological, not shuffled.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    d = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    cut = d[TIMESTAMP].quantile(train_frac)
    train = d[d[TIMESTAMP] <= cut].reset_index(drop=True)
    test = d[d[TIMESTAMP] > cut].reset_index(drop=True)
    # Edge case: all timestamps equal → fall back to row-index cut
    if len(test) == 0 or len(train) == 0:
        n = max(1, int(len(d) * train_frac))
        train, test = d.iloc[:n].copy(), d.iloc[n:].copy()
        log.warning(
            "temporal_split: degenerate timestamps; fell back to row cut "
            "n_train=%d n_test=%d",
            len(train),
            len(test),
        )
    log.info(
        "temporal_split: train_frac=%.2f cut=%s | train=%d (pos=%.3f) "
        "test=%d (pos=%.3f)",
        train_frac,
        cut,
        len(train),
        float(train[LABEL].mean()) if len(train) else float("nan"),
        len(test),
        float(test[LABEL].mean()) if len(test) else float("nan"),
    )
    return train, test


def user_disjoint_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition users into disjoint train/test sets; keep all their events.

    Stratifies lightly on whether a user has any positive label so both sides
    usually retain positives when the dataset has enough malicious users.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    rng = np.random.default_rng(seed)
    user_pos = df.groupby(USER)[LABEL].max()
    pos_users = user_pos[user_pos == 1].index.to_numpy()
    neg_users = user_pos[user_pos == 0].index.to_numpy()
    rng.shuffle(pos_users)
    rng.shuffle(neg_users)

    def _take(users: np.ndarray) -> set:
        n_tr = int(round(len(users) * train_frac))
        # Keep at least one user on each side when possible
        if len(users) >= 2:
            n_tr = min(max(n_tr, 1), len(users) - 1)
        return set(users[:n_tr])

    train_users = _take(pos_users) | _take(neg_users)
    all_users = set(user_pos.index)
    test_users = all_users - train_users
    if not test_users:
        # Single-user edge case: put that user in test, empty train flagged upstream
        test_users = set(rng.choice(list(all_users), size=1))
        train_users = all_users - test_users

    train = df[df[USER].isin(train_users)].reset_index(drop=True)
    test = df[df[USER].isin(test_users)].reset_index(drop=True)
    log.info(
        "user_disjoint_split: train_frac=%.2f seed=%d | "
        "train_users=%d/%d events=%d (pos=%.3f) | "
        "test_users=%d/%d events=%d (pos=%.3f)",
        train_frac,
        seed,
        len(train_users),
        len(all_users),
        len(train),
        float(train[LABEL].mean()) if len(train) else float("nan"),
        len(test_users),
        len(all_users),
        len(test),
        float(test[LABEL].mean()) if len(test) else float("nan"),
    )
    return train, test
