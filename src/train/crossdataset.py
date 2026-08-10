"""
The cross-dataset generalization protocol -- the core contribution's harness.

Given a set of datasets and a model factory, train on each source and evaluate on
every target, producing the in-distribution (diagonal) vs out-of-distribution
(off-diagonal) matrix. The paper's headline claim is that off-diagonal collapses
for models that ace the diagonal.

Also provides leave-one-domain-out (train on K-1, test on held-out) for the
feature-based baselines that expose fit(X)/anomaly_scores(X).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.features import user_day_features, X_y
from .domain_concat import tag_and_concat
from .evaluate import compute_metrics

log = logging.getLogger(__name__)


def _prep(df):
    feat = user_day_features(df)
    return X_y(feat)


def cross_dataset_matrix(dfs: dict, model_factory, metric="pr_auc", seed=0):
    """
    dfs: {name: unified_dataframe}
    model_factory: callable -> model with fit(X)/anomaly_scores(X)
    Returns a (source x target) DataFrame of the chosen metric plus the full
    per-cell metric dicts.
    """
    names = list(dfs.keys())
    prepared = {n: _prep(dfs[n]) for n in names}

    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    details = {}
    for s in names:
        Xs, ys = prepared[s]
        # unsupervised: fit on benign-majority source features
        model = model_factory()
        model.fit(Xs)
        for t in names:
            Xt, yt = prepared[t]
            scores = model.anomaly_scores(Xt)
            m = compute_metrics(yt, scores)
            details[(s, t)] = m
            matrix.loc[s, t] = m.get(metric, float("nan"))
    return matrix, details


def leave_one_domain_out(dfs: dict, model_factory, metric="pr_auc"):
    """Train on the concatenation of K-1 domains; evaluate on the held-out one.

    Returns a Series indexed by held-out domain name, plus per-fold metric dicts.
    """
    names = list(dfs.keys())
    if len(names) < 2:
        raise ValueError("leave_one_domain_out needs at least 2 domains")
    scores = {}
    details = {}
    for held in names:
        train_dfs = {n: dfs[n] for n in names if n != held}
        train_df = tag_and_concat(train_dfs)
        # Tag the held-out set too so user/host namespaces stay consistent if a
        # model ever mixed them; features themselves are namespace-agnostic.
        test_df = tag_and_concat({held: dfs[held]})
        log.info(
            "LODO fold held_out=%s train_domains=%s train_rows=%d test_rows=%d",
            held,
            list(train_dfs.keys()),
            len(train_df),
            len(test_df),
        )
        Xtr, _ = _prep(train_df)
        Xte, yte = _prep(test_df)
        model = model_factory()
        model.fit(Xtr)
        m = compute_metrics(yte, model.anomaly_scores(Xte))
        details[held] = m
        scores[held] = m.get(metric, float("nan"))
    return pd.Series(scores, name=metric), details


def generalization_gap(matrix: pd.DataFrame) -> float:
    """Mean(diagonal) - Mean(off-diagonal): how much performance is lost OOD."""
    vals = matrix.to_numpy(dtype=float)
    diag = np.nanmean(np.diag(vals))
    off = vals.copy()
    np.fill_diagonal(off, np.nan)
    return float(diag - np.nanmean(off))
