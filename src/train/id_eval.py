"""
Honest in-distribution evaluation under temporal AND user-disjoint splits.

Why both: a temporal-only split still lets the same user's habits leak into the
test set; a user-disjoint-only split still lets future activity of train users
leak backward. Reporting both closes those two common optimism paths.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import pandas as pd

from ..data.features import user_day_features, X_y
from .evaluate import compute_metrics
from .splits import temporal_split, user_disjoint_split

log = logging.getLogger(__name__)


def _eval_feature_model(model_factory, train_df, test_df) -> dict:
    """fit(X)/anomaly_scores(X) path used by IsolationForest / OCSVM."""
    Xtr, _ = X_y(user_day_features(train_df))
    Xte, yte = X_y(user_day_features(test_df))
    model = model_factory()
    model.fit(Xtr)
    scores = model.anomaly_scores(Xte)
    return compute_metrics(yte, scores)


def _eval_event_model(fit_fn, score_fn, train_df, test_df) -> dict:
    """Event-stream path: fit(train_df), score_user_day(test_df) → metrics."""
    fit_fn(train_df)
    agg = score_fn(test_df)
    return compute_metrics(agg["label"].to_numpy(), agg["score"].to_numpy())


def in_distribution_eval(
    df: pd.DataFrame,
    *,
    model_factory: Optional[Callable] = None,
    fit_score: Optional[tuple] = None,
    train_frac: float = 0.7,
    seed: int = 0,
    metric: str = "pr_auc",
) -> Dict[str, dict]:
    """Run temporal + user-disjoint ID protocols on one dataset.

    Provide exactly one of:
      * model_factory — callable → object with fit(X) / anomaly_scores(X)
      * fit_score — (fit_fn, score_fn) where fit_fn(df), score_fn(df)->agg frame
                    with columns score, label (GNN path)

    Returns ``{"temporal": metrics, "user_disjoint": metrics}``.
    """
    if (model_factory is None) == (fit_score is None):
        raise ValueError("Provide exactly one of model_factory or fit_score")

    out: Dict[str, dict] = {}
    for name, splitter in (
        ("temporal", lambda d: temporal_split(d, train_frac=train_frac)),
        ("user_disjoint", lambda d: user_disjoint_split(d, train_frac=train_frac, seed=seed)),
    ):
        train_df, test_df = splitter(df)
        if len(train_df) == 0 or len(test_df) == 0:
            log.warning("id_eval/%s: empty split — skipping", name)
            out[name] = {metric: float("nan"), "n": 0, "n_pos": 0}
            continue
        if model_factory is not None:
            m = _eval_feature_model(model_factory, train_df, test_df)
        else:
            fit_fn, score_fn = fit_score
            m = _eval_event_model(fit_fn, score_fn, train_df, test_df)
        m["split"] = name
        out[name] = m
        log.info(
            "id_eval/%s: %s=%.3f n=%d pos=%d",
            name,
            metric,
            m.get(metric, float("nan")),
            m.get("n", 0),
            m.get("n_pos", 0),
        )
    return out


def id_splits(dfs: dict, **kw) -> pd.DataFrame:
    """Paper-facing alias: temporal + user-disjoint ID eval table for ``dfs``.

    Pass ``model_factory=...`` for baselines or ``detector_factory=...`` for a
    TemporalGNN-style fit(df)/score_user_day(df) detector. For the usual GNN
    path prefer ``src.train.train_gnn.gnn_id_eval``.
    """
    return id_eval_table(dfs, **kw)


def id_eval_table(
    dfs: dict,
    *,
    model_factory: Optional[Callable] = None,
    detector_factory: Optional[Callable] = None,
    train_frac: float = 0.7,
    seed: int = 0,
    metric: str = "pr_auc",
) -> pd.DataFrame:
    """Per-dataset ID metrics under both splits → tidy DataFrame.

    detector_factory: callable → object with fit(df)/score_user_day(df)
                      (TemporalGNNDetector). Prefer model_factory for baselines.
    Fresh detector per split so training state never leaks across protocols.
    """
    rows = []
    for name, df in dfs.items():
        if detector_factory is not None:
            # Bind fit/score through a thin adapter that rebuilds the detector
            # on every fit() call (one fresh model per split).
            class _Fresh:
                def __init__(self):
                    self._det = None

                def fit(self, train_df, df_tgt=None):
                    self._det = detector_factory()
                    return self._det.fit(train_df, df_tgt)

                def score_user_day(self, test_df):
                    return self._det.score_user_day(test_df)

            adapter = _Fresh()
            res = in_distribution_eval(
                df,
                fit_score=(adapter.fit, adapter.score_user_day),
                train_frac=train_frac,
                seed=seed,
                metric=metric,
            )
        else:
            res = in_distribution_eval(
                df,
                model_factory=model_factory,
                train_frac=train_frac,
                seed=seed,
                metric=metric,
            )
        for split, m in res.items():
            rows.append({"dataset": name, "split": split, **m})
    return pd.DataFrame(rows)
