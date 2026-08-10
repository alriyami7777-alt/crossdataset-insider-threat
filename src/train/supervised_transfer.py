"""
Honest supervised (and unsupervised) cross-dataset transfer matrices.

FIX 1 — diagonal never trains and scores the same rows:
  * Off-diagonal: fit on FULL source, score FULL target.
  * Diagonal: held-out split of the source (temporal OR user-disjoint).

FIX 2 — base-rate comparability:
  Every cell reports target base rate p = n_pos/n and lift = PR-AUC / p.
  Generalization gap is computed on both raw PR-AUC and lift.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.features import user_day_features, X_y
from ..data.schema import USER, LABEL
from ..models.baselines import isolation_forest, ocsvm
from .evaluate import compute_metrics
from .splits import temporal_split, user_disjoint_split

log = logging.getLogger(__name__)


def _base_rate(y) -> float:
    y = np.asarray(y)
    return float(y.mean()) if len(y) else float("nan")


def _with_lift(m: dict, y_true) -> dict:
    p = _base_rate(y_true)
    m = dict(m)
    m["base_rate"] = p
    pr = m.get("pr_auc", float("nan"))
    m["lift"] = float(pr / p) if (p and p > 0 and np.isfinite(pr)) else float("nan")
    return m


def generalization_gap(matrix: pd.DataFrame) -> float:
    vals = matrix.to_numpy(dtype=float)
    diag = np.nanmean(np.diag(vals))
    off = vals.copy()
    np.fill_diagonal(off, np.nan)
    return float(diag - np.nanmean(off))


def _split_user_day_frame(feat: pd.DataFrame, protocol: str, train_frac=0.7, seed=0):
    """Held-out split on an already-built user-day feature frame."""
    if protocol == "temporal":
        f = feat.sort_values("day", kind="mergesort")
        cut = f["day"].quantile(train_frac)
        train, test = f[f["day"] <= cut], f[f["day"] > cut]
        if len(train) == 0 or len(test) == 0:
            n = max(1, int(len(f) * train_frac))
            train, test = f.iloc[:n], f.iloc[n:]
        return train, test
    if protocol == "user_disjoint":
        rng = np.random.default_rng(seed)
        user_pos = feat.groupby(USER)[LABEL].max()
        pos_u = user_pos[user_pos == 1].index.to_numpy()
        neg_u = user_pos[user_pos == 0].index.to_numpy()
        rng.shuffle(pos_u)
        rng.shuffle(neg_u)

        def _take(users):
            n_tr = int(round(len(users) * train_frac))
            if len(users) >= 2:
                n_tr = min(max(n_tr, 1), len(users) - 1)
            return set(users[:n_tr])

        train_users = _take(pos_u) | _take(neg_u)
        test_users = set(user_pos.index) - train_users
        return feat[feat[USER].isin(train_users)], feat[feat[USER].isin(test_users)]
    raise ValueError(f"unknown diagonal protocol: {protocol}")


# ---- model adapters ---------------------------------------------------------
class SupervisedAdapter:
    """fit(X,y) + predict_proba -> anomaly_scores interface."""

    def __init__(self, factory: Callable):
        self.factory = factory
        self.model = None

    def fit(self, X, y=None):
        self.model = self.factory()
        if y is None:
            raise ValueError("SupervisedAdapter requires y")
        self.model.fit(X, y)
        return self

    def anomaly_scores(self, X):
        return self.model.predict_proba(X)[:, 1]


class UnsupervisedAdapter:
    def __init__(self, factory: Callable):
        self.factory = factory
        self.model = None

    def fit(self, X, y=None):
        self.model = self.factory()
        self.model.fit(X)  # ignores y
        return self

    def anomaly_scores(self, X):
        return self.model.anomaly_scores(X)


def rf_factory(seed=0):
    return lambda: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def lr_factory(seed=0):
    return lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="lbfgs",
        )),
    ])


FEATURE_MODEL_REGISTRY = {
    "random_forest": ("supervised", rf_factory),
    "logistic_regression": ("supervised", lr_factory),
    "isolation_forest": ("unsupervised", lambda seed=0: lambda: isolation_forest(seed=seed)),
    "ocsvm": ("unsupervised", lambda seed=0: ocsvm),
}


def _make_adapter(name: str, seed=0):
    kind, factory_maker = FEATURE_MODEL_REGISTRY[name]
    factory = factory_maker(seed) if kind == "supervised" else factory_maker(seed)
    if kind == "supervised":
        return SupervisedAdapter(factory)
    return UnsupervisedAdapter(factory)


def cross_dataset_transfer(
    dfs: dict,
    model_name: str = "random_forest",
    *,
    deviation: bool = True,
    diagonal_protocol: str = "temporal",
    train_frac: float = 0.7,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Build honest source×target PR-AUC and lift matrices for one model.

    Returns (pr_auc_matrix, lift_matrix, details, target_stats).
    """
    names = list(dfs.keys())
    # Features on FULL streams so temporal test days keep causal history.
    feats = {n: user_day_features(dfs[n], deviation=deviation) for n in names}
    target_stats = {}
    for n in names:
        y = feats[n][LABEL].to_numpy()
        target_stats[n] = {
            "base_rate": _base_rate(y),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
        }
        log.info(
            "target %s: p=%.6f n=%d n_pos=%d",
            n, target_stats[n]["base_rate"], target_stats[n]["n"], target_stats[n]["n_pos"],
        )

    pr_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    lift_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    details = {}

    for s in names:
        for t in names:
            adapter = _make_adapter(model_name, seed=seed)
            if s == t:
                train_f, test_f = _split_user_day_frame(
                    feats[s], diagonal_protocol, train_frac=train_frac, seed=seed
                )
                log.info(
                    "diagonal %s [%s]: train=%d (pos=%.4f) test=%d (pos=%.4f)",
                    s, diagonal_protocol, len(train_f),
                    float(train_f[LABEL].mean()) if len(train_f) else float("nan"),
                    len(test_f),
                    float(test_f[LABEL].mean()) if len(test_f) else float("nan"),
                )
                Xtr, ytr = X_y(train_f)
                Xte, yte = X_y(test_f)
            else:
                Xtr, ytr = X_y(feats[s])
                Xte, yte = X_y(feats[t])
                log.info(
                    "off-diag %s->%s: fit full source n=%d, score full target n=%d",
                    s, t, len(ytr), len(yte),
                )

            adapter.fit(Xtr, ytr)
            scores = adapter.anomaly_scores(Xte)
            m = _with_lift(compute_metrics(yte, scores), yte)
            m["source"] = s
            m["target"] = t
            m["model"] = model_name
            m["diagonal_protocol"] = diagonal_protocol if s == t else "full_source"
            details[(s, t)] = m
            pr_mat.loc[s, t] = m["pr_auc"]
            lift_mat.loc[s, t] = m["lift"]

    return pr_mat, lift_mat, details, target_stats


def print_transfer_report(
    model_name: str,
    pr_mat: pd.DataFrame,
    lift_mat: pd.DataFrame,
    details: dict,
    target_stats: dict,
    diagonal_protocol: str,
):
    print(f"\n======== {model_name} | diagonal={diagonal_protocol} ========")
    print("Target base rates:")
    for n, st in target_stats.items():
        print(f"  {n}: p={st['base_rate']:.6f}  n={st['n']}  n_pos={st['n_pos']}")
    print("\nPR-AUC:")
    print(pr_mat.round(3).to_string())
    print(f"gap_pr_auc = {generalization_gap(pr_mat):.3f}")
    print("\nLift (= PR-AUC / target_base_rate):")
    print(lift_mat.round(3).to_string())
    print(f"gap_lift = {generalization_gap(lift_mat):.3f}")


def run_all_feature_models(
    dfs: dict,
    *,
    deviation: bool = True,
    seed: int = 0,
    models: Optional[list] = None,
) -> dict:
    """Run every feature model under BOTH diagonal protocols.

    Returns nested dict: model -> protocol -> {pr, lift, details, target_stats}.
    """
    models = models or list(FEATURE_MODEL_REGISTRY.keys())
    out = {}
    for model_name in models:
        out[model_name] = {}
        for protocol in ("temporal", "user_disjoint"):
            pr, lift, details, tstats = cross_dataset_transfer(
                dfs, model_name,
                deviation=deviation,
                diagonal_protocol=protocol,
                seed=seed,
            )
            print_transfer_report(model_name, pr, lift, details, tstats, protocol)
            out[model_name][protocol] = {
                "pr_auc": pr, "lift": lift, "details": details, "target_stats": tstats,
            }
    return out


def run_gnn_transfer(
    dfs: dict,
    *,
    diagonal_protocol: str = "temporal",
    train_frac: float = 0.7,
    seed: int = 0,
    **gnn_kw,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """GNN matrix with the same honest-diagonal + lift protocol.

    Diagonal: temporal_split / user_disjoint_split on raw events, then fit/score.
    Off-diagonal: fit full source events, score full target.
    Uses deviation-augmented user-day readout when enabled in gnn_kw.
    """
    from .train_gnn import TemporalGNNDetector

    names = list(dfs.keys())
    # Target stats from user-day labels (same units as feature models)
    target_stats = {}
    for n in names:
        feat = user_day_features(dfs[n], deviation=True)
        y = feat[LABEL].to_numpy()
        target_stats[n] = {
            "base_rate": _base_rate(y), "n": int(len(y)), "n_pos": int(y.sum()),
        }

    pr_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    lift_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    details = {}
    gnn_kw = dict(use_ud_features=True, deviation=True, pos_weight=True, seed=seed, **gnn_kw)

    for s in names:
        for t in names:
            det = TemporalGNNDetector(**gnn_kw)
            if s == t:
                if diagonal_protocol == "temporal":
                    train_df, test_df = temporal_split(dfs[s], train_frac=train_frac)
                else:
                    train_df, test_df = user_disjoint_split(
                        dfs[s], train_frac=train_frac, seed=seed
                    )
                log.info(
                    "GNN diagonal %s [%s]: train_edges=%d test_edges=%d",
                    s, diagonal_protocol, len(train_df), len(test_df),
                )
                det.fit(train_df)
                agg = det.score_user_day(test_df)
            else:
                log.info("GNN off-diag %s->%s: full source / full target", s, t)
                det.fit(dfs[s])
                agg = det.score_user_day(dfs[t])
            y = agg["label"].to_numpy()
            m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
            m.update({
                "source": s, "target": t, "model": "temporal_gnn",
                "diagonal_protocol": diagonal_protocol if s == t else "full_source",
            })
            details[(s, t)] = m
            pr_mat.loc[s, t] = m["pr_auc"]
            lift_mat.loc[s, t] = m["lift"]
            log.info(
                "GNN %s->%s [%s]: pr_auc=%.3f lift=%.2f p=%.5f",
                s, t, m["diagonal_protocol"], m["pr_auc"], m["lift"], m["base_rate"],
            )

    return pr_mat, lift_mat, details, target_stats
