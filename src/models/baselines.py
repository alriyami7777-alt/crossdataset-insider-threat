"""
Non-deep baselines (unsupervised anomaly detectors) that run with sklearn only.

These exist so the cross-dataset harness is runnable today, and because they are
standard ITD baselines the paper must beat. All expose the same interface:
    fit(X_train)                 # train on (mostly benign) source features
    anomaly_scores(X)  -> np.ndarray  # higher = more anomalous
so the cross-dataset runner can treat every model identically.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class _ScaledDetector:
    def __init__(self, core):
        self.core = core
        self.scaler = StandardScaler()

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.core.fit(Xs)
        return self

    def anomaly_scores(self, X):
        Xs = self.scaler.transform(X)
        # sklearn: higher score_samples = more normal -> negate for anomaly
        return -self.core.score_samples(Xs)


def isolation_forest(seed=0, **kw):
    return _ScaledDetector(IsolationForest(
        n_estimators=kw.get("n_estimators", 200),
        contamination=kw.get("contamination", "auto"),
        random_state=seed,
    ))


def ocsvm(**kw):
    return _ScaledDetector(OneClassSVM(
        kernel=kw.get("kernel", "rbf"),
        nu=kw.get("nu", 0.05),
        gamma=kw.get("gamma", "scale"),
    ))


MODEL_FACTORY = {
    "isolation_forest": isolation_forest,
    "ocsvm": ocsvm,
}
