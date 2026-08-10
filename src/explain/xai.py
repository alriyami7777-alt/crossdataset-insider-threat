"""
Explainability layer, reframed as a TRANSFER DIAGNOSTIC (Contribution 2's twist).

Rather than only explaining individual alerts (as prior work does), we use feature
attribution to answer: which signals are TRANSFERABLE (drive detection on both
source and target) vs DATASET-SPECIFIC ARTIFACTS (drive detection only in-domain)?
The gap between the two attribution rankings is itself evidence for the overfitting
claim.

`transfer_diagnostic` works on any model exposing anomaly_scores(X) over the shared
user-day feature space, so it runs today with the sklearn baselines; GNNExplainer /
attention export for the deep models is stubbed for Cursor.
"""
from __future__ import annotations

import numpy as np

from ..data.features import FEATURE_COLUMNS


def permutation_importance(model, X, y, metric_fn, n_repeats=5, seed=0):
    """Model-agnostic permutation importance over the shared feature space."""
    rng = np.random.default_rng(seed)
    base = metric_fn(y, model.anomaly_scores(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - metric_fn(y, model.anomaly_scores(Xp)))
        imp[j] = float(np.mean(drops))
    return imp


def transfer_diagnostic(model, X_src, y_src, X_tgt, y_tgt, metric_fn, top=10):
    """Compare feature importance in-domain vs out-of-domain.
    Returns ranked features and a 'transferability' score per feature =
    min(importance_src, importance_tgt) normalised. Low transferability + high
    src importance == a CERT-style artifact the model leaned on."""
    imp_s = permutation_importance(model, X_src, y_src, metric_fn)
    imp_t = permutation_importance(model, X_tgt, y_tgt, metric_fn)
    names = FEATURE_COLUMNS
    def norm(v):
        v = np.clip(v, 0, None)
        return v / (v.max() + 1e-9)
    ns, nt = norm(imp_s), norm(imp_t)
    transfer = np.minimum(ns, nt)
    artifact = ns - nt          # high => source-only signal (suspected artifact)
    order = np.argsort(-ns)
    rows = [{
        "feature": names[i], "imp_src": float(ns[i]), "imp_tgt": float(nt[i]),
        "transferability": float(transfer[i]), "artifact_score": float(artifact[i]),
    } for i in order[:top]]
    return rows


# ---- deep-model explainers (implement in Cursor) ---------------------------
def gnn_explain(model, data, edge_id):
    """GNNExplainer subgraph rationale for a flagged access edge. TODO(Cursor)."""
    raise NotImplementedError("Wire torch_geometric.explain.GNNExplainer.")
