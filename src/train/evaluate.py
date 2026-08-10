"""
Evaluation metrics tuned for extreme class imbalance (insider positives << 1%).

AUC-PR (average precision) is the headline metric, not AUC-ROC, because ROC is
optimistic under heavy imbalance. We also report detection rate at a fixed low
false-positive rate (what an analyst actually cares about) and precision@k.
Bootstrap confidence intervals let us make honest cross-dataset comparisons.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def detection_rate_at_fpr(y_true, scores, target_fpr=0.01):
    """Recall achievable while holding FPR <= target_fpr. Higher is better."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    neg = scores[y_true == 0]
    if neg.size == 0 or (y_true == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - target_fpr)   # threshold giving ~target_fpr
    pred = scores >= thr
    tp = np.sum(pred & (y_true == 1))
    return float(tp / np.sum(y_true == 1))


def precision_at_k(y_true, scores, k=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if k is None:
        k = int(np.sum(y_true == 1))          # k = number of true positives
    k = max(1, min(k, len(scores)))
    idx = np.argsort(-scores)[:k]
    return float(np.mean(y_true[idx] == 1))


def compute_metrics(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    out = {}
    # roc/pr undefined if only one class present
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["dr_at_1pct_fpr"] = detection_rate_at_fpr(y_true, scores, 0.01)
    out["dr_at_5pct_fpr"] = detection_rate_at_fpr(y_true, scores, 0.05)
    out["precision_at_k"] = precision_at_k(y_true, scores)
    out["n"] = int(len(y_true))
    out["n_pos"] = int(np.sum(y_true == 1))
    return out


def bootstrap_ci(y_true, scores, metric="pr_auc", n_boot=1000, alpha=0.05, seed=0):
    """Stratified bootstrap CI for a chosen metric. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    def _one(idx):
        return compute_metrics(y_true[idx], scores[idx]).get(metric, float("nan"))

    point = _one(np.arange(len(y_true)))
    stats = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True) if len(pos) else pos
        bn = rng.choice(neg, size=len(neg), replace=True) if len(neg) else neg
        idx = np.concatenate([bp, bn])
        val = _one(idx)
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi
