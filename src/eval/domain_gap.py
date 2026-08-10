"""
Domain-gap diagnostics on shared user-day features (CPU / sklearn only).

Computes proxy A-distance (Ben-David et al.) via domain classifiers and
linear / RBF MMD between two feature matrices.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Cap for RBF kernel (O(n^2)); domain classifiers may use a larger balanced set.
DEFAULT_MMD_CAP = 4000
# Cap for domain classifiers when both domains are huge (CERT↔CERT).
DEFAULT_DOMAIN_CAP = 20_000


def _subsample_pair(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seed: int,
    cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Balance to min(|A|,|B|), optionally further cap both sides equally."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(X_a), len(X_b)
    n_bal = min(n_a, n_b)
    info = {
        "n_a_full": int(n_a),
        "n_b_full": int(n_b),
        "n_balanced": int(n_bal),
        "capped": False,
        "n_used": int(n_bal),
    }
    if n_a > n_bal:
        idx = rng.choice(n_a, size=n_bal, replace=False)
        X_a = X_a[idx]
        log.info("subsample A: %d -> %d (seed=%d)", n_a, n_bal, seed)
    if n_b > n_bal:
        idx = rng.choice(n_b, size=n_bal, replace=False)
        X_b = X_b[idx]
        log.info("subsample B: %d -> %d (seed=%d)", n_b, n_bal, seed)

    if cap is not None and n_bal > cap:
        idx = rng.choice(n_bal, size=cap, replace=False)
        X_a = X_a[idx]
        X_b = X_b[idx]
        info["capped"] = True
        info["n_used"] = int(cap)
        log.info(
            "further cap both sides: %d -> %d (seed=%d)", n_bal, cap, seed
        )
    return X_a, X_b, info


def proxy_a_distance_and_auc(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> dict:
    """Domain classifiers on standardized features; return AUC + d̂_A.

    ε = mean CV error = 1 − accuracy.
    d̂_A = 2 (1 − 2ε)  (Ben-David proxy A-distance; clipped to [0, 2]).
    """
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a), dtype=int), np.ones(len(X_b), dtype=int)])
    # StratifiedKFold needs ≥ n_splits per class
    n_splits = int(min(n_splits, y.sum(), (y == 0).sum()))
    if n_splits < 2:
        return {
            "auc_lr": float("nan"),
            "auc_rf": float("nan"),
            "err_lr": float("nan"),
            "err_rf": float("nan"),
            "da_lr": float("nan"),
            "da_rf": float("nan"),
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    rf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=seed,
        ),
    )

    out = {}
    for name, clf in (("lr", lr), ("rf", rf)):
        proba = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        pred = (proba >= 0.5).astype(int)
        err = float(1.0 - (pred == y).mean())
        try:
            auc = float(roc_auc_score(y, proba))
        except ValueError:
            auc = float("nan")
        da = float(np.clip(2.0 * (1.0 - 2.0 * err), 0.0, 2.0))
        out[f"auc_{name}"] = auc
        out[f"err_{name}"] = err
        out[f"da_{name}"] = da
        log.info(
            "domain clf %s seed=%d: AUC=%.4f err=%.4f dA=%.4f n=%d",
            name, seed, auc, err, da, len(y),
        )
    return out


def linear_mmd(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Squared linear MMD = || mean(A) − mean(B) ||^2 on already-scaled features."""
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    d = mu_a - mu_b
    return float(d @ d)


def _median_heuristic_gamma(X: np.ndarray, rng: np.random.Generator, max_pairs: int = 5000) -> float:
    """gamma = 1 / (2 * sigma^2) with sigma = median pairwise Euclidean distance."""
    n = len(X)
    if n < 2:
        return 1.0
    # subsample rows for bandwidth estimate
    m = min(n, 2000)
    if m < n:
        idx = rng.choice(n, size=m, replace=False)
        Z = X[idx]
    else:
        Z = X
    # random pairs
    n_pairs = min(max_pairs, m * (m - 1) // 2)
    i = rng.integers(0, m, size=n_pairs)
    j = rng.integers(0, m, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 1.0
    dists = np.linalg.norm(Z[i] - Z[j], axis=1)
    dists = dists[dists > 0]
    if len(dists) == 0:
        return 1.0
    sigma = float(np.median(dists))
    if sigma < 1e-12:
        return 1.0
    return 1.0 / (2.0 * sigma * sigma)


def rbf_mmd(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seed: int,
) -> float:
    """Biased RBF-MMD^2 with median-heuristic bandwidth on pooled features."""
    rng = np.random.default_rng(seed)
    X = np.vstack([X_a, X_b])
    gamma = _median_heuristic_gamma(X, rng)

    def _k(U, V):
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u·v
        u2 = np.sum(U * U, axis=1)[:, None]
        v2 = np.sum(V * V, axis=1)[None, :]
        sq = np.maximum(u2 + v2 - 2.0 * (U @ V.T), 0.0)
        return np.exp(-gamma * sq)

    Kaa = _k(X_a, X_a)
    Kbb = _k(X_b, X_b)
    Kab = _k(X_a, X_b)
    n = len(X_a)
    m = len(X_b)
    # biased estimator
    mmd2 = (
        Kaa.sum() / (n * n)
        + Kbb.sum() / (m * m)
        - 2.0 * Kab.sum() / (n * m)
    )
    return float(max(mmd2, 0.0))


def pairwise_domain_gap(
    X_a: np.ndarray,
    X_b: np.ndarray,
    seeds: Iterable[int],
    domain_cap: int = DEFAULT_DOMAIN_CAP,
    mmd_cap: int = DEFAULT_MMD_CAP,
) -> pd.DataFrame:
    """Run multi-seed domain-gap metrics; returns one row per seed."""
    rows = []
    for seed in seeds:
        Xa_d, Xb_d, info_d = _subsample_pair(X_a, X_b, seed=seed, cap=domain_cap)

        # Standardize on the pooled domain-classifier sample for MMD consistency
        scaler = StandardScaler().fit(np.vstack([Xa_d, Xb_d]))
        Za = scaler.transform(Xa_d)
        Zb = scaler.transform(Xb_d)

        clf = proxy_a_distance_and_auc(Za, Zb, seed=seed)

        # MMD on (possibly further) capped standardized features
        if info_d["n_used"] > mmd_cap:
            Xa_m, Xb_m, info_m = _subsample_pair(
                Za, Zb, seed=seed + 10_000, cap=mmd_cap
            )
            n_mmd = int(info_m["n_used"])
        else:
            Xa_m, Xb_m = Za, Zb
            n_mmd = int(info_d["n_used"])

        mmd_lin = linear_mmd(Xa_m, Xb_m)
        mmd_rbf = rbf_mmd(Xa_m, Xb_m, seed=seed)

        rows.append({
            "seed": int(seed),
            "n_a_full": info_d["n_a_full"],
            "n_b_full": info_d["n_b_full"],
            "n_balanced": info_d["n_balanced"],
            "n_domain": info_d["n_used"],
            "n_mmd": n_mmd,
            "auc_lr": clf["auc_lr"],
            "auc_rf": clf["auc_rf"],
            "err_lr": clf["err_lr"],
            "err_rf": clf["err_rf"],
            "da_lr": clf["da_lr"],
            "da_rf": clf["da_rf"],
            # Primary d̂_A from LR (classic linear proxy); RF reported via AUC.
            "da": clf["da_lr"],
            "mmd_linear": mmd_lin,
            "mmd_rbf": mmd_rbf,
            "mmd": mmd_rbf,
        })
    return pd.DataFrame(rows)


def summarize_domain_gap(details: pd.DataFrame, pair: str) -> dict:
    """Mean±std summary for one unordered pair."""
    def _ms(col):
        v = details[col].to_numpy(dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=0))

    auc_lr_m, auc_lr_s = _ms("auc_lr")
    auc_rf_m, auc_rf_s = _ms("auc_rf")
    da_m, da_s = _ms("da")
    mmd_m, mmd_s = _ms("mmd")
    mmd_lin_m, mmd_lin_s = _ms("mmd_linear")

    n_a = int(details["n_a_full"].iloc[0])
    n_b = int(details["n_b_full"].iloc[0])
    return {
        "pair": pair,
        "n_A": n_a,
        "n_B": n_b,
        "domain_AUC_LR": f"{auc_lr_m:.4f}±{auc_lr_s:.4f}",
        "domain_AUC_RF": f"{auc_rf_m:.4f}±{auc_rf_s:.4f}",
        "dA_hat": f"{da_m:.4f}±{da_s:.4f}",
        "MMD": f"{mmd_m:.4f}±{mmd_s:.4f}",
        "MMD_linear": f"{mmd_lin_m:.4f}±{mmd_lin_s:.4f}",
        "auc_lr_mean": auc_lr_m,
        "auc_lr_std": auc_lr_s,
        "auc_rf_mean": auc_rf_m,
        "auc_rf_std": auc_rf_s,
        "da_mean": da_m,
        "da_std": da_s,
        "mmd_mean": mmd_m,
        "mmd_std": mmd_s,
        "mmd_linear_mean": mmd_lin_m,
        "mmd_linear_std": mmd_lin_s,
        "n_domain": int(details["n_domain"].iloc[0]),
        "n_mmd": int(details["n_mmd"].iloc[0]),
    }
