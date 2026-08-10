"""TASK 2: supervised CERT/SPEDIA ceiling with vs without causal deviation features."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import user_day_features, X_y
from src.data.graph_builder import summary_stats
from src.data.loaders import load_cert, load_spedia
from src.data.schema import validate
from src.train.evaluate import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
SOURCES = ("logon", "device", "file", "email")
GATE_PR_AUC = 0.30  # CERT user-disjoint + deviation


def _models(seed=0):
    return {
        "rf_balanced": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "lr_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs",
            )),
        ]),
    }


def _split_user_days(feat: pd.DataFrame, split_name: str, seed=0, train_frac=0.7):
    """Split the user-day frame (features already causal on the full timeline).

    Temporal: past user-days -> train, future -> test (history in train may
    inform a user's test-day deviation — that is intentional and leakage-safe).
    User-disjoint: partition users; same as splitting events first.
    """
    from src.data.schema import USER, LABEL
    import numpy as np

    if split_name == "temporal":
        f = feat.sort_values("day", kind="mergesort")
        cut = f["day"].quantile(train_frac)
        train, test = f[f["day"] <= cut], f[f["day"] > cut]
        if len(test) == 0 or len(train) == 0:
            n = max(1, int(len(f) * train_frac))
            train, test = f.iloc[:n], f.iloc[n:]
        return train, test

    if split_name == "user_disjoint":
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
        return (
            feat[feat[USER].isin(train_users)],
            feat[feat[USER].isin(test_users)],
        )
    raise ValueError(split_name)


def eval_one(df, dataset, deviation, seed=0):
    rows = []
    feat_tag = "counts+deviation" if deviation else "counts-only"
    # Build once on the full event stream so temporal test days keep causal history.
    feat = user_day_features(df, deviation=deviation)
    for split_name in ("temporal", "user_disjoint"):
        train_f, test_f = _split_user_days(feat, split_name, seed=seed)
        Xtr, ytr = X_y(train_f)
        Xte, yte = X_y(test_f)
        for mname, model in _models(seed).items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            scores = model.predict_proba(Xte)[:, 1]
            m = compute_metrics(yte, scores)
            rows.append({
                "dataset": dataset,
                "split": split_name,
                "features": feat_tag,
                "model": mname,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "dr_at_1pct_fpr": m["dr_at_1pct_fpr"],
                "n": m["n"],
                "n_pos": m["n_pos"],
                "fit_s": round(time.time() - t0, 2),
            })
            print(
                f"  {dataset:6s} {split_name:14s} {feat_tag:18s} {mname:12s} "
                f"PR-AUC={m['pr_auc']:.3f}"
            )
    return rows


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)

    print("Loading CERT (insider_aware http, benign_http_frac=0.05) ...")
    t0 = time.time()
    cert = load_cert(
        CERT,
        sources=SOURCES,
        http_mode="insider_aware",
        benign_http_frac=0.05,
        seed=7,
    )
    validate(cert)
    sc = summary_stats(cert)
    print(
        f"  cert edges={sc['n_edges']} mal_rate={sc['mal_edge_rate']:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

    print("Loading SPEDIA real_only=True ...")
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(spedia)
    ss = summary_stats(spedia)
    print(f"  spedia edges={ss['n_edges']} mal_rate={ss['mal_edge_rate']:.4f}")

    all_rows = []
    for name, df in (("cert", cert), ("spedia", spedia)):
        for deviation in (False, True):
            print(f"\n=== {name} deviation={deviation} ===")
            all_rows.extend(eval_one(df, name, deviation=deviation))

    table = pd.DataFrame(all_rows)
    table.to_csv(out / "deviation_gate_supervised.csv", index=False)

    # Compact RF-focused gate table
    rf = table[table["model"] == "rf_balanced"]
    pivot = rf.pivot_table(
        index=["dataset", "split"],
        columns="features",
        values="pr_auc",
        aggfunc="first",
    )
    print("\n======== GATE TABLE (RF PR-AUC) ========")
    print(pivot.round(3).to_string())

    cert_ud = rf[
        (rf["dataset"] == "cert")
        & (rf["split"] == "user_disjoint")
        & (rf["features"] == "counts+deviation")
    ]["pr_auc"]
    score = float(cert_ud.iloc[0]) if len(cert_ud) else float("nan")
    print(f"\nCERT user-disjoint counts+deviation RF PR-AUC = {score:.3f}")
    if score > GATE_PR_AUC:
        print(f"GATE CLEARED (>{GATE_PR_AUC}): proceed to TASK 3 (GNN readout).")
    else:
        print(
            f"GATE NOT CLEARED (<={GATE_PR_AUC}): STOP. "
            "Content-free aligned features (even with causal deviation) are "
            "insufficient for CERT's content-driven scenarios."
        )

    print("\n======== FULL (incl. LR) ========")
    cols = ["dataset", "split", "features", "model", "pr_auc", "roc_auc", "n", "n_pos"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
