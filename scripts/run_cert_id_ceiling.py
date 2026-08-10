"""STEP 1–2: supervised CERT in-distribution ceiling (temporal + user-disjoint)."""
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
from src.data.loaders import load_cert
from src.data.schema import validate
from src.train.evaluate import compute_metrics
from src.train.splits import temporal_split, user_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SOURCES = ("logon", "device", "file", "email")


def _supervised_models(seed=0):
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
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
            )),
        ]),
    }


def _score_supervised(model, Xte):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    # Pipeline
    return model.predict_proba(Xte)[:, 1]


def eval_cert_supervised(df, tag: str, seed=0) -> pd.DataFrame:
    rows = []
    models = _supervised_models(seed=seed)
    for split_name, splitter in (
        ("temporal", lambda d: temporal_split(d, train_frac=0.7)),
        ("user_disjoint", lambda d: user_disjoint_split(d, train_frac=0.7, seed=seed)),
    ):
        train_df, test_df = splitter(df)
        Xtr, ytr = X_y(user_day_features(train_df))
        Xte, yte = X_y(user_day_features(test_df))
        print(
            f"[{tag}/{split_name}] train n={len(ytr)} pos={ytr.mean():.4f} | "
            f"test n={len(yte)} pos={yte.mean():.4f}"
        )
        for mname, model in models.items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            scores = _score_supervised(model, Xte)
            m = compute_metrics(yte, scores)
            m.update({
                "tag": tag,
                "split": split_name,
                "model": mname,
                "fit_s": round(time.time() - t0, 2),
            })
            rows.append(m)
            print(
                f"  {mname:14s} PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
                f"DR@1%={m['dr_at_1pct_fpr']:.3f} ({m['fit_s']}s)"
            )
    return pd.DataFrame(rows)


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)

    # ---- STEP 1: no http ----
    print("\n======== STEP 1: CERT supervised ID (no http) ========")
    t0 = time.time()
    cert = load_cert(CERT, sources=SOURCES, include_http=False)
    validate(cert)
    s = summary_stats(cert)
    print(
        f"loaded in {time.time()-t0:.1f}s | users={s['n_users']} edges={s['n_edges']} "
        f"mal_rate={s['mal_edge_rate']:.4f}"
    )
    step1 = eval_cert_supervised(cert, tag="no_http")
    step1.to_csv(out / "cert_supervised_id_no_http.csv", index=False)

    # ---- STEP 2: + http sample ----
    print("\n======== STEP 2: CERT supervised ID (+ http_nrows=5e6) ========")
    t0 = time.time()
    cert_http = load_cert(
        CERT,
        sources=SOURCES,
        include_http=True,
        http_nrows=5_000_000,
    )
    validate(cert_http)
    s = summary_stats(cert_http)
    print(
        f"loaded in {time.time()-t0:.1f}s | users={s['n_users']} edges={s['n_edges']} "
        f"mal_rate={s['mal_edge_rate']:.4f}"
    )
    step2 = eval_cert_supervised(cert_http, tag="with_http_5m")
    step2.to_csv(out / "cert_supervised_id_with_http_5m.csv", index=False)

    print("\n======== SUMMARY (PR-AUC) ========")
    cols = ["tag", "split", "model", "pr_auc", "roc_auc", "dr_at_1pct_fpr", "n", "n_pos"]
    print(pd.concat([step1, step2])[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
