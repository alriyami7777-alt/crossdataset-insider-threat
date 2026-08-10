"""Decisive honest cross-dataset transfer matrices (RF first, then peers + GNN)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.data.graph_builder import summary_stats
from src.data.loaders import load_cert, load_spedia
from src.data.schema import validate
from src.train.supervised_transfer import (
    cross_dataset_transfer,
    generalization_gap,
    print_transfer_report,
    run_gnn_transfer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
SOURCES = ("logon", "device", "file", "email")
OUT = Path("results")


def _save(model, protocol, pr, lift, details, tstats):
    OUT.mkdir(exist_ok=True)
    tag = f"{model}_{protocol}"
    pr.to_csv(OUT / f"transfer_pr_auc_{tag}.csv")
    lift.to_csv(OUT / f"transfer_lift_{tag}.csv")
    rows = [{"source": s, "target": t, **m} for (s, t), m in details.items()]
    pd.DataFrame(rows).to_csv(OUT / f"transfer_details_{tag}.csv", index=False)
    pd.DataFrame([{"target": k, **v} for k, v in tstats.items()]).to_csv(
        OUT / f"transfer_target_stats_{tag}.csv", index=False
    )


def main():
    print("Loading datasets ...")
    t0 = time.time()
    cert = load_cert(
        CERT, sources=SOURCES, http_mode="insider_aware",
        benign_http_frac=0.05, seed=7,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(cert); validate(spedia)
    dfs = {"cert": cert, "spedia": spedia}
    for n, df in dfs.items():
        s = summary_stats(df)
        print(
            f"  {n}: edges={s['n_edges']} mal_rate={s['mal_edge_rate']:.4f} "
            f"users={s['n_users']}"
        )
    print(f"Load wall time: {time.time()-t0:.1f}s")

    # ---- RandomForest FIRST (decisive) ----
    print("\n################ RANDOM FOREST (decisive) ################")
    for protocol in ("temporal", "user_disjoint"):
        pr, lift, details, tstats = cross_dataset_transfer(
            dfs, "random_forest", deviation=True,
            diagonal_protocol=protocol, seed=0,
        )
        print_transfer_report("random_forest", pr, lift, details, tstats, protocol)
        _save("random_forest", protocol, pr, lift, details, tstats)

    # ---- Other feature models (skip OCSVM: O(n^2) on ~330k CERT user-days) ----
    for model in ("logistic_regression", "isolation_forest"):
        print(f"\n################ {model.upper()} ################")
        for protocol in ("temporal", "user_disjoint"):
            pr, lift, details, tstats = cross_dataset_transfer(
                dfs, model, deviation=True,
                diagonal_protocol=protocol, seed=0,
            )
            print_transfer_report(model, pr, lift, details, tstats, protocol)
            _save(model, protocol, pr, lift, details, tstats)
    print("\n[skip] ocsvm: infeasible at CERT user-day scale (~3e5 rows); not reported.")

    # ---- GNN on same protocol (lighter epochs for wall-clock) ----
    print("\n################ TEMPORAL GNN (ud features) ################")
    gnn_kw = dict(epochs=15, ssl_epochs=3, batch_edges=8192)
    for protocol in ("temporal", "user_disjoint"):
        pr, lift, details, tstats = run_gnn_transfer(
            dfs, diagonal_protocol=protocol, seed=0, **gnn_kw
        )
        print_transfer_report("temporal_gnn", pr, lift, details, tstats, protocol)
        _save("temporal_gnn", protocol, pr, lift, details, tstats)

    print("\nDone. CSVs under results/transfer_*.csv")


if __name__ == "__main__":
    main()
