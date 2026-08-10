"""STEP 3: CERT x SPEDIA GNN with BCE pos_weight + more epochs (+ http sample)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.graph_builder import summary_stats
from src.data.loaders import load_cert, load_spedia
from src.data.schema import validate
from src.train.crossdataset import generalization_gap
from src.train.evaluate import compute_metrics
from src.train.train_gnn import (
    TemporalGNNDetector,
    leave_one_domain_out_gnn_settings,
    run_cross_dataset_gnn_settings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
SOURCES = ("logon", "device", "file", "email")
EPOCHS = 30
SSL_EPOCHS = 5
BATCH_EDGES = 8192


def _cert_diag(df, tag, out, **kw):
    print(f"\n=== CERT->CERT [{tag}] ===")
    t0 = time.time()
    det = TemporalGNNDetector(use_dann=False, pos_weight=True, **kw)
    det.fit(df)
    agg = det.score_user_day(df)
    m = compute_metrics(agg["label"].to_numpy(), agg["score"].to_numpy())
    print(
        f"cert->cert [{tag}] PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
        f"DR@1%={m['dr_at_1pct_fpr']:.3f} n={m['n']} pos={m['n_pos']} "
        f"({time.time()-t0:.1f}s)"
    )
    pd.DataFrame([{"tag": tag, "source": "cert", "target": "cert", **m}]).to_csv(
        out / f"cert_gnn_diag_{tag}.csv", index=False
    )
    return m


def main():
    print(
        "device:",
        "cuda" if torch.cuda.is_available() else "cpu",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    gnn_kw = dict(epochs=EPOCHS, ssl_epochs=SSL_EPOCHS, batch_edges=BATCH_EDGES)

    print("Loading CERT (no http) ...")
    t_load = time.time()
    cert_no = load_cert(CERT, sources=SOURCES, include_http=False)
    validate(cert_no)
    print(f"  no_http edges={len(cert_no)} ({time.time()-t_load:.1f}s)")
    m_no = _cert_diag(cert_no, "posweight_no_http", out, **gnn_kw)

    print("Loading CERT + http_nrows=5e6 ...")
    t_load = time.time()
    cert = load_cert(
        CERT, sources=SOURCES, include_http=True, http_nrows=5_000_000
    )
    validate(cert)
    print(f"  with_http edges={len(cert)} ({time.time()-t_load:.1f}s)")
    m_http = _cert_diag(cert, "posweight_http_5m", out, **gnn_kw)

    spedia = load_spedia(SPEDIA, real_only=True)
    validate(spedia)
    for name, df in (("cert", cert), ("spedia", spedia)):
        s = summary_stats(df)
        print(
            f"  {name}: users={s['n_users']} hosts={s['n_hosts']} "
            f"edges={s['n_edges']} mal_rate={s['mal_edge_rate']:.4f}"
        )

    dfs = {"cert": cert, "spedia": spedia}
    t0 = time.time()

    print("\n=== STEP 3b: zero-shot vs DANN-UDA (pos_weight on, +http) ===")
    print(
        f"(precomputed diag no_http={m_no['pr_auc']:.3f} "
        f"http={m_http['pr_auc']:.3f})"
    )
    both = run_cross_dataset_gnn_settings(dfs, pos_weight=True, **gnn_kw)
    for name, (matrix, details) in both.items():
        matrix.to_csv(out / f"cert_spedia_gnn_pr_auc_{name}_posweight_http.csv")
        rows = [{"source": s, "target": t, **mm} for (s, t), mm in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"cert_spedia_gnn_details_{name}_posweight_http.csv", index=False
        )
        print(f"{name} gap: {generalization_gap(matrix):.3f}")
        print(f"{name} cert->cert: {matrix.loc['cert', 'cert']:.3f}")

    print("\n=== STEP 3c: LODO (pos_weight on, +http) ===")
    lodo = leave_one_domain_out_gnn_settings(dfs, pos_weight=True, **gnn_kw)
    for name, (series, details) in lodo.items():
        series.to_csv(
            out / f"cert_spedia_gnn_lodo_{name}_posweight_http.csv", header=True
        )
        rows = [{"held_out": h, **mm} for h, mm in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"cert_spedia_gnn_lodo_details_{name}_posweight_http.csv",
            index=False,
        )

    print(f"\nElapsed (train+eval after load): {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
