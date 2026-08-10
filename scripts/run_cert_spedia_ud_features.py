"""TASK 3: GNN with causal deviation user-day readout on CERT (insider_aware) x SPEDIA."""
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
    gnn_id_eval,
    run_cross_dataset_gnn_settings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
SOURCES = ("logon", "device", "file", "email")
EPOCHS = 25
SSL_EPOCHS = 5
BATCH_EDGES = 8192


def main():
    print(
        "device:",
        "cuda" if torch.cuda.is_available() else "cpu",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    )
    out = Path("results")
    out.mkdir(exist_ok=True)

    print("Loading CERT http_mode=insider_aware ...")
    t0 = time.time()
    cert = load_cert(
        CERT, sources=SOURCES, http_mode="insider_aware",
        benign_http_frac=0.05, seed=7,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(cert); validate(spedia)
    print(f"Load wall time: {time.time()-t0:.1f}s")
    for name, df in (("cert", cert), ("spedia", spedia)):
        s = summary_stats(df)
        print(
            f"  {name}: users={s['n_users']} hosts={s['n_hosts']} "
            f"edges={s['n_edges']} mal_rate={s['mal_edge_rate']:.4f}"
        )

    dfs = {"cert": cert, "spedia": spedia}
    gnn_kw = dict(
        epochs=EPOCHS, ssl_epochs=SSL_EPOCHS, batch_edges=BATCH_EDGES,
        pos_weight=True, use_ud_features=True, deviation=True,
    )

    # CERT->CERT diagonal first
    print("\n=== CERT->CERT (ud features + deviation) ===")
    t1 = time.time()
    det = TemporalGNNDetector(use_dann=False, **gnn_kw)
    det.fit(cert)
    agg = det.score_user_day(cert)
    m = compute_metrics(agg["label"].to_numpy(), agg["score"].to_numpy())
    print(
        f"cert->cert PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
        f"DR@1%={m['dr_at_1pct_fpr']:.3f} n={m['n']} pos={m['n_pos']} "
        f"({time.time()-t1:.1f}s)"
    )
    pd.DataFrame([{"source": "cert", "target": "cert", **m}]).to_csv(
        out / "cert_gnn_diag_ud_deviation.csv", index=False
    )

    # ID splits
    id_table = gnn_id_eval(dfs, **gnn_kw)
    id_table.to_csv(out / "cert_spedia_gnn_id_eval_ud_deviation.csv", index=False)

    # Zero-shot + DANN matrices
    both = run_cross_dataset_gnn_settings(dfs, **gnn_kw)
    for name, (matrix, details) in both.items():
        matrix.to_csv(out / f"cert_spedia_gnn_pr_auc_{name}_ud_deviation.csv")
        rows = [{"source": s, "target": t, **mm} for (s, t), mm in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"cert_spedia_gnn_details_{name}_ud_deviation.csv", index=False
        )
        print(f"{name} gap: {generalization_gap(matrix):.3f}")
        print(f"{name} cert->cert: {float(matrix.loc['cert', 'cert']):.3f}")


if __name__ == "__main__":
    main()
