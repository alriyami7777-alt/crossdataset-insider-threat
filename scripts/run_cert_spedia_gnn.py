"""First real CERT×SPEDIA cross-dataset GNN result (three locked reports)."""
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
from src.train.train_gnn import (
    gnn_id_eval,
    leave_one_domain_out_gnn_settings,
    run_cross_dataset_gnn_settings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
# http skipped (14.5GB). email included; drop to ("logon","device","file") if heavy.
SOURCES = ("logon", "device", "file", "email")


def main():
    print(
        "device:",
        "cuda" if torch.cuda.is_available() else "cpu",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    )

    t_load = time.time()
    print(f"\nLoading CERT sources={SOURCES} (no http) ...")
    cert = load_cert(CERT, sources=SOURCES)
    print(f"Loading SPEDIA real_only=True ...")
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(cert)
    validate(spedia)
    print(f"Load wall time: {time.time() - t_load:.1f}s")

    for name, df in (("cert", cert), ("spedia", spedia)):
        s = summary_stats(df)
        print(
            f"  {name}: users={s['n_users']} hosts={s['n_hosts']} "
            f"edges={s['n_edges']} mal_rate={s['mal_edge_rate']:.4f} "
            f"span={s['span_days']:.1f}d pos_events={int(df.label.sum())}"
        )

    dfs = {"cert": cert, "spedia": spedia}
    out = Path("results")
    out.mkdir(exist_ok=True)
    t0 = time.time()

    # 1) In-distribution controls
    id_table = gnn_id_eval(dfs, epochs=15, ssl_epochs=5)
    id_table.to_csv(out / "cert_spedia_gnn_id_eval.csv", index=False)

    # 2) Zero-shot vs DANN-UDA matrices
    both = run_cross_dataset_gnn_settings(dfs, epochs=15, ssl_epochs=5)
    for name, (matrix, details) in both.items():
        matrix.to_csv(out / f"cert_spedia_gnn_pr_auc_{name}.csv")
        rows = [{"source": s, "target": t, **m} for (s, t), m in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"cert_spedia_gnn_details_{name}.csv", index=False
        )
        print(f"{name} generalization gap: {generalization_gap(matrix):.3f}")

    # 3) Leave-one-domain-out
    lodo = leave_one_domain_out_gnn_settings(dfs, epochs=15, ssl_epochs=5)
    for name, (series, details) in lodo.items():
        series.to_csv(out / f"cert_spedia_gnn_lodo_{name}.csv", header=True)
        rows = [{"held_out": h, **m} for h, m in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"cert_spedia_gnn_lodo_details_{name}.csv", index=False
        )

    print(f"\nElapsed (train+eval): {time.time() - t0:.1f}s")
    print(f"Wrote results under {out.resolve()}")


if __name__ == "__main__":
    main()
