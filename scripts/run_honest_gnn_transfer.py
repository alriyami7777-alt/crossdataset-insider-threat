"""GNN-only honest transfer matrices (RF/LR/IF already saved)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.data.loaders import load_cert, load_spedia
from src.data.schema import validate
from src.train.supervised_transfer import print_transfer_report, run_gnn_transfer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
OUT = Path("results")


def main():
    print("Loading ...")
    t0 = time.time()
    cert = load_cert(
        CERT, sources=("logon", "device", "file", "email"),
        http_mode="insider_aware", benign_http_frac=0.05, seed=7,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(cert); validate(spedia)
    dfs = {"cert": cert, "spedia": spedia}
    print(f"Loaded in {time.time()-t0:.1f}s")

    gnn_kw = dict(epochs=15, ssl_epochs=3, batch_edges=8192)
    for protocol in ("temporal", "user_disjoint"):
        pr, lift, details, tstats = run_gnn_transfer(
            dfs, diagonal_protocol=protocol, seed=0, **gnn_kw
        )
        print_transfer_report("temporal_gnn", pr, lift, details, tstats, protocol)
        tag = f"temporal_gnn_{protocol}"
        pr.to_csv(OUT / f"transfer_pr_auc_{tag}.csv")
        lift.to_csv(OUT / f"transfer_lift_{tag}.csv")
        rows = [{"source": s, "target": t, **m} for (s, t), m in details.items()]
        pd.DataFrame(rows).to_csv(OUT / f"transfer_details_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
