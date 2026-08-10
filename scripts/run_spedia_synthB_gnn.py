"""SPEDIA <-> synthB: ID splits, zero-shot + DANN matrices, and LODO."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.loaders import load, load_spedia
from src.data.schema import validate
from src.train.crossdataset import generalization_gap
from src.train.train_gnn import (
    gnn_id_eval,
    leave_one_domain_out_gnn_settings,
    run_cross_dataset_gnn_settings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SPEDIA_PATH = Path(r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv")


def main():
    print(
        "device:",
        "cuda" if torch.cuda.is_available() else "cpu",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    )

    # Verify-snippet path (load without real_only) kept available; for the paper
    # cell we drop CERT-derived rows and log the subsetting inside the loader.
    spedia = load_spedia(str(SPEDIA_PATH), real_only=True)
    synthB = load("synthB")
    validate(spedia)
    validate(synthB)
    dfs = {"spedia": spedia, "synthB": synthB}
    print(f"spedia real_only n={len(spedia)} pos={spedia.label.mean():.3f}")
    print(f"synthB n={len(synthB)} pos={synthB.label.mean():.3f}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    t0 = time.time()

    # A) Honest in-distribution eval (temporal AND user-disjoint)
    id_table = gnn_id_eval(dfs, epochs=15, ssl_epochs=5)
    id_table.to_csv(out / "spedia_synthB_gnn_id_eval.csv", index=False)

    # B) Both settings (zero-shot + DANN-UDA). zero_shot matrix is the
    #    backward-compatible verify-snippet result.
    both = run_cross_dataset_gnn_settings(dfs, epochs=15, ssl_epochs=5)
    for name, (matrix, details) in both.items():
        matrix.to_csv(out / f"spedia_synthB_gnn_pr_auc_{name}.csv")
        rows = [{"source": s, "target": t, **mm} for (s, t), mm in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"spedia_synthB_gnn_details_{name}.csv", index=False
        )
        print(f"{name} generalization gap: {generalization_gap(matrix):.3f}")

    # C) Leave-one-domain-out under both settings
    lodo = leave_one_domain_out_gnn_settings(dfs, epochs=15, ssl_epochs=5)
    for name, (series, details) in lodo.items():
        series.to_csv(out / f"spedia_synthB_gnn_lodo_{name}.csv", header=True)
        rows = [{"held_out": h, **mm} for h, mm in details.items()]
        pd.DataFrame(rows).to_csv(
            out / f"spedia_synthB_gnn_lodo_details_{name}.csv", index=False
        )

    print(f"\nElapsed: {time.time() - t0:.1f}s")
    print(f"Wrote results under {out.resolve()}")


if __name__ == "__main__":
    main()
