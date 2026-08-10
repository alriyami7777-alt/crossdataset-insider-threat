"""Temporary runner: fixed A1 only."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.features import user_day_features
from src.data.loaders import load_cert, load_spedia
from src.train.ablation_ladder import run_a1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"


def main():
    print("loading...")
    cert = load_cert(
        CERT,
        sources=("logon", "device", "file", "email"),
        http_mode="insider_aware",
        benign_http_frac=0.05,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    rows = []
    for name, df in (("cert", cert), ("spedia", spedia)):
        feat = user_day_features(df, deviation=True)
        rows.extend(run_a1(
            feat, name, seed=0,
            epochs=100, patience=25, min_epochs=50,
        ))
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "ablation_ladder_A1.csv"
    out.to_csv(path, index=False)
    print(out[["dataset", "split", "variant", "pr_auc", "lift", "best_epoch"]].to_string(index=False))
    print("saved", path)


if __name__ == "__main__":
    main()
