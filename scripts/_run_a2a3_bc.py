"""Run A2/A3 + B/C after A1, saving incrementally."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.features import user_day_features
from src.data.loaders import load_cert, load_spedia
from src.train.ablation_ladder import run_a2_a3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_a2a3_bc")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"


def main():
    log.info("Loading data...")
    dfs = {
        "cert": load_cert(
            CERT,
            sources=("logon", "device", "file", "email"),
            http_mode="insider_aware",
            benign_http_frac=0.05,
        ),
        "spedia": load_spedia(SPEDIA, real_only=True),
    }
    full_feats = {n: user_day_features(df, deviation=True) for n, df in dfs.items()}

    out_a = ROOT / "results" / "ablation_ladder_A.csv"
    out_bc = ROOT / "results" / "ablation_ladder_BC.csv"
    a1_path = ROOT / "results" / "ablation_ladder_A1.csv"
    # Keep finished A1 (+ any prior SPEDIA A2/A3); drop incomplete CERT GNN rows
    if a1_path.exists():
        rows_a = list(pd.read_csv(a1_path).to_dict("records"))
    elif out_a.exists():
        prev = pd.read_csv(out_a)
        rows_a = list(prev[prev["variant"] == "A1"].to_dict("records"))
    else:
        rows_a = []
    # Reuse finished non-CERT A2/A3 if present
    if out_a.exists():
        prev = pd.read_csv(out_a)
        keep = prev[(prev["variant"].isin(["A2", "A3"])) & (prev["dataset"] != "cert")]
        rows_a.extend(keep.to_dict("records"))

    def _kw_for(name: str, epochs: int) -> dict:
        # CERT: leaner schedule (frozen encoder → head-only is cheap per day step,
        # but streaming 5M edges/epoch still dominates).
        if name == "cert":
            return dict(
                epochs=epochs,
                ssl_epochs=1,
                patience=8,
                min_epochs=12,
                val_every=4,
                lr=3e-4,
                lr_schedule="cosine",
                loss="bce",
                batch_edges=16384,
            )
        return dict(
            epochs=epochs,
            ssl_epochs=2,
            patience=12,
            min_epochs=20,
            val_every=2,
            lr=3e-4,
            lr_schedule="cosine",
            loss="bce",
            batch_edges=8192,
        )

    done = {(r["dataset"], r["variant"]) for r in rows_a if r.get("variant") in ("A2", "A3")}
    for name in ("spedia", "cert"):
        df = dfs[name]
        feat = full_feats[name]
        for variant in ("A2", "A3"):
            if (name, variant) in done:
                log.info("SKIP %s %s (already saved)", name, variant)
                continue
            log.info("=== %s %s ===", name, variant)
            batch = run_a2_a3(
                df, feat, name, variant, seed=0, **_kw_for(name, 40),
            )
            rows_a.extend(batch)
            pd.DataFrame(rows_a).to_csv(out_a, index=False)
            for r in batch:
                log.info(
                    "DONE %s %s/%s pr=%.4f lift=%.2f",
                    r["variant"], r["dataset"], r["split"], r["pr_auc"], r["lift"],
                )

    rows_bc = []
    if out_bc.exists():
        rows_bc = list(pd.read_csv(out_bc).to_dict("records"))
    done_bc = {(r["dataset"], r["variant"]) for r in rows_bc}
    for name in ("spedia", "cert"):
        df = dfs[name]
        feat = full_feats[name]
        for loss in ("bce", "focal"):
            tag = f"BC_{loss}"
            if (name, tag) in done_bc:
                log.info("SKIP %s %s", name, tag)
                continue
            kw = _kw_for(name, 60 if name == "cert" else 80)
            kw["loss"] = loss
            log.info("=== %s %s ===", name, tag)
            batch = run_a2_a3(df, feat, name, "BC", seed=0, **kw)
            for r in batch:
                r["variant"] = tag
                r["loss_tag"] = loss
            rows_bc.extend(batch)
            pd.DataFrame(rows_bc).to_csv(out_bc, index=False)
            for r in batch:
                log.info(
                    "DONE %s %s/%s pr=%.4f lift=%.2f",
                    r["variant"], r["dataset"], r["split"], r["pr_auc"], r["lift"],
                )

    print("\n======== STEP A ========")
    print(pd.DataFrame(rows_a)[
        ["dataset", "split", "variant", "pr_auc", "lift", "n", "n_pos"]
    ].sort_values(["dataset", "split", "variant"]).to_string(index=False))
    print("\n======== STEP B/C ========")
    print(pd.DataFrame(rows_bc)[
        ["dataset", "split", "variant", "pr_auc", "lift", "n", "n_pos", "best_epoch"]
    ].sort_values(["dataset", "split", "variant"]).to_string(index=False))


if __name__ == "__main__":
    main()
