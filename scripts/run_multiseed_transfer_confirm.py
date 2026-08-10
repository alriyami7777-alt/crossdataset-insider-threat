"""
Multi-seed confirmation: does zero-shot day-supervised GNN out-transfer RF?

Claim (single-seed STEP D):
  cert->spedia  GNN 0.65 vs RF 0.55
  spedia->cert  GNN 0.27 vs RF 0.10

This runner repeats the honest protocol across >=5 seeds, reports mean PR-AUC
with 95% bootstrap CIs, lift, and paired GNN−RF deltas. Claim HOLDS only if
off-diagonal deltas are positive with CIs excluding 0.

Usage:
  set PYTHONIOENCODING=utf-8
  python -m scripts.run_multiseed_transfer_confirm
  python -m scripts.run_multiseed_transfer_confirm --seeds 0,1,2,3,4 --skip-dann
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.loaders import load_cert, load_spedia
from src.data.schema import validate
from src.train.multiseed_transfer import (
    print_confirmation_report,
    run_multiseed_confirm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("multiseed_confirm")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
OUT = ROOT / "results"


def _parse_seeds(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4", help="seeds for RF/GNN/baselines")
    ap.add_argument("--dann-seeds", default="0", help="seeds for DANN-UDA (expensive)")
    ap.add_argument(
        "--models",
        default="random_forest,isolation_forest,ocsvm,lstm_ae,transformer,static_gcn,athitd",
        help="comma-separated feature/neural baselines",
    )
    ap.add_argument("--skip-gnn", action="store_true")
    ap.add_argument("--skip-dann", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="only RF + GNN (fast path for claim)")
    ap.add_argument("--parallel-feature-seeds", type=int, default=2)
    ap.add_argument("--tag", default="confirm", help="output filename tag")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    dann_seeds = _parse_seeds(args.dann_seeds)
    if args.skip_baselines:
        feature_models = ["random_forest"]
    else:
        feature_models = [m.strip() for m in args.models.split(",") if m.strip()]

    OUT.mkdir(exist_ok=True)
    log.info("Loading CERT + SPEDIA ...")
    t0 = time.time()
    cert = load_cert(
        CERT,
        sources=("logon", "device", "file", "email"),
        http_mode="insider_aware",
        benign_http_frac=0.05,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(cert)
    validate(spedia)
    dfs = {"cert": cert, "spedia": spedia}
    log.info("Loaded in %.1fs | cert=%d spedia=%d", time.time() - t0, len(cert), len(spedia))

    # Optional CERT releases — skip if paths missing / too slow
    # (load_cert_releases exists but r5.2/r6.2 not required for the claim.)

    t1 = time.time()
    results = run_multiseed_confirm(
        dfs,
        seeds=seeds,
        dann_seeds=dann_seeds,
        feature_models=feature_models,
        run_gnn=not args.skip_gnn,
        run_dann=not args.skip_dann,
        parallel_feature_seeds=args.parallel_feature_seeds,
    )
    log.info("All models finished in %.1fs", time.time() - t1)

    tag = args.tag
    results["details"].to_csv(OUT / f"multiseed_details_{tag}.csv", index=False)
    results["summary"].to_csv(OUT / f"multiseed_summary_{tag}.csv", index=False)
    results["deltas"].to_csv(OUT / f"multiseed_deltas_{tag}.csv", index=False)
    results["verdict"].to_csv(OUT / f"multiseed_verdict_{tag}.csv", index=False)
    results["target_stats"].to_csv(OUT / f"multiseed_target_stats_{tag}.csv", index=False)

    # Wide PR-AUC / lift matrices per model (mean only) for quick glance
    summary = results["summary"]
    for model in summary["model"].unique():
        for protocol in ("temporal", "user_disjoint"):
            # Build matrix: diagonal from protocol rows; off-diag from full_source
            names = list(dfs.keys())
            pr = pd.DataFrame(index=names, columns=names, dtype=float)
            lf = pd.DataFrame(index=names, columns=names, dtype=float)
            sub = summary[summary["model"] == model]
            for s in names:
                for t in names:
                    if s == t:
                        row = sub[
                            (sub["source"] == s) & (sub["target"] == t)
                            & (sub["diagonal_protocol"] == protocol)
                        ]
                    else:
                        row = sub[
                            (sub["source"] == s) & (sub["target"] == t)
                            & (sub["diagonal_protocol"] == "full_source")
                        ]
                    if len(row):
                        pr.loc[s, t] = float(row.iloc[0]["pr_auc_mean"])
                        lf.loc[s, t] = float(row.iloc[0]["lift_mean"])
            safe = model.replace("/", "_")
            pr.to_csv(OUT / f"multiseed_pr_auc_{safe}_{protocol}_{tag}.csv")
            lf.to_csv(OUT / f"multiseed_lift_{safe}_{protocol}_{tag}.csv")

    print_confirmation_report(results)
    print(f"\nSaved CSVs under {OUT}/multiseed_*_{tag}.csv")


if __name__ == "__main__":
    main()
