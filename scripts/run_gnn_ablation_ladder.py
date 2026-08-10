"""
STEP A–C: in-distribution GNN ablation ladder vs RF ceiling.

  A1  MLP on x_ud only (control; should approach RF ~0.97)
  A2  memory-only day readout
  A3  standardize(x_ud) || memory
  B/C day-level supervision + scaler + 128-64 MLP + BCE/focal + early stop

Usage:
  python -m scripts.run_gnn_ablation_ladder
  python -m scripts.run_gnn_ablation_ladder --only A1
  python -m scripts.run_gnn_ablation_ladder --only A,BC
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.loaders import load_cert, load_spedia
from src.train.ablation_ladder import run_ablation_ladder, run_bc_id
from src.train.supervised_transfer import (
    cross_dataset_transfer,
    print_transfer_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_gnn_ablation_ladder")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"


def _load():
    log.info("Loading CERT (insider_aware http) + SPEDIA real_only ...")
    cert = load_cert(
        CERT,
        sources=("logon", "device", "file", "email"),
        http_mode="insider_aware",
        benign_http_frac=0.05,
    )
    spedia = load_spedia(SPEDIA, real_only=True)
    log.info("CERT rows=%d SPEDIA rows=%d", len(cert), len(spedia))
    return {"cert": cert, "spedia": spedia}


def _print_ladder(df: pd.DataFrame, title: str):
    print(f"\n======== {title} ========")
    cols = [c for c in (
        "dataset", "split", "variant", "pr_auc", "lift", "base_rate",
        "n", "n_pos", "best_epoch", "best_val_pr", "loss",
    ) if c in df.columns]
    print(df[cols].sort_values(["dataset", "split", "variant"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}",
    ))


def _competitive(bc_df: pd.DataFrame, rf_thresh: float = 0.80) -> bool:
    """True if best CERT ID PR-AUC under BC is near RF territory."""
    sub = bc_df[bc_df["dataset"] == "cert"]
    if sub.empty:
        return False
    best = float(sub["pr_auc"].max())
    log.info("Competitive check: best CERT BC pr_auc=%.4f (thresh=%.2f)", best, rf_thresh)
    return best >= rf_thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", default="A,BC",
        help="Comma list: A, A1, A2, A3, BC, D",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ssl-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    dfs = _load()

    a_df = None
    if only & {"A", "A1", "A2", "A3"}:
        variants = []
        if "A" in only:
            variants = ["A1", "A2", "A3"]
        else:
            for v in ("A1", "A2", "A3"):
                if v in only:
                    variants.append(v)
        log.info("STEP A variants=%s", variants)
        # A1 is cheap; A2/A3 share day-GNN knobs
        a_df = run_ablation_ladder(
            dfs,
            variants=tuple(variants),
            seed=args.seed,
            a1_kw=dict(epochs=max(args.epochs, 60), patience=12),
            gnn_kw=dict(
                epochs=args.epochs,
                ssl_epochs=args.ssl_epochs,
                patience=10,
                min_epochs=15,
                lr_schedule="cosine",
                loss="bce",
            ),
        )
        a_path = out / "ablation_ladder_A.csv"
        a_df.to_csv(a_path, index=False)
        _print_ladder(a_df, "STEP A ablation ladder")
        print(f"\nSaved {a_path}")

        # Quick interpretation aid
        for ds in a_df["dataset"].unique():
            for sp in a_df["split"].unique():
                sub = a_df[(a_df["dataset"] == ds) & (a_df["split"] == sp)]
                if sub.empty:
                    continue
                a1 = sub.loc[sub["variant"] == "A1", "pr_auc"]
                if len(a1):
                    v = float(a1.iloc[0])
                    verdict = (
                        "head+training OK → dig into memory integration"
                        if v >= 0.80 else
                        "head/training/readout pipeline broken → fix B first"
                    )
                    print(f"  interpret {ds}/{sp}: A1={v:.4f} → {verdict}")

    bc_df = None
    if "BC" in only or "B" in only or "C" in only:
        log.info("STEP B/C day-supervised + capacity")
        bc_df = run_bc_id(
            dfs,
            seed=args.seed,
            losses=("bce", "focal"),
            epochs=max(args.epochs, 80),
            ssl_epochs=args.ssl_epochs,
            patience=12,
            min_epochs=20,
            lr_schedule="cosine",
        )
        bc_path = out / "ablation_ladder_BC.csv"
        bc_df.to_csv(bc_path, index=False)
        _print_ladder(bc_df, "STEP B/C day-supervised")
        print(f"\nSaved {bc_path}")

    if "D" in only or (bc_df is not None and _competitive(bc_df)):
        if bc_df is not None and not _competitive(bc_df) and "D" not in only:
            print("\nSTEP D skipped: in-distribution GNN not yet competitive with RF.")
            return
        log.info("STEP D: honest cross-dataset transfer (GNN vs RF)")
        for protocol in ("temporal", "user_disjoint"):
            pr, lift, details, tstats = cross_dataset_transfer(
                dfs, "random_forest", deviation=True,
                diagonal_protocol=protocol, seed=args.seed,
            )
            print_transfer_report(
                "random_forest", pr, lift, details, tstats, protocol,
            )
            pr.to_csv(out / f"transfer_pr_auc_random_forest_{protocol}_D.csv")
            lift.to_csv(out / f"transfer_lift_random_forest_{protocol}_D.csv")

            gnn_kw = dict(
                epochs=max(args.epochs, 80),
                ssl_epochs=args.ssl_epochs,
                # reuse day-supervised path via TemporalGNNDetector? keep prior
                # honest GNN transfer API; day-sup detector plugged below if needed
            )
            # Prefer DaySupervisedGNN-backed transfer for fair B/C comparison
            from src.train.day_supervised_gnn import DaySupervisedGNN
            from src.train.supervised_transfer import (
                _base_rate, _with_lift, generalization_gap,
            )
            from src.data.features import user_day_features
            from src.data.schema import LABEL
            from src.train.splits import temporal_split, user_disjoint_split
            from src.train.evaluate import compute_metrics

            names = list(dfs.keys())
            full_feats = {n: user_day_features(dfs[n], deviation=True) for n in names}
            target_stats = {}
            for n in names:
                y = full_feats[n][LABEL].to_numpy()
                target_stats[n] = {
                    "base_rate": _base_rate(y),
                    "n": int(len(y)),
                    "n_pos": int(y.sum()),
                }
            pr_mat = pd.DataFrame(index=names, columns=names, dtype=float)
            lift_mat = pd.DataFrame(index=names, columns=names, dtype=float)
            gdetails = {}
            for s in names:
                for t in names:
                    det = DaySupervisedGNN(
                        variant="BC", loss="bce",
                        epochs=max(args.epochs, 80),
                        ssl_epochs=args.ssl_epochs,
                        patience=12, min_epochs=20,
                        seed=args.seed,
                    )
                    det.set_full_features(full_feats[s] if s == t else full_feats[s])
                    if s == t:
                        if protocol == "temporal":
                            tr, te = temporal_split(dfs[s])
                        else:
                            tr, te = user_disjoint_split(dfs[s], seed=args.seed)
                        det.set_full_features(full_feats[s])
                        det.fit(tr)
                        agg = det.score_user_day(te)
                    else:
                        det.set_full_features(full_feats[s])
                        det.fit(dfs[s])
                        # score target with target's full-stream features
                        det.set_full_features(full_feats[t])
                        agg = det.score_user_day(dfs[t])
                    y = agg["label"].to_numpy()
                    m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
                    m.update({
                        "source": s, "target": t, "model": "day_supervised_gnn",
                        "diagonal_protocol": protocol if s == t else "full_source",
                    })
                    gdetails[(s, t)] = m
                    pr_mat.loc[s, t] = m["pr_auc"]
                    lift_mat.loc[s, t] = m["lift"]
                    log.info(
                        "D GNN %s->%s [%s]: pr_auc=%.4f lift=%.2f",
                        s, t, m["diagonal_protocol"], m["pr_auc"], m["lift"],
                    )
            print(f"\n=== STEP D GNN ({protocol}) PR-AUC ===")
            print(pr_mat.round(4).to_string())
            print(f"lift gap={generalization_gap(lift_mat):.3f}")
            pr_mat.to_csv(out / f"transfer_pr_auc_day_gnn_{protocol}_D.csv")
            lift_mat.to_csv(out / f"transfer_lift_day_gnn_{protocol}_D.csv")
            pd.DataFrame(list(gdetails.values())).to_csv(
                out / f"transfer_details_day_gnn_{protocol}_D.csv", index=False,
            )
    elif bc_df is not None:
        print("\nSTEP D skipped: in-distribution GNN not yet competitive with RF.")


if __name__ == "__main__":
    main()
