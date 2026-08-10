"""STEP D: honest cross-dataset transfer — RF vs day-supervised GNN (zero-shot + DANN-UDA)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.features import user_day_features
from src.data.loaders import load_cert, load_spedia
from src.data.schema import LABEL
from src.train.day_supervised_gnn import DaySupervisedGNN, _with_lift
from src.train.evaluate import compute_metrics
from src.train.splits import temporal_split, user_disjoint_split
from src.train.supervised_transfer import (
    _base_rate,
    cross_dataset_transfer,
    generalization_gap,
    print_transfer_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_step_d")

CERT = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"


def _gnn_kw(source: str, use_dann: bool):
    return dict(
        variant="BC",
        loss="bce",
        epochs=60 if source == "cert" else 80,
        ssl_epochs=1 if source == "cert" else 2,
        patience=8 if source == "cert" else 12,
        min_epochs=12 if source == "cert" else 20,
        val_every=4 if source == "cert" else 2,
        batch_edges=16384 if source == "cert" else 8192,
        use_dann=use_dann,
        seed=0,
    )


def gnn_transfer(dfs, full_feats, protocol, use_dann=False):
    names = list(dfs.keys())
    setting = "dann_uda" if use_dann else "zero_shot"
    target_stats = {}
    for n in names:
        y = full_feats[n][LABEL].to_numpy()
        target_stats[n] = {
            "base_rate": _base_rate(y), "n": int(len(y)), "n_pos": int(y.sum()),
        }
    pr_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    lift_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    details = {}
    for s in names:
        for t in names:
            adapt = bool(use_dann and s != t)
            det = DaySupervisedGNN(**_gnn_kw(s, use_dann=adapt))
            if s == t:
                if protocol == "temporal":
                    tr, te = temporal_split(dfs[s])
                else:
                    tr, te = user_disjoint_split(dfs[s], seed=0)
                det.set_full_features(full_feats[s])
                det.fit(tr)
                agg = det.score_user_day(te)
            else:
                det.set_full_features(full_feats[s])
                det.fit(dfs[s], df_tgt=(dfs[t] if adapt else None))
                det.set_full_features(full_feats[t])
                agg = det.score_user_day(dfs[t])
            y = agg["label"].to_numpy()
            m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
            m.update({
                "source": s, "target": t, "model": "day_supervised_gnn",
                "setting": setting,
                "diagonal_protocol": protocol if s == t else "full_source",
            })
            details[(s, t)] = m
            pr_mat.loc[s, t] = m["pr_auc"]
            lift_mat.loc[s, t] = m["lift"]
            log.info(
                "D GNN [%s] %s->%s [%s]: pr_auc=%.4f lift=%.2f",
                setting, s, t, m["diagonal_protocol"], m["pr_auc"], m["lift"],
            )
    return pr_mat, lift_mat, details, target_stats


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    log.info("Loading...")
    dfs = {
        "cert": load_cert(
            CERT, sources=("logon", "device", "file", "email"),
            http_mode="insider_aware", benign_http_frac=0.05,
        ),
        "spedia": load_spedia(SPEDIA, real_only=True),
    }
    full_feats = {n: user_day_features(df, deviation=True) for n, df in dfs.items()}

    for protocol in ("temporal", "user_disjoint"):
        print(f"\n######## STEP D protocol={protocol} ########")
        pr, lift, details, tstats = cross_dataset_transfer(
            dfs, "random_forest", deviation=True,
            diagonal_protocol=protocol, seed=0,
        )
        print_transfer_report("random_forest", pr, lift, details, tstats, protocol)
        pr.to_csv(out / f"transfer_pr_auc_random_forest_{protocol}_D.csv")
        lift.to_csv(out / f"transfer_lift_random_forest_{protocol}_D.csv")

        for use_dann, tag in ((False, "zero_shot"), (True, "dann_uda")):
            gpr, glift, gdet, _ = gnn_transfer(
                dfs, full_feats, protocol, use_dann=use_dann,
            )
            print(f"\n=== day_gnn {tag} PR-AUC ({protocol}) ===")
            print(gpr.round(4).to_string())
            print(f"\n=== day_gnn {tag} lift ({protocol}) ===")
            print(glift.round(2).to_string())
            print(
                f"PR gap={generalization_gap(gpr):.3f}  "
                f"lift gap={generalization_gap(glift):.3f}"
            )
            gpr.to_csv(out / f"transfer_pr_auc_day_gnn_{tag}_{protocol}_D.csv")
            glift.to_csv(out / f"transfer_lift_day_gnn_{tag}_{protocol}_D.csv")
            pd.DataFrame(list(gdet.values())).to_csv(
                out / f"transfer_details_day_gnn_{tag}_{protocol}_D.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
