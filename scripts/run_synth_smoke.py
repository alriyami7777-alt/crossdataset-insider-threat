"""
End-to-end smoke test on SYNTHETIC data -- no torch, no GPU, no real datasets.

Proves the whole harness is wired: generate two domains -> build graph stats ->
aggregate features -> run the cross-dataset matrix with the Isolation Forest
baseline -> print the in-dist vs out-of-dist metrics, the generalization gap, and
a transfer diagnostic. When real data lands, swap synth loaders for real ones and
rerun the SAME code.

Run:  python -m scripts.run_synth_smoke      (from repo root)
"""
from __future__ import annotations

import sys
import numpy as np

from src.data import synth, loaders
from src.data.graph_builder import summary_stats
from src.data.features import user_day_features, X_y
from src.models.baselines import isolation_forest
from src.train.crossdataset import cross_dataset_matrix, generalization_gap
from src.train.evaluate import compute_metrics, bootstrap_ci
from src.explain.xai import transfer_diagnostic


def main():
    np.set_printoptions(precision=3, suppress=True)
    print("== 1. Generate two synthetic domains (source=synthA, target=synthB) ==")
    a, b = synth.two_domains(seed=7)
    dfs = {"synthA": a, "synthB": b}
    for name, df in dfs.items():
        s = summary_stats(df)
        print(f"  {name}: users={s['n_users']} hosts={s['n_hosts']} "
              f"edges={s['n_edges']} mal_rate={s['mal_edge_rate']:.4f} "
              f"span={s['span_days']:.1f}d")

    print("\n== 2. Cross-dataset matrix (metric = PR-AUC), Isolation Forest ==")
    matrix, details = cross_dataset_matrix(dfs, isolation_forest, metric="pr_auc")
    print(matrix.round(3).to_string())
    gap = generalization_gap(matrix)
    print(f"\n  Generalization gap (mean diag - mean off-diag PR-AUC): {gap:.3f}")
    print("  (Positive gap = models do worse cross-dataset -> the paper's claim.)")

    print("\n== 3. Detailed metrics for A->A (in-dist) vs A->B (out-of-dist) ==")
    for cell in [("synthA", "synthA"), ("synthA", "synthB")]:
        m = details[cell]
        print(f"  {cell[0]}->{cell[1]}: PR-AUC={m['pr_auc']:.3f} "
              f"ROC-AUC={m['roc_auc']:.3f} DR@1%FPR={m['dr_at_1pct_fpr']:.3f} "
              f"P@k={m['precision_at_k']:.3f} (n={m['n']}, pos={m['n_pos']})")

    print("\n== 4. Bootstrap 95% CI for A->B PR-AUC ==")
    fa = user_day_features(a); Xa, ya = X_y(fa)
    fb = user_day_features(b); Xb, yb = X_y(fb)
    model = isolation_forest(); model.fit(Xa)
    scores_b = model.anomaly_scores(Xb)
    point, lo, hi = bootstrap_ci(yb, scores_b, metric="pr_auc", n_boot=300)
    print(f"  A->B PR-AUC = {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n== 5. Transfer diagnostic (which features transfer vs are artifacts) ==")
    def pr(y, s):
        from sklearn.metrics import average_precision_score as ap
        return ap(y, s) if len(set(y)) > 1 else float("nan")
    rows = transfer_diagnostic(model, Xa, ya, Xb, yb, pr, top=6)
    for r in rows:
        print(f"  {r['feature']:>18}  imp_src={r['imp_src']:.2f} "
              f"imp_tgt={r['imp_tgt']:.2f} transfer={r['transferability']:.2f} "
              f"artifact={r['artifact_score']:+.2f}")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    sys.exit(main())
