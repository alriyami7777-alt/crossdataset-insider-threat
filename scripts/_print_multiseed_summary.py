"""Print a concise multiseed confirmation table from saved CSVs."""
from __future__ import annotations

import pandas as pd

s = pd.read_csv("results/multiseed_summary_confirm.csv")
d = pd.read_csv("results/multiseed_deltas_confirm.csv")
det = pd.read_csv("results/multiseed_details_confirm.csv")
t = pd.read_csv("results/multiseed_target_stats_confirm.csv")

print("=== TARGET BASE RATES ===")
print(t.to_string(index=False))

cols = [
    "model", "source", "target", "diagonal_protocol", "cell_kind", "n_seeds",
    "pr_auc_mean", "pr_auc_lo", "pr_auc_hi", "lift_mean", "lift_lo", "lift_hi", "base_rate",
]
print("\n=== OFF-DIAGONAL ===")
off = s[s.cell_kind == "off_diagonal"][cols].sort_values(["source", "target", "model"])
for _, r in off.iterrows():
    print(
        f"{r['model']:22s} {r['source']}->{r['target']}: "
        f"PR={r['pr_auc_mean']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}]  "
        f"lift={r['lift_mean']:.2f} [{r['lift_lo']:.2f},{r['lift_hi']:.2f}]  "
        f"p={r['base_rate']:.5f} n={int(r['n_seeds'])}"
    )

for protocol in ("temporal", "user_disjoint"):
    print(f"\n=== DIAGONAL {protocol} ===")
    diag = s[(s.cell_kind == "diagonal") & (s.diagonal_protocol == protocol)][cols]
    diag = diag.sort_values(["source", "model"])
    for _, r in diag.iterrows():
        print(
            f"{r['model']:22s} {r['source']}: "
            f"PR={r['pr_auc_mean']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}]  "
            f"lift={r['lift_mean']:.2f}"
        )

print("\n=== GNN-RF DELTAS ===")
print(d.to_string(index=False))

for src, tgt in [("cert", "spedia"), ("spedia", "cert")]:
    print(f"\nSeeds {src}->{tgt}:")
    for model in ["day_gnn_zero_shot", "random_forest"]:
        g = det[
            (det.model == model)
            & (det.source == src)
            & (det.target == tgt)
            & (det.cell_kind == "off_diagonal")
        ].sort_values("seed")
        print(
            model,
            [(int(se), round(float(pr), 4), round(float(lf), 2))
             for se, pr, lf in zip(g.seed, g.pr_auc, g.lift)],
        )
