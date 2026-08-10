"""Salvage completed RF/GNN cells from a multiseed_releases run log into a CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LOG = Path("results/multiseed_releases_core_run.log")
# Prefer terminal capture if Tee truncated prints
TERM = Path(
    r"C:\Users\User\.cursor\projects"
    r"\c-PhD-04-Journal-Papers-New-Paper-Aug-2026-itd-crossdataset-1-itd-crossdataset"
    r"\terminals\665592.txt"
)
OUT = Path("results/multiseed_releases_details_partial_core.csv")

# random_forest seed=0 cert42->cert52: pr_auc=0.5941 lift=...
# day_gnn_zero_shot seed=0 cert42->cert52: pr_auc=0.4035 lift=...
# random_forest seed=0 cert42->cert42 [temporal]: pr_auc=...
PAT = re.compile(
    r"(?P<model>random_forest|day_gnn_zero_shot) seed=(?P<seed>\d+) "
    r"(?P<source>\w+)->(?P<target>\w+)"
    r"(?: \[(?P<protocol>temporal|user_disjoint)\])?"
    r": pr_auc=(?P<pr>[0-9.]+)\s+lift=(?P<lift>[0-9.]+)"
)

BASE = {
    "cert42": 0.0029232687349448633,
    "cert52": 0.001884238281897617,
    "spedia": 0.3203125,
}


# Tee/PowerShell wraps long lines; join a result header with following pr_auc line.
HEADER = re.compile(
    r"(?P<model>random_forest|day_gnn_zero_shot) seed=(?P<seed>\d+) "
    r"(?P<source>\w+)->(?P<target>\w+)"
    r"(?: \[(?P<protocol>temporal|user_disjoint)\])?:\s*$"
)
METRICS = re.compile(r"pr_auc=(?P<pr>[0-9.]+)\s+lift=(?P<lift>[0-9.]+)")
INLINE = re.compile(
    r"(?P<model>random_forest|day_gnn_zero_shot) seed=(?P<seed>\d+) "
    r"(?P<source>\w+)->(?P<target>\w+)"
    r"(?: \[(?P<protocol>temporal|user_disjoint)\])?"
    r":\s*pr_auc=(?P<pr>[0-9.]+)\s+lift=(?P<lift>[0-9.]+)"
)


def parse(text: str) -> list[dict]:
    rows = []
    # Collapse wrapped "header\npr_auc=..." into one line
    text2 = re.sub(
        r"((?:random_forest|day_gnn_zero_shot) seed=\d+ \w+->\w+"
        r"(?: \[(?:temporal|user_disjoint)\])?:\s*)\n\s*(pr_auc=)",
        r"\1\2",
        text,
    )
    for m in INLINE.finditer(text2):
        src, tgt = m.group("source"), m.group("target")
        proto = m.group("protocol")
        if src == tgt:
            if proto is None:
                continue
            cell_kind = "diagonal"
            diag = proto
        else:
            cell_kind = "off_diagonal"
            diag = "full_source"
        pr = float(m.group("pr"))
        br = BASE[tgt]
        lift = float(m.group("lift"))
        rows.append({
            "source": src,
            "target": tgt,
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "diagonal_protocol": diag,
            "cell_kind": cell_kind,
            "pr_auc": pr,
            "lift": lift,
            "base_rate": br,
            "roc_auc": float("nan"),
            "n": float("nan"),
            "n_pos": float("nan"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df = df.drop_duplicates(
        subset=["model", "seed", "source", "target", "diagonal_protocol"],
        keep="last",
    )
    return df.to_dict(orient="records")


def main():
    texts = []
    for p in (LOG, TERM):
        if p.exists():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
            print(f"read {p} ({len(texts[-1])} chars)")
    text = "\n".join(texts)
    rows = parse(text)
    df = pd.DataFrame(rows)
    print(f"salvaged {len(df)} cells")
    if not df.empty:
        print(df.groupby(["model", "seed"]).size().to_string())
        # expected complete seed: 6 off + 6 diag = 12
        for (model, seed), g in df.groupby(["model", "seed"]):
            n = len(g)
            status = "COMPLETE" if n >= 12 else f"PARTIAL ({n}/12)"
            print(f"  {model} seed={seed}: {status}")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
