"""Finalize / recover multiseed_releases results from checkpoint + log.

Durable against flaky agent connections:
  - merges checkpoint CSV with cells parsed from the training log
  - writes continuous details CSV
  - when RF + day_gnn cover seeds 0..4 for all cells, writes summary/deltas/verdict

Usage:
  python -m scripts.finalize_multiseed_releases --tag core
  python -m scripts.finalize_multiseed_releases --tag core --log results/multiseed_releases_core_resume.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.train.multiseed_transfer import (
    aggregate_seed_rows,
    gnn_minus_rf_table,
    print_confirmation_report,
)

OUT = ROOT / "results"

# PowerShell tee often wraps "pr_auc=... lift=..." across lines — allow \s+.
# day_gnn_zero_shot seed=4 cert42->cert52: pr_auc=0.4681 lift=248.42
# day_gnn_zero_shot seed=4 cert42->cert42 [temporal]: pr_auc=0.9373 lift=290.72
_OFF = re.compile(
    r"(?P<model>[\w/]+)\s+seed=(?P<seed>\d+)\s+"
    r"(?P<source>\w+)->(?P<target>\w+):\s+"
    r"pr_auc=(?P<pr>[\d.]+)(?:\s+lift=(?P<lift>[\d.]+))?",
    re.MULTILINE,
)
_DIAG = re.compile(
    r"(?P<model>[\w/]+)\s+seed=(?P<seed>\d+)\s+"
    r"(?P<source>\w+)->(?P<target>\w+)\s+\[(?P<proto>\w+)\]:\s+"
    r"pr_auc=(?P<pr>[\d.]+)(?:\s+lift=(?P<lift>[\d.]+))?",
    re.MULTILINE,
)
_LIFT_ONLY = re.compile(r"^\s*lift=(?P<lift>[\d.]+)\s*$", re.MULTILINE)

BASE_RATES = {
    "cert42": 0.0029232687349448633,
    "cert52": 0.001884238281897617,
    "spedia": 0.3203125,
}

DOMAINS = ["cert42", "cert52", "spedia"]
MODELS = ["random_forest", "day_gnn_zero_shot"]
SEEDS = [0, 1, 2, 3, 4]


def _expected_cells():
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for s in DOMAINS:
                for t in DOMAINS:
                    if s == t:
                        continue
                    rows.append((model, seed, s, t, "full_source", "off_diagonal"))
            for proto in ("temporal", "user_disjoint"):
                for s in DOMAINS:
                    rows.append((model, seed, s, s, proto, "diagonal"))
    return rows


def _cell_key(r):
    return (
        str(r["model"]),
        int(r["seed"]),
        str(r["source"]),
        str(r["target"]),
        str(r["diagonal_protocol"]),
    )


def _read_text_shared(path: Path) -> str:
    """Read a possibly-locked log; handle PowerShell UTF-16 redirects."""
    import os

    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        parts = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            parts.append(chunk)
        raw = b"".join(parts)
    finally:
        os.close(fd)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    # Heuristic: high NUL ratio ⇒ UTF-16 without reliable BOM handling
    if raw and (raw.count(0) / len(raw)) > 0.3:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Collapse PowerShell soft-wraps: join orphan "lift=..." onto previous line.
    raw_lines = _read_text_shared(path).splitlines()
    lines = []
    for ln in raw_lines:
        if _LIFT_ONLY.match(ln) and lines:
            lines[-1] = lines[-1].rstrip() + " " + ln.strip()
        else:
            lines.append(ln)
    text = "\n".join(lines)
    rows = []

    def _row(m, proto: str, kind: str):
        src, tgt = m.group("source"), m.group("target")
        br = BASE_RATES.get(tgt, float("nan"))
        pr = float(m.group("pr"))
        lift_s = m.group("lift")
        lift = float(lift_s) if lift_s is not None else (
            pr / br if br and br > 0 else float("nan")
        )
        return {
            "source": src,
            "target": tgt,
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "diagonal_protocol": proto,
            "cell_kind": kind,
            "pr_auc": pr,
            "lift": lift,
            "base_rate": br,
            "roc_auc": None,
            "n": None,
            "n_pos": None,
        }

    for m in _DIAG.finditer(text):
        src, tgt = m.group("source"), m.group("target")
        kind = "diagonal" if src == tgt else "off_diagonal"
        rows.append(_row(m, m.group("proto"), kind))
    for m in _OFF.finditer(text):
        src, tgt = m.group("source"), m.group("target")
        if src == tgt:
            continue
        # Avoid double-count if a diag-ish line somehow matched
        window = text[m.start(): min(len(text), m.end() + 40)]
        if re.search(r"\[\w+\]", window):
            continue
        rows.append(_row(m, "full_source", "off_diagonal"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep last occurrence per cell (log may contain resume restarts)
    df = df.drop_duplicates(
        subset=["model", "seed", "source", "target", "diagonal_protocol"],
        keep="last",
    )
    return df


def merge_details(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and len(f)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # Prefer checkpoint/CSV rows (may have roc_auc/n) over log-parsed
    df["_prio"] = df["roc_auc"].notna().astype(int) if "roc_auc" in df.columns else 0
    df = df.sort_values("_prio")
    df = df.drop_duplicates(
        subset=["model", "seed", "source", "target", "diagonal_protocol"],
        keep="last",
    ).drop(columns=["_prio"], errors="ignore")
    return df.reset_index(drop=True)


def missing_cells(details: pd.DataFrame):
    have = {_cell_key(r) for _, r in details.iterrows()} if len(details) else set()
    miss = []
    for model, seed, s, t, proto, kind in _expected_cells():
        k = (model, seed, s, t, proto)
        if k not in have:
            miss.append({
                "model": model, "seed": seed, "source": s, "target": t,
                "diagonal_protocol": proto, "cell_kind": kind,
            })
    return pd.DataFrame(miss)


def _save_matrices(summary: pd.DataFrame, tag: str):
    for model in summary["model"].unique():
        for protocol in ("temporal", "user_disjoint"):
            pr = pd.DataFrame(index=DOMAINS, columns=DOMAINS, dtype=float)
            lf = pd.DataFrame(index=DOMAINS, columns=DOMAINS, dtype=float)
            sub = summary[summary["model"] == model]
            for s in DOMAINS:
                for t in DOMAINS:
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
            safe = str(model).replace("/", "_")
            pr.to_csv(OUT / f"multiseed_releases_pr_auc_{safe}_{protocol}_{tag}.csv")
            lf.to_csv(OUT / f"multiseed_releases_lift_{safe}_{protocol}_{tag}.csv")


def _print_hold_count(deltas: pd.DataFrame):
    off = deltas[deltas["cell_kind"] == "off_diagonal"].copy()
    n = len(off)
    holds = int(off["claim_holds"].fillna(False).astype(bool).sum())
    print("\n======== OFF-DIAGONAL HOLD COUNT (day_gnn beats RF, CI excludes 0) ========")
    print(f"  {holds} / {n} off-diagonal cells HOLD")
    for _, r in off.sort_values(["source", "target"]).iterrows():
        status = "HOLDS" if r["claim_holds"] else "FAILS"
        print(
            f"  {r['source']}->{r['target']}: {status}  "
            f"GNN={r['gnn_pr_mean']:.3f} RF={r['rf_pr_mean']:.3f}  "
            f"ΔPR={r['delta_pr_mean']:.3f} "
            f"[{r['delta_pr_lo']:.3f}, {r['delta_pr_hi']:.3f}]"
        )
    return holds, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="core")
    ap.add_argument(
        "--log",
        default="",
        help="training log to parse for cells not yet checkpointed",
    )
    ap.add_argument(
        "--details",
        default="",
        help="primary details/checkpoint CSV (default: checkpoint then partial)",
    )
    args = ap.parse_args()
    tag = args.tag
    OUT.mkdir(exist_ok=True)

    candidates = []
    if args.details.strip():
        candidates.append(Path(args.details))
    candidates.extend([
        OUT / f"multiseed_releases_details_checkpoint_{tag}.csv",
        OUT / f"multiseed_releases_details_partial_{tag}.csv",
        OUT / f"multiseed_releases_details_{tag}.csv",
    ])
    frames = []
    for p in candidates:
        if p.exists():
            print(f"[finalize] load {p} ({sum(1 for _ in open(p, encoding='utf-8')) - 1} rows)")
            frames.append(pd.read_csv(p))

    log_path = Path(args.log) if args.log.strip() else OUT / f"multiseed_releases_{tag}_resume.log"
    if not log_path.exists():
        alt = OUT / f"multiseed_releases_{tag}_run.log"
        if alt.exists():
            log_path = alt
    log_df = parse_log(log_path)
    if len(log_df):
        print(f"[finalize] parsed {len(log_df)} unique cells from {log_path}")
        frames.append(log_df)

    details = merge_details(*frames)
    cont = OUT / f"multiseed_releases_details_continuous_{tag}.csv"
    details.to_csv(cont, index=False)
    print(f"[finalize] continuous details -> {cont} ({len(details)} rows)")

    miss = missing_cells(details)
    print(f"[finalize] missing cells: {len(miss)} / {len(_expected_cells())}")
    if len(miss):
        miss.to_csv(OUT / f"multiseed_releases_missing_{tag}.csv", index=False)
        print(miss.to_string(index=False))
        # Still write partial summary if useful
        if len(details) == 0:
            return
        print("\n[finalize] incomplete — not writing final summary/deltas yet.")
        # Partial off-diag preview when both models have rows
        try:
            summary = aggregate_seed_rows(details)
            deltas = gnn_minus_rf_table(summary, details)
            off = deltas[deltas["cell_kind"] == "off_diagonal"]
            if len(off):
                print("\n======== PARTIAL OFF-DIAGONAL DELTAS ========")
                print(off.sort_values(["source", "target"]).to_string(index=False))
        except Exception as e:
            print(f"[finalize] partial aggregate skipped: {e}")
        return

    summary = aggregate_seed_rows(details)
    deltas = gnn_minus_rf_table(summary, details)
    verdict_rows = []
    off = deltas[deltas["cell_kind"] == "off_diagonal"]
    for protocol in ("temporal", "user_disjoint"):
        for _, r in off.iterrows():
            verdict_rows.append({
                "report_protocol": protocol,
                "source": r["source"],
                "target": r["target"],
                "gnn_pr_mean": r["gnn_pr_mean"],
                "rf_pr_mean": r["rf_pr_mean"],
                "delta_pr_mean": r["delta_pr_mean"],
                "delta_pr_lo": r["delta_pr_lo"],
                "delta_pr_hi": r["delta_pr_hi"],
                "claim_holds": r["claim_holds"],
                "n_seeds": r["n_seeds"],
            })
    verdict = pd.DataFrame(verdict_rows)

    details.to_csv(OUT / f"multiseed_releases_details_{tag}.csv", index=False)
    summary.to_csv(OUT / f"multiseed_releases_summary_{tag}.csv", index=False)
    deltas.to_csv(OUT / f"multiseed_releases_deltas_{tag}.csv", index=False)
    verdict.to_csv(OUT / f"multiseed_releases_verdict_{tag}.csv", index=False)
    _save_matrices(summary, tag)

    results = {
        "details": details,
        "summary": summary,
        "deltas": deltas,
        "verdict": verdict,
        "target_stats": pd.read_csv(OUT / f"multiseed_releases_target_stats_{tag}.csv")
        if (OUT / f"multiseed_releases_target_stats_{tag}.csv").exists()
        else pd.DataFrame(),
    }
    print_confirmation_report(results)
    holds, n = _print_hold_count(deltas)
    print(f"\nSaved CSVs under {OUT}/multiseed_releases_*_{tag}.csv")
    print(f"SUMMARY: day_gnn beats RF with CI excluding 0 on {holds}/{n} off-diagonal cells.")


if __name__ == "__main__":
    main()
