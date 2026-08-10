"""
READ-ONLY manuscript number exporter.

Maps every paper placeholder token to a value computed from existing
``results/*.csv`` files. Does not train models, re-run seeds, or modify
any source artefacts other than writing ``results/paper_numbers.csv``.

Usage (from repo root):
  python -m scripts.export_paper_numbers
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.multiseed_transfer import paired_delta_ci  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("paper_numbers")

OUT = ROOT / "results"
OUT_CSV = OUT / "paper_numbers.csv"

# Files we attempt to read (skip missing gracefully).
SOURCE_FILES = {
    "transfer": OUT / "transfer_matrix_phaseA_checkpoint.csv",
    # 3-domain multiseed (cert42/cert52/spedia) — GNN user-disjoint diagonals
    "multiseed_core": OUT / "multiseed_releases_details_core.csv",
    "gap_summary": OUT / "domain_gap_summary.csv",
    "gap_5d": OUT / "domain_gap_5domain.csv",
    "ablation": OUT / "ablation_into_spedia.csv",
    "xai": OUT / "xai_transfer_diagnostic.csv",
    "threshold": OUT / "spedia_threshold_sensitivity.csv",
    "efficiency": OUT / "efficiency.csv",
}

# Paper "Temporal PR-AUC" = day_gnn user-disjoint ID score (NOT temporal-split).
# Expected means for the 3-domain core run; mismatch → abort rather than emit.
TEMP_EXPECTED = {
    "cert42": 0.890,
    "cert52": 0.854,
    "spedia": 0.828,
}
TEMP_EXPECTED_TOL = 0.0015  # allow float noise around 3-dp targets
TEMP_MIN_SEEDS = 5

DOMAINS = ("cert42", "cert52", "cert62", "spedia", "lanl")
# M: matrix targets = full 5×5 minus cert62-as-target
M_TARGETS = ("cert42", "cert52", "spedia", "lanl")

CORE_C2_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("cert52", "spedia"),
    ("cert42", "spedia"),
    ("spedia", "cert42"),
    ("spedia", "cert52"),
    ("cert52", "cert42"),
    ("cert42", "cert52"),
)

RF_MODEL = "random_forest"
GNN_MODEL = "day_gnn_zero_shot"

# domain_gap uses cert_r42; transfer matrix uses cert42
GAP_TO_PAPER = {
    "cert_r42": "cert42",
    "cert_r52": "cert52",
    "cert_r62": "cert62",
    "spedia": "spedia",
    "lanl": "lanl",
}
PAPER_TO_GAP = {v: k for k, v in GAP_TO_PAPER.items()}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def fmt_pr(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_lift(x: float) -> str:
    return f"{int(round(float(x)))}x"


def fmt_base(x: float) -> str:
    """Two significant figures."""
    return f"{float(x):.2g}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4g}"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_csv(key: str) -> Optional[pd.DataFrame]:
    path = SOURCE_FILES[key]
    if not path.exists():
        log.warning("ABSENT: %s", path.name)
        return None
    df = pd.read_csv(path)
    log.info("Loaded %s  rows=%d cols=%d", path.name, len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# helpers on transfer checkpoint
# ---------------------------------------------------------------------------
def _seed_means(
    df: pd.DataFrame,
    *,
    model: str,
    cell_kind: str,
    protocol: Optional[str] = None,
) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["cell_kind"] == cell_kind)].copy()
    if protocol is not None:
        sub = sub[sub["diagonal_protocol"] == protocol]
    if sub.empty:
        return sub
    return (
        sub.groupby(["source", "target"], as_index=False)
        .agg(pr_auc=("pr_auc", "mean"), lift=("lift", "mean"),
             base_rate=("base_rate", "mean"), n_seeds=("seed", "nunique"))
    )


def _cell_seed_series(
    df: pd.DataFrame,
    source: str,
    target: str,
    model: str,
    *,
    cell_kind: str = "off_diagonal",
    protocol: Optional[str] = None,
) -> pd.Series:
    sub = df[
        (df["source"] == source)
        & (df["target"] == target)
        & (df["model"] == model)
        & (df["cell_kind"] == cell_kind)
    ]
    if protocol is not None:
        sub = sub[sub["diagonal_protocol"] == protocol]
    return sub.sort_values("seed").set_index("seed")["pr_auc"]


def hold_label(d_pr: float, lo: float, hi: float) -> str:
    """Yes / No / RF from paired ΔPR CI."""
    if not np.isfinite(d_pr):
        return "No"
    if d_pr > 0 and lo > 0:
        return "Yes"
    if d_pr <= 0:
        return "RF"
    return "No"  # positive point estimate but CI includes 0


# ---------------------------------------------------------------------------
# token registry
# ---------------------------------------------------------------------------
class TokenTable:
    def __init__(self):
        self.rows: List[Dict[str, str]] = []
        self.expected: List[str] = []

    def expect(self, *tokens: str):
        self.expected.extend(tokens)

    def put(self, token: str, value: Any, source_file: str, note: str = ""):
        self.rows.append({
            "token": token,
            "value": "" if value is None else str(value),
            "source_file": source_file,
            "note": note or "",
        })

    def resolved_tokens(self) -> List[str]:
        return [r["token"] for r in self.rows if r["value"] != ""]

    def missing_expected(self) -> List[str]:
        have = {r["token"] for r in self.rows if r["value"] != ""}
        return [t for t in self.expected if t not in have]


# ---------------------------------------------------------------------------
# D: in-domain diagonals
# ---------------------------------------------------------------------------
def _gnn_ud_diag_means(df: pd.DataFrame) -> pd.DataFrame:
    """day_gnn_zero_shot user-disjoint diagonal seed-means (source==target)."""
    m = _seed_means(
        df, model=GNN_MODEL, cell_kind="diagonal", protocol="user_disjoint",
    )
    if m.empty:
        return m
    return m[m["source"] == m["target"]].copy()


def _checkpoint_has_full_gnn_ud_diag(transfer: pd.DataFrame, domains: Sequence[str]) -> bool:
    """True iff every named domain has ≥ TEMP_MIN_SEEDS GNN UD-diag seeds."""
    gnn = _gnn_ud_diag_means(transfer)
    if gnn.empty:
        return False
    for dom in domains:
        row = gnn[gnn["source"] == dom]
        if row.empty or int(row.iloc[0]["n_seeds"]) < TEMP_MIN_SEEDS:
            return False
    return True


def emit_D(
    tt: TokenTable,
    transfer: Optional[pd.DataFrame],
    multiseed_core: Optional[pd.DataFrame] = None,
):
    for dom in DOMAINS:
        tt.expect(
            f"D:{dom}-rf", f"D:{dom}-lift", f"D:{dom}-base", f"D:{dom}-temp",
        )
    if transfer is None:
        return
    src_name = SOURCE_FILES["transfer"].name
    ud = _seed_means(
        transfer, model=RF_MODEL, cell_kind="diagonal", protocol="user_disjoint",
    )
    for dom in DOMAINS:
        row = ud[(ud["source"] == dom) & (ud["target"] == dom)]
        if row.empty:
            continue
        r = row.iloc[0]
        pr = float(r["pr_auc"])
        lift = float(r["lift"])
        # base ≡ PR-AUC/lift (stored per-seed as base_rate; mean over seeds)
        base = float(r["base_rate"])
        tt.put(f"D:{dom}-rf", fmt_pr(pr), src_name,
               note=f"RF user_disjoint diagonal mean over {int(r['n_seeds'])} seeds")
        tt.put(f"D:{dom}-lift", fmt_lift(lift), src_name,
               note="RF user_disjoint diagonal lift")
        tt.put(f"D:{dom}-base", fmt_base(base), src_name,
               note="base rate (= PR-AUC/lift) from RF user_disjoint diagonal")

    # D:*-temp = day_gnn USER-DISJOINT diagonal (paper "Temporal PR-AUC" column
    # = temporal-GNN in-distribution score, NOT a temporal data split).
    core_doms = tuple(TEMP_EXPECTED.keys())  # cert42, cert52, spedia
    if _checkpoint_has_full_gnn_ud_diag(transfer, core_doms):
        gnn_src_df = transfer
        gnn_src_name = SOURCE_FILES["transfer"].name
        log.info(
            "D:*-temp: using %s (full ≥%d-seed GNN user_disjoint diagonals)",
            gnn_src_name, TEMP_MIN_SEEDS,
        )
    else:
        if multiseed_core is None:
            log.warning(
                "D:*-temp: checkpoint lacks full GNN user_disjoint diagonals "
                "for %s and multiseed_core is absent — leaving MISSING",
                list(core_doms),
            )
            gnn_src_df = None
            gnn_src_name = ""
        else:
            gnn_src_df = multiseed_core
            gnn_src_name = SOURCE_FILES["multiseed_core"].name
            log.info(
                "D:*-temp: checkpoint incomplete for GNN UD diags; "
                "falling back to %s",
                gnn_src_name,
            )

    if gnn_src_df is None:
        return

    gnn_ud = _gnn_ud_diag_means(gnn_src_df)
    mismatches: List[str] = []
    pending: List[Tuple[str, float, int, str]] = []  # dom, pr, n_seeds, file

    for dom in DOMAINS:
        row = gnn_ud[gnn_ud["source"] == dom]
        if row.empty:
            continue
        r = row.iloc[0]
        pr = float(r["pr_auc"])
        n_seeds = int(r["n_seeds"])
        if dom in TEMP_EXPECTED:
            exp = TEMP_EXPECTED[dom]
            if abs(pr - exp) > TEMP_EXPECTED_TOL:
                mismatches.append(
                    f"D:{dom}-temp raw={pr:.6f} → {fmt_pr(pr)} "
                    f"expected≈{exp:.3f} (tol={TEMP_EXPECTED_TOL}) "
                    f"from {gnn_src_name} n_seeds={n_seeds}"
                )
                continue
        pending.append((dom, pr, n_seeds, gnn_src_name))

    if mismatches:
        msg = (
            "D:*-temp MISMATCH — refusing to emit. "
            "Paper Temporal PR-AUC expects day_gnn user_disjoint ID ≈ "
            f"{TEMP_EXPECTED}. Got:\n  " + "\n  ".join(mismatches)
        )
        log.error(msg)
        print("\n" + "=" * 72)
        print("STOP: D:*-temp value mismatch (not emitted)")
        print("=" * 72)
        for line in mismatches:
            print(f"  {line}")
        raise SystemExit(2)

    for dom, pr, n_seeds, file_name in pending:
        print(f"  D:{dom}-temp  <- {file_name}  PR={fmt_pr(pr)}  "
              f"(day_gnn user_disjoint diag, n_seeds={n_seeds})")
        tt.put(
            f"D:{dom}-temp", fmt_pr(pr), file_name,
            note=(
                f"day_gnn_zero_shot user_disjoint diagonal "
                f"(paper Temporal PR-AUC = temporal-GNN ID, not temporal-split); "
                f"n_seeds={n_seeds}; source={file_name}"
            ),
        )


# ---------------------------------------------------------------------------
# M: RF transfer matrix
# ---------------------------------------------------------------------------
def emit_M(tt: TokenTable, transfer: Optional[pd.DataFrame], d_rf: Dict[str, str]):
    for src in DOMAINS:
        for tgt in M_TARGETS:
            tt.expect(f"M:{src}-{tgt}")
    tt.expect("M:gap-aggregate")
    if transfer is None:
        return
    src_name = SOURCE_FILES["transfer"].name
    off = _seed_means(transfer, model=RF_MODEL, cell_kind="off_diagonal")

    for src in DOMAINS:
        for tgt in M_TARGETS:
            tok = f"M:{src}-{tgt}"
            if src == tgt:
                # copy diagonal from D:
                val = d_rf.get(f"D:{src}-rf")
                if val is None:
                    continue
                lift_tok = d_rf.get(f"D:{src}-lift", "")
                tt.put(
                    tok, val, src_name,
                    note=f"diagonal copied from D:{src}-rf; lift={lift_tok}",
                )
                continue
            row = off[(off["source"] == src) & (off["target"] == tgt)]
            if row.empty:
                continue
            r = row.iloc[0]
            tt.put(
                tok, fmt_pr(float(r["pr_auc"])), src_name,
                note=f"RF off-diag lift={fmt_lift(float(r['lift']))}; "
                     f"n_seeds={int(r['n_seeds'])}",
            )

    # gap-aggregate: mean(UD diag) − mean(off-diag), EXCLUDING cert62 diagonal
    ud = _seed_means(
        transfer, model=RF_MODEL, cell_kind="diagonal", protocol="user_disjoint",
    )
    diag = ud[(ud["source"] == ud["target"]) & (ud["source"] != "cert62")]
    # off-diag in the same M: scope (no cert62-as-target)
    off_scope = off[off["target"] != "cert62"]
    if not diag.empty and not off_scope.empty:
        gap = float(diag["pr_auc"].mean() - off_scope["pr_auc"].mean())
        tt.put(
            "M:gap-aggregate", fmt_pr(gap), src_name,
            note="transparency stat; mixes ~1000x base rates",
        )


# ---------------------------------------------------------------------------
# C2: core GNN vs RF pairs
# ---------------------------------------------------------------------------
def emit_C2(tt: TokenTable, transfer: Optional[pd.DataFrame]):
    for s, t in CORE_C2_PAIRS:
        pair = f"{s}-{t}"
        tt.expect(
            f"C2:{pair}-gnn", f"C2:{pair}-rf", f"C2:{pair}-dpr",
            f"C2:{pair}-ci", f"C2:{pair}-hold",
        )
    tt.expect("C2:wilcoxon-W", "C2:wilcoxon-p")
    if transfer is None:
        return
    src_name = SOURCE_FILES["transfer"].name

    paired_cells: List[Tuple[float, float]] = []  # (gnn_mean, rf_mean) per cell

    for s, t in CORE_C2_PAIRS:
        pair = f"{s}-{t}"
        g = _cell_seed_series(transfer, s, t, GNN_MODEL)
        r = _cell_seed_series(transfer, s, t, RF_MODEL)
        if g.empty or r.empty:
            continue
        merged = pd.concat([g.rename("gnn"), r.rename("rf")], axis=1).dropna()
        if merged.empty:
            continue
        g_m = float(merged["gnn"].mean())
        r_m = float(merged["rf"].mean())
        d_pr, lo, hi = paired_delta_ci(
            merged["gnn"], merged["rf"], n_boot=5000, seed=0,
        )
        hold = hold_label(d_pr, lo, hi)
        tt.put(f"C2:{pair}-gnn", fmt_pr(g_m), src_name,
               note=f"day_gnn_zero_shot; n_seeds={len(merged)}")
        tt.put(f"C2:{pair}-rf", fmt_pr(r_m), src_name,
               note=f"random_forest; n_seeds={len(merged)}")
        tt.put(f"C2:{pair}-dpr", f"{d_pr:+.3f}", src_name,
               note="mean(GNN)−mean(RF) over paired seeds")
        tt.put(f"C2:{pair}-ci", fmt_ci(lo, hi), src_name,
               note="paired bootstrap 5000 resamples, seed=0")
        tt.put(f"C2:{pair}-hold", hold, src_name,
               note="Yes: Δ>0 & CI>0; RF: Δ≤0; No: CI includes 0")
        paired_cells.append((g_m, r_m))

    # Wilcoxon over ALL available off-diag GNN-vs-RF paired cells (not just core 6)
    off = transfer[transfer["cell_kind"] == "off_diagonal"]
    gnn = (
        off[off["model"] == GNN_MODEL]
        .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
        .rename(columns={"pr_auc": "gnn"})
    )
    rf = (
        off[off["model"] == RF_MODEL]
        .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
        .rename(columns={"pr_auc": "rf"})
    )
    m = gnn.merge(rf, on=["source", "target"])
    if len(m) >= 2:
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(
                m["gnn"], m["rf"], alternative="greater", zero_method="wilcox",
            )
            tt.put(
                "C2:wilcoxon-W", fmt_float(float(stat), 3), src_name,
                note=f"signed-rank over {len(m)} off-diag GNN-vs-RF cells; "
                     f"n_gnn>rf={int((m['gnn'] > m['rf']).sum())}",
            )
            tt.put(
                "C2:wilcoxon-p", fmt_p(float(p)), src_name,
                note=f"alternative=greater; n_pairs={len(m)}",
            )
        except Exception as exc:
            log.warning("Wilcoxon failed: %s", exc)


# ---------------------------------------------------------------------------
# G: domain-gap metrics
# ---------------------------------------------------------------------------
def _gap_pair_token(a: str, b: str) -> str:
    return f"{GAP_TO_PAPER.get(a, a)}-{GAP_TO_PAPER.get(b, b)}"


def emit_G(
    tt: TokenTable,
    gap_summary: Optional[pd.DataFrame],
    gap_5d: Optional[pd.DataFrame],
):
    # Expected pairs = union of summary / 5d (all undirected CERT/SPEDIA/LANL pairs)
    expected_pairs: List[str] = []
    if gap_summary is not None and "pair" in gap_summary.columns:
        for p in gap_summary["pair"]:
            a, b = str(p).split("__", 1)
            expected_pairs.append(_gap_pair_token(a, b))
    elif gap_5d is not None:
        for _, r in gap_5d.iterrows():
            expected_pairs.append(_gap_pair_token(str(r["src"]), str(r["tgt"])))
    else:
        # hard-coded catalogue so MISSING list is informative
        expected_pairs = [
            "cert42-cert52", "cert42-cert62", "cert42-spedia", "cert42-lanl",
            "cert52-cert62", "cert52-spedia", "cert52-lanl",
            "cert62-spedia", "cert62-lanl", "spedia-lanl",
        ]
    for pair in expected_pairs:
        tt.expect(f"G:{pair}-lr", f"G:{pair}-da", f"G:{pair}-mmd")

    if gap_summary is None and gap_5d is None:
        return

    # Prefer numeric means from summary; fall back to 5domain for dA/MMD
    by_pair: Dict[str, Dict[str, float]] = {}
    src_lr = SOURCE_FILES["gap_summary"].name if gap_summary is not None else ""
    src_5d = SOURCE_FILES["gap_5d"].name if gap_5d is not None else ""

    if gap_summary is not None:
        for _, r in gap_summary.iterrows():
            a, b = str(r["pair"]).split("__", 1)
            tok = _gap_pair_token(a, b)
            by_pair[tok] = {
                "lr": float(r["auc_lr_mean"]),
                "da": float(r["da_mean"]),
                "mmd": float(r["mmd_mean"]),
            }
    if gap_5d is not None:
        for _, r in gap_5d.iterrows():
            tok = _gap_pair_token(str(r["src"]), str(r["tgt"]))
            by_pair.setdefault(tok, {})
            by_pair[tok].setdefault("da", float(r["dA"]))
            by_pair[tok].setdefault("mmd", float(r["mmd_rbf"]))

    for pair, vals in by_pair.items():
        if "lr" in vals:
            tt.put(
                f"G:{pair}-lr", fmt_pr(vals["lr"]), src_lr or src_5d,
                note="domain classifier LR-AUC (auc_lr_mean)",
            )
        if "da" in vals:
            src = src_lr if gap_summary is not None else src_5d
            tt.put(
                f"G:{pair}-da", fmt_float(vals["da"], 3), src,
                note="proxy A-distance d̂_A",
            )
        if "mmd" in vals:
            src = src_lr if gap_summary is not None else src_5d
            tt.put(
                f"G:{pair}-mmd", fmt_float(vals["mmd"], 3), src,
                note="MMD RBF (mmd_mean / mmd_rbf)",
            )


# ---------------------------------------------------------------------------
# R: correlation dA vs transfer lift
# ---------------------------------------------------------------------------
def emit_R(
    tt: TokenTable,
    transfer: Optional[pd.DataFrame],
    gap_summary: Optional[pd.DataFrame],
    gap_5d: Optional[pd.DataFrame],
):
    tt.expect("R:corr-r", "R:corr-p")
    if transfer is None:
        return
    # Build unordered dA lookup in paper names
    da: Dict[frozenset, float] = {}
    src_gap = ""
    if gap_summary is not None:
        src_gap = SOURCE_FILES["gap_summary"].name
        for _, r in gap_summary.iterrows():
            a, b = str(r["pair"]).split("__", 1)
            key = frozenset({GAP_TO_PAPER.get(a, a), GAP_TO_PAPER.get(b, b)})
            if len(key) == 2:
                da[key] = float(r["da_mean"])
    elif gap_5d is not None:
        src_gap = SOURCE_FILES["gap_5d"].name
        for _, r in gap_5d.iterrows():
            a = GAP_TO_PAPER.get(str(r["src"]), str(r["src"]))
            b = GAP_TO_PAPER.get(str(r["tgt"]), str(r["tgt"]))
            key = frozenset({a, b})
            if len(key) == 2:
                da[key] = float(r["dA"])
    if not da:
        return

    off = _seed_means(transfer, model=RF_MODEL, cell_kind="off_diagonal")
    xs, ys, labels = [], [], []
    for _, r in off.iterrows():
        key = frozenset({r["source"], r["target"]})
        if key not in da:
            continue
        xs.append(da[key])
        ys.append(float(r["lift"]))  # LIFT not PR-AUC
        labels.append(f"{r['source']}->{r['target']}")

    if len(xs) < 3:
        log.warning("R:corr needs ≥3 paired (dA, lift) points; got %d", len(xs))
        return

    from scipy.stats import pearsonr
    rho, p = pearsonr(xs, ys)
    has_spedia = any("spedia" in lab for lab in labels)
    note = (
        f"Pearson(symmetric dA, RF off-diag LIFT); n={len(xs)} directed cells; "
        f"sources={SOURCE_FILES['transfer'].name}+{src_gap}"
    )
    if has_spedia:
        note += "; spedia (n=256) point is high-variance"
    tt.put("R:corr-r", fmt_float(float(rho), 3), SOURCE_FILES["transfer"].name, note=note)
    tt.put("R:corr-p", fmt_p(float(p)), SOURCE_FILES["transfer"].name, note=note)


# ---------------------------------------------------------------------------
# Optional: ablation / XAI / threshold / efficiency
# ---------------------------------------------------------------------------
def emit_ablation(tt: TokenTable, abl: Optional[pd.DataFrame], transfer: Optional[pd.DataFrame]):
    if abl is None:
        # announce expected skeleton so MISSING is clear
        for s, t in (("cert42", "spedia"), ("cert52", "spedia")):
            for v in ("V0_full", "V1_-SSL", "V2_-memory", "V3_-timeenc", "V4_-deviation"):
                tt.expect(
                    f"A:{s}-{t}-{v}-pr",
                    f"A:{s}-{t}-{v}-dpr_full",
                    f"A:{s}-{t}-{v}-dpr_rf",
                )
        return

    src_name = SOURCE_FILES["ablation"].name
    # RF seed PRs from transfer checkpoint for dPR_vs_rf
    rf_prs: Dict[Tuple[str, str, int], float] = {}
    if transfer is not None:
        rf = transfer[
            (transfer["model"] == RF_MODEL)
            & (transfer["cell_kind"] == "off_diagonal")
        ]
        for _, r in rf.iterrows():
            rf_prs[(r["source"], r["target"], int(r["seed"]))] = float(r["pr_auc"])

    variants = list(dict.fromkeys(abl["variant"].tolist()))
    cells = (
        abl[["source", "target"]].drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for s, t in cells:
        v0 = abl[(abl["source"] == s) & (abl["target"] == t) & (abl["variant"] == "V0_full")]
        v0_by_seed = {int(r["seed"]): float(r["pr_auc"]) for _, r in v0.iterrows()}
        for variant in variants:
            sub = abl[
                (abl["source"] == s) & (abl["target"] == t) & (abl["variant"] == variant)
            ].sort_values("seed")
            if sub.empty:
                continue
            prefix = f"A:{s}-{t}-{variant}"
            tt.expect(f"{prefix}-pr", f"{prefix}-dpr_full", f"{prefix}-dpr_rf")
            pr_m = float(sub["pr_auc"].mean())
            tt.put(f"{prefix}-pr", fmt_pr(pr_m), src_name,
                   note=f"n_seeds={len(sub)}")

            if variant == "V0_full":
                d_full = d_full_lo = d_full_hi = 0.0
            else:
                pv, pf = [], []
                for _, r in sub.iterrows():
                    seed = int(r["seed"])
                    if seed in v0_by_seed:
                        pv.append(float(r["pr_auc"]))
                        pf.append(v0_by_seed[seed])
                d_full, d_full_lo, d_full_hi = paired_delta_ci(pv, pf, n_boot=5000, seed=0)
            tt.put(
                f"{prefix}-dpr_full", f"{d_full:+.3f}", src_name,
                note=f"CI {fmt_ci(d_full_lo, d_full_hi)}",
            )

            pg, prf = [], []
            for _, r in sub.iterrows():
                seed = int(r["seed"])
                rp = rf_prs.get((s, t, seed))
                if rp is not None:
                    pg.append(float(r["pr_auc"]))
                    prf.append(rp)
            if pg:
                d_rf, d_rf_lo, d_rf_hi = paired_delta_ci(pg, prf, n_boot=5000, seed=0)
                tt.put(
                    f"{prefix}-dpr_rf", f"{d_rf:+.3f}", src_name,
                    note=f"CI {fmt_ci(d_rf_lo, d_rf_hi)}",
                )


def emit_xai(tt: TokenTable, xai: Optional[pd.DataFrame]):
    # Expected for the two into-SPEDIA cells
    for s, t in (("cert52", "spedia"), ("cert42", "spedia")):
        tt.expect(f"X:{s}-{t}-txai", f"X:{s}-{t}-rho")
    if xai is None:
        return
    src_name = SOURCE_FILES["xai"].name
    need = {"cell", "feature", "alpha_src", "alpha_tgt"}
    if not need.issubset(set(xai.columns)):
        log.warning("xai CSV missing columns %s", need - set(xai.columns))
        return
    for cell, g in xai.groupby("cell"):
        a_s = g["alpha_src"].to_numpy(dtype=float)
        a_t = g["alpha_tgt"].to_numpy(dtype=float)
        t_xai = float(1.0 - 0.5 * np.abs(a_s - a_t).sum())
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(a_s, a_t)
            rho = float(rho)
        except Exception:
            rho = float("nan")
        # cell is "cert52->spedia" → token X:cert52-spedia-*
        pair = str(cell).replace("->", "-")
        tt.put(f"X:{pair}-txai", fmt_pr(t_xai), src_name,
               note="T_XAI = 1 − 0.5 Σ|α_src − α_tgt|")
        tt.put(f"X:{pair}-rho", fmt_float(rho, 3), src_name,
               note="Spearman(α_src, α_tgt)")


def emit_threshold(tt: TokenTable, thr: Optional[pd.DataFrame]):
    if thr is None:
        tt.expect("T:spedia-threshold-holds")
        return
    src_name = SOURCE_FILES["threshold"].name
    # Flexible: emit one token per row if schema is unknown but informative
    cols = {c.lower(): c for c in thr.columns}
    if "holds" in cols or "hold" in cols:
        hcol = cols.get("holds") or cols.get("hold")
        # majority / all-hold summary
        vals = thr[hcol]
        if vals.dtype == object:
            holds = vals.astype(str).str.lower().isin(("yes", "true", "1", "holds"))
        else:
            holds = vals.astype(bool)
        tt.put(
            "T:spedia-threshold-holds",
            "Yes" if bool(holds.all()) else ("Partial" if bool(holds.any()) else "No"),
            src_name,
            note=f"{int(holds.sum())}/{len(holds)} rows hold",
        )
    # Also dump numeric summary columns as T:<col>
    for c in thr.columns:
        if pd.api.types.is_numeric_dtype(thr[c]):
            tok = f"T:{c}"
            tt.expect(tok)
            tt.put(tok, fmt_float(float(thr[c].mean()), 3), src_name,
                   note="mean over threshold rows")


def emit_efficiency(tt: TokenTable, eff: Optional[pd.DataFrame]):
    if eff is None:
        tt.expect("E:wall_s", "E:peak_rss_gb")
        return
    src_name = SOURCE_FILES["efficiency"].name
    for c in eff.columns:
        if pd.api.types.is_numeric_dtype(eff[c]):
            tok = f"E:{c}"
            tt.expect(tok)
            tt.put(tok, fmt_float(float(eff[c].mean()), 3), src_name,
                   note="mean over efficiency rows")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("READ-ONLY paper-numbers export (no training, no re-runs)")
    transfer = load_csv("transfer")
    multiseed_core = load_csv("multiseed_core")
    gap_summary = load_csv("gap_summary")
    gap_5d = load_csv("gap_5d")
    abl = load_csv("ablation")
    xai = load_csv("xai")
    thr = load_csv("threshold")
    eff = load_csv("efficiency")

    tt = TokenTable()
    emit_D(tt, transfer, multiseed_core=multiseed_core)
    # Collect D:rf values for M: diagonal copy
    d_rf = {r["token"]: r["value"] for r in tt.rows if r["token"].startswith("D:")}
    # also need lifts for notes
    d_rf.update({r["token"]: r["value"] for r in tt.rows if r["token"].endswith("-lift")})

    emit_M(tt, transfer, d_rf)
    emit_C2(tt, transfer)
    emit_G(tt, gap_summary, gap_5d)
    emit_R(tt, transfer, gap_summary, gap_5d)
    emit_ablation(tt, abl, transfer)
    emit_xai(tt, xai)
    emit_threshold(tt, thr)
    emit_efficiency(tt, eff)

    out = pd.DataFrame(tt.rows, columns=["token", "value", "source_file", "note"])
    # Drop empty-valued rows from the CSV (they stay in MISSING via expect list)
    out_written = out[out["value"].astype(str).str.len() > 0].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    out_written.to_csv(OUT_CSV, index=False)
    log.info("Wrote %s (%d tokens)", OUT_CSV, len(out_written))

    resolved = tt.resolved_tokens()
    missing = tt.missing_expected()

    print()
    print("=" * 72)
    print(f"RESOLVED ({len(resolved)})")
    print("=" * 72)
    for tok in resolved:
        row = next(r for r in tt.rows if r["token"] == tok)
        print(f"  {tok:40s}  = {row['value']}")

    print()
    print("=" * 72)
    print(f"MISSING ({len(missing)}) — source CSV absent or cell not yet run")
    print("=" * 72)
    if not missing:
        print("  (none)")
    else:
        for tok in missing:
            print(f"  {tok}")

    print()
    print(f"Wrote {OUT_CSV} ({len(out_written)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
